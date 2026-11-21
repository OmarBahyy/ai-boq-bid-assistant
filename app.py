import asyncio
import io
import json
import os
import re
from collections import Counter
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from pypdf import PdfReader

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-5-20250929"
BATCH_SIZE = 35
MAX_CONCURRENT_REQUESTS = 5

st.set_page_config(
    page_title="GAEB Bid Assistant | GU-Kalkulation",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==============================================================================
# 1. DETERMINISTIC PARSERS (GAEB XML Primary + Lightweight PDF Fallback)
# ==============================================================================
def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_clean_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def parse_gaeb_xml(file_bytes: bytes) -> dict:
    """Deterministically parses GAEB DA XML (.X81, .X83, .X84, .X86) v3.1–3.3."""
    xml_text = None
    for enc in ("utf-8-sig", "utf-8", "iso-8859-1", "cp1252"):
        try:
            xml_text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if xml_text is None:
        raise ValueError("Datei-Encoding konnte nicht gelesen werden.")

    root = ET.fromstring(xml_text)

    project_name = "Unbekanntes Bauvorhaben"
    vorbemerkungen = []
    for elem in root.iter():
        tag = strip_ns(elem.tag)
        if tag in ("NamePrj", "LblBoQ") and elem.text and elem.text.strip():
            if project_name == "Unbekanntes Bauvorhaben":
                project_name = elem.text.strip()
        elif tag == "DetailAddText":
            txt = extract_clean_text(elem)
            if len(txt) > 30:
                vorbemerkungen.append(txt[:1500])

    items = []
    optional_count = 0
    lumpsum_count = 0

    def walk_boq(node: ET.Element, prefix_rno: str = "", category_title: str = ""):
        nonlocal optional_count, lumpsum_count
        for child in node:
            tag = strip_ns(child.tag)

            if tag == "BoQCtgy":
                rno_part = child.attrib.get("RNoPart", "")
                new_prefix = f"{prefix_rno}.{rno_part}".strip(".") if rno_part else prefix_rno
                lbl_text = category_title
                for sub in child:
                    if strip_ns(sub.tag) == "LblTx":
                        extracted = extract_clean_text(sub)
                        if extracted:
                            lbl_text = extracted
                            break
                walk_boq(child, new_prefix, lbl_text)

            elif tag in ("Itemlist", "BoQBody", "BoQ", "Award"):
                walk_boq(child, prefix_rno, category_title)

            elif tag == "Item":
                rno_part = child.attrib.get("RNoPart", "")
                oz = f"{prefix_rno}.{rno_part}".strip(".") if rno_part else f"POS-{len(items)+1}"

                qty, unit, kurztext, langtext = "0", "-", "", ""
                is_optional, is_lumpsum = False, False

                for sub in child.iter():
                    sub_tag = strip_ns(sub.tag)
                    if sub_tag == "Qty" and sub.text:
                        qty = sub.text.strip()
                    elif sub_tag == "QU" and sub.text:
                        unit = sub.text.strip()
                    elif sub_tag == "OutlineText":
                        kurztext = extract_clean_text(sub)
                    elif sub_tag == "CompleteText":
                        langtext = extract_clean_text(sub)
                    elif sub_tag in ("Provis", "Altn"):
                        is_optional = True
                    elif sub_tag == "LumpSumItem" and sub.text and sub.text.strip().lower() == "yes":
                        is_lumpsum = True

                if not kurztext and langtext:
                    kurztext = langtext[:110] + ("..." if len(langtext) > 110 else "")
                if unit.lower() in ("psch", "psch.", "pau", "pauschal"):
                    is_lumpsum = True

                if is_optional:
                    optional_count += 1
                if is_lumpsum:
                    lumpsum_count += 1

                items.append({
                    "oz": oz,
                    "category": category_title or "Allgemein",
                    "qty": qty,
                    "unit": unit,
                    "is_optional": is_optional,
                    "is_lumpsum": is_lumpsum,
                    "kurztext": kurztext,
                    "langtext": langtext[:900],
                })

    walk_boq(root)
    if not items:
        raise ValueError("Keine LV-Positionen (<Item>) in dieser GAEB-Datei gefunden.")

    return {
        "project_name": project_name,
        "source_type": "GAEB XML",
        "vorbemerkungen": "\n\n".join(vorbemerkungen)[:4000],
        "total_items": len(items),
        "optional_items": optional_count,
        "lumpsum_items": lumpsum_count,
        "items": items,
    }


def parse_pdf_fallback(file_bytes: bytes, filename: str) -> dict:
    """Lightweight PDF fallback: chunks pages into analyzable text blocks."""
    reader = PdfReader(io.BytesIO(file_bytes))
    items = []
    for idx, page in enumerate(reader.pages):
        text = re.sub(r"\s+", " ", (page.extract_text() or "")).strip()
        if len(text) < 40:
            continue
        items.append({
            "oz": f"Seite {idx + 1}",
            "category": "PDF-Auszug",
            "qty": "-",
            "unit": "-",
            "is_optional": "bedarfsposition" in text.lower(),
            "is_lumpsum": "psch" in text.lower() or "pauschal" in text.lower(),
            "kurztext": text[:110] + "...",
            "langtext": text[:1800],
        })
    if not items:
        raise ValueError("Kein lesbarer Text in der PDF gefunden.")
    return {
        "project_name": filename,
        "source_type": "PDF (Fallback)",
        "vorbemerkungen": "",
        "total_items": len(items),
        "optional_items": sum(1 for i in items if i["is_optional"]),
        "lumpsum_items": sum(1 for i in items if i["is_lumpsum"]),
        "items": items,
    }


# ==============================================================================
# 2. CLAUDE AI PIPELINE (Structured Outputs + Async Batching)
# ==============================================================================
ITEM_ANALYSIS_TOOL = {
    "name": "record_boq_analysis",
    "description": "Speichert Fehler-, Risiko- und Nachtragsanalyse für LV-Positionen.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "oz": {"type": "string", "description": "Exakte OZ aus dem Input"},
                        "trade": {
                            "type": "string",
                            "description": "Gewerk (z.B. Baustelleneinrichtung, Tiefbau/Erdbau, Rohbau/Beton, Rohrleitungsbau, Anlagenbau/TGA, Elektro, Ausbau/Trockenbau, Dach/Abdichtung, Sonstiges)"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                            "description": "Nur echte Kosten-/Fehler-Treiber als HIGH markieren."
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "FEHLER_WIDERSPRUCH",
                                "VOB_RISIKOFALLE",
                                "NACHTRAGSPOTENZIAL",
                                "UNKLARE_SPEZIFIKATION",
                                "OK"
                            ],
                            "description": "Hauptkategorie des Befunds"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Prägnante Begründung auf Deutsch (max. 20 Wörter). Leer bei OK/LOW."
                        },
                        "bieterfrage": {
                            "type": "string",
                            "description": "Formelle Bieterfrage an den AG bei HIGH oder FEHLER_WIDERSPRUCH, sonst leer."
                        }
                    },
                    "required": ["oz", "trade", "severity", "issue_type", "reason", "bieterfrage"]
                }
            }
        },
        "required": ["results"]
    }
}


async def analyze_batch_haiku(
    client: AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    batch: list[dict],
) -> list[dict]:
    payload = [
        {
            "oz": it["oz"],
            "titel": it["category"],
            "menge": f"{it['qty']} {it['unit']}",
            "bedarfspos": it["is_optional"],
            "pauschal": it["is_lumpsum"],
            "kurztext": it["kurztext"],
            "langtext": it["langtext"],
        }
        for it in batch
    ]

    system_prompt = (
        "Du bist ein Senior-Kalkulator bei einem deutschen Generalunternehmer (Bieter). "
        "Prüfe die LV-Positionen auf:\n"
        "1. FEHLER_WIDERSPRUCH: Widersprüche zwischen Kurztext und Langtext (z.B. verschiedene Maße/Höhen), "
        "physikalisch unmögliche Angaben/Einheiten, Tippfehler mit Kostenfolge, Copy-Paste-Fehler.\n"
        "2. VOB_RISIKOFALLE: Unvergütete Nebenleistungen ('ohne gesonderte Vergütung', Gerüste/Kräne in EP einzurechnen), "
        "riskante Pauschalierungen, unklare Entsorgungsklassen.\n"
        "3. NACHTRAGSPOTENZIAL: Lückenhafte Leistungsbeschreibungen oder fehlende Vorleistungen, die nach § 2 VOB/B "
        "Nachträge auslösen können.\n"
        "4. UNKLARE_SPEZIFIKATION: Fehlende Leitfabrikate, unklare Mengen/Schnittstellen.\n"
        "WICHTIGE REGEL: Ignoriere Anonymisierungs-Platzhalter wie [FIRMA_006], [PERSON_115], [ORT_009] komplett — "
        "das sind KEINE Fehler!\n"
        "Sei streng: Normale Positionen bekommen severity='LOW' und issue_type='OK'."
    )

    async with semaphore:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,
            tools=[ITEM_ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "record_boq_analysis"},
            messages=[
                {
                    "role": "user",
                    "content": f"Analysiere diese {len(batch)} LV-Positionen:\n{json.dumps(payload, ensure_ascii=False)}"
                }
            ],
        )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_boq_analysis":
            return block.input.get("results", [])
    return []


async def generate_executive_briefing_sonnet(
    client: AsyncAnthropic,
    parsed: dict,
    merged_items: list[dict],
) -> str:
    trade_counts = Counter(i["trade"] for i in merged_items)
    flagged = [i for i in merged_items if i["severity"] in ("HIGH", "MEDIUM")]
    high_items = [i for i in flagged if i["severity"] == "HIGH"]

    context = {
        "projekt": parsed["project_name"],
        "gesamtpositionen": parsed["total_items"],
        "bedarfspositionen": parsed["optional_items"],
        "pauschalpositionen": parsed["lumpsum_items"],
        "vorbemerkungen_auszug": parsed["vorbemerkungen"][:2000],
        "gewerke": dict(trade_counts.most_common()),
        "top_befunde": [
            {
                "oz": i["oz"],
                "trade": i["trade"],
                "type": i["issue_type"],
                "reason": i["reason"],
            }
            for i in high_items[:20]
        ],
    }

    prompt = f"""Erstelle ein kurzes, direktes **Angebots- & Kalkulations-Briefing** für einen Generalunternehmer (Bieter).

Daten:
{json.dumps(context, ensure_ascii=False, indent=2)}

Strukturiere exakt in 3 knappe Abschnitte (Markdown, keine Floskeln):
### 1. Kritische Fehler & VOB-Margenfallen
Die 3–5 wichtigsten Kostenrisiken (inkl. Vorbemerkungen und konkreten OZ-Nummern).
### 2. Nachtragspotenzial (§ 2 VOB/B) & Kalkulationsstrategie
Wo bestehen lückenhafte Ausschreibungen oder Mengenrisiken (Bedarfs-/Pauschalpositionen) für gezielte Kalkulation?
### 3. Nachunternehmer-Einkauf (RfQ-Priorität)
Welche Gewerke müssen sofort angefragt werden?"""

    response = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1800,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def run_pipeline(api_key: str, parsed: dict, progress_bar) -> tuple[list[dict], str]:
    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    items = parsed["items"]
    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    tasks = [analyze_batch_haiku(client, semaphore, b) for b in batches]
    results = []
    done = 0

    for coro in asyncio.as_completed(tasks):
        results.extend(await coro)
        done += 1
        progress_bar.progress(
            min(0.85, (done / len(batches)) * 0.85),
            text=f"Prüfe Positionen mit Claude Haiku... ({done}/{len(batches)} Pakete)"
        )

    res_map = {r["oz"]: r for r in results}
    merged = []
    for it in items:
        r = res_map.get(it["oz"], {})
        merged.append({
            **it,
            "trade": r.get("trade", "Sonstiges"),
            "severity": r.get("severity", "LOW"),
            "issue_type": r.get("issue_type", "OK"),
            "reason": r.get("reason", ""),
            "bieterfrage": r.get("bieterfrage", ""),
        })

    progress_bar.progress(0.90, text="Erstelle Management-Briefing mit Claude Sonnet...")
    memo = await generate_executive_briefing_sonnet(client, parsed, merged)
    progress_bar.progress(1.0, text="Fertig!")
    return merged, memo


# ==============================================================================
# 3. SIMPLE 3-ZONE INTERFACE
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Einstellungen")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
    )
    st.caption(f"Scanner: `{HAIKU_MODEL}`\nBriefing: `{SONNET_MODEL}`")

st.title("🏗️ AI BoQ & Bid Assistant")
st.caption("GAEB `.X83` Risiko-, Fehler- & Nachtragsanalyse für Generalunternehmer")

up_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
with up_col:
    uploaded = st.file_uploader(
        "Leistungsverzeichnis hochladen (GAEB .X83 / .X81 / .X84 bevorzugt, oder .PDF Fallback)",
        type=["x81", "x82", "x83", "x84", "x85", "x86", "xml", "pdf"],
    )
with btn_col:
    run_btn = st.button("🚀 Analysieren", type="primary", use_container_width=True)

if run_btn:
    if not api_key:
        st.error("Bitte Anthropic API Key in der Seitenleiste (oben links) oder als `ANTHROPIC_API_KEY` setzen.")
        st.stop()
    if not uploaded:
        st.error("Bitte zuerst eine GAEB-Datei (.X83) oder PDF hochladen.")
        st.stop()

    try:
        raw_bytes = uploaded.read()
        if uploaded.name.lower().endswith(".pdf"):
            parsed = parse_pdf_fallback(raw_bytes, uploaded.name)
        else:
            parsed = parse_gaeb_xml(raw_bytes)

        prog = st.progress(0.05, text="Starte KI-Analyse...")
        merged, memo = asyncio.run(run_pipeline(api_key, parsed, prog))
        prog.empty()

        st.session_state["parsed"] = parsed
        st.session_state["df"] = pd.DataFrame(merged)
        st.session_state["memo"] = memo
    except Exception as e:
        st.error(f"Fehler: {e}")
        st.stop()

if "df" in st.session_state:
    parsed = st.session_state["parsed"]
    df = st.session_state["df"]
    memo = st.session_state["memo"]

    errors_cnt = int((df["issue_type"] == "FEHLER_WIDERSPRUCH").sum())
    high_risk_cnt = int((df["severity"] == "HIGH").sum())
    nachtrag_cnt = int((df["issue_type"] == "NACHTRAGSPOTENZIAL").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Positionen gesamt", parsed["total_items"], help=f"Format: {parsed['source_type']}", border=True)
    k2.metric("⚠️ Fehler & Widersprüche", errors_cnt, border=True)
    k3.metric("🔴 Kritische Risiken (HIGH)", high_risk_cnt, border=True)
    k4.metric("💰 Nachtragspotenzial", nachtrag_cnt, border=True)

    tab1, tab2, tab3 = st.tabs([
        "🚨 1. Red Flags & Fehler",
        "📋 2. Bid Strategy & Gewerke",
        "🎛️ 3. What-If Simulator",
    ])

    with tab1:
        flagged_df = df[df["severity"].isin(["HIGH", "MEDIUM"])].copy()

        f_col1, f_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with f_col1:
            selected_types = st.multiselect(
                "Filter nach Befund-Typ:",
                options=["FEHLER_WIDERSPRUCH", "VOB_RISIKOFALLE", "NACHTRAGSPOTENZIAL", "UNKLARE_SPEZIFIKATION"],
                default=["FEHLER_WIDERSPRUCH", "VOB_RISIKOFALLE", "NACHTRAGSPOTENZIAL", "UNKLARE_SPEZIFIKATION"],
            )
        with f_col2:
            q_df = df[df["bieterfrage"].str.len() > 5]
            if not q_df.empty:
                export_txt = f"BIETERFRAGEN — {parsed['project_name']}\n" + "=" * 55 + "\n\n"
                export_txt += "\n\n".join(
                    f"OZ {r.oz} [{r.issue_type}]: {r.bieterfrage}\n  -> Hintergrund: {r.reason}"
                    for r in q_df.itertuples()
                )
                st.download_button(
                    f"⬇️ Bieterfragen ({len(q_df)})",
                    data=export_txt,
                    file_name="Bieterfragen_Export.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True,
                )

        view_df = flagged_df[flagged_df["issue_type"].isin(selected_types)]
        st.dataframe(
            view_df[["oz", "severity", "issue_type", "trade", "reason", "bieterfrage"]],
            use_container_width=True,
            hide_index=True,
        )

        if not view_df.empty:
            with st.expander("🔍 Original-LV-Text (Kurztext vs. Langtext) für markierte Positionen prüfen"):
                chosen_oz = st.selectbox("Position (OZ) auswählen:", view_df["oz"].tolist())
                row = view_df[view_df["oz"] == chosen_oz].iloc[0]
                st.markdown(f"**Kurztext:** {row['kurztext']}")
                st.markdown(f"**Langtext:** {row['langtext']}")

    with tab2:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(memo)
        with right:
            st.subheader("Nachunternehmer-Gewerke (RfQ)")
            trade_counts = df["trade"].value_counts()
            st.bar_chart(trade_counts)
            st.caption(
                f"Bedarfspositionen: **{parsed['optional_items']}** | "
                f"Pauschalpositionen: **{parsed['lumpsum_items']}**"
            )

    with tab3:
        @st.fragment
        def what_if_simulator():
            st.subheader("Kalkulations- & Risiko-Simulator (What-If)")
            st.caption("Simuliert die Auswirkung von Preissteigerungen und Risikozuschlägen auf die Angebotssumme.")

            s1, s2, s3, s4 = st.columns(4)
            base_bid = s1.number_input("Geschätzte Basis-Angebotssumme (€)", value=500_000, step=50_000)
            mat_shock = s2.slider("Materialpreis-Steigerung (%)", -10, 30, 5)
            risk_buf = s3.slider("Risikozuschlag auf HIGH-Pos. (%)", 0, 25, 8)
            opt_act = s4.slider("Abruf Bedarfspositionen (%)", 0, 100, 30)

            high_share = max(0.05, high_risk_cnt / max(1, parsed["total_items"]))
            opt_share = parsed["optional_items"] / max(1, parsed["total_items"])

            mat_impact = base_bid * 0.45 * (mat_shock / 100.0)
            risk_impact = base_bid * high_share * (risk_buf / 100.0)
            opt_impact = base_bid * opt_share * (opt_act / 100.0)
            adjusted_bid = base_bid + mat_impact + risk_impact + opt_impact
            delta_pct = ((adjusted_bid - base_bid) / base_bid) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Angepasste Angebotssumme", f"{adjusted_bid:,.0f} €", f"{delta_pct:+.1f}%", border=True)
            m2.metric("Material- & Risikopuffer", f"{(mat_impact + risk_impact):+,.0f} €", border=True)
            m3.metric("Potenzial Bedarfspositionen", f"{opt_impact:+,.0f} €", border=True)

            chart_df = pd.DataFrame({
                "Komponente": ["Basis-Angebot", "Material-Effekt", "VOB-Risikopuffer", "Bedarfspositionen"],
                "Betrag (€)": [base_bid, mat_impact, risk_impact, opt_impact],
            }).set_index("Komponente")
            st.bar_chart(chart_df)

        what_if_simulator()
else:
    st.info("Lade oben eine `.X83`-Datei (z.B. `GAEB-[PROJEKT_004].X83` oder `8209.003_Leistungsverzeichnis_leer.X83`) hoch und klicke auf **Analysieren**.")
