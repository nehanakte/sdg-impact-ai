"""Turns the analysis dict into a clean markdown report and saves it."""

from datetime import datetime
from pathlib import Path
from src.config import BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def build_markdown(initiative: str, analysis: dict) -> str:
    r = analysis["result"]
    primary = r["primary_sdg"]
    secondary = r.get("secondary_sdgs", [])
    impact = r["impact_analysis"]
    kpis = r.get("suggested_kpis", [])

    md = []
    md.append("# SDG Impact Analysis Report\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append("## Initiative Description\n")
    md.append(f"> {initiative}\n")

    md.append("## Primary SDG\n")
    md.append(f"**SDG {primary['sdg_id']}: {primary['title']}**\n")
    md.append(f"{primary['reason']}\n")

    if secondary:
        md.append("## Secondary SDGs\n")
        for s in secondary:
            md.append(f"- **SDG {s['sdg_id']}: {s['title']}** — {s['reason']}")
        md.append("")

    md.append("## Impact Analysis\n")
    md.append(f"**Overall Impact Level:** `{impact['impact_level']}`\n")
    md.append(f"*{impact['justification']}*\n")

    md.append("### Positive Outcomes")
    for p in impact.get("positive_outcomes", []):
        md.append(f"- {p}")

    md.append("\n### Risks & Trade-offs")
    for x in impact.get("risks_and_tradeoffs", []):
        md.append(f"- {x}")

    md.append("\n## Suggested KPIs")
    for k in kpis:
        md.append(f"- {k}")

    md.append("\n---\n")
    md.append("### Retrieved Context (RAG evidence)")
    for item in analysis["retrieved"]:
        m = item["metadata"]
        md.append(f"- SDG {m['sdg_id']} *{m['title']}* (similarity {item['score']:.3f})")

    return "\n".join(md)


def save_report(initiative: str, analysis: dict) -> Path:
    md = build_markdown(initiative, analysis)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_{timestamp}.md"
    path.write_text(md, encoding="utf-8")
    return path