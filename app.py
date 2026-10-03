"""Streamlit front-end for SDG Impact AI."""

import streamlit as st
from src.sdg_analyzer import analyze_initiative
from src.report_generator import build_markdown, save_report

st.set_page_config(page_title="SDG Impact AI", page_icon="🌍", layout="wide")

st.title("🌍 SDG Impact AI")
st.caption("AI-powered mapping of sustainability initiatives to UN SDGs, with RAG-based impact analysis.")

with st.sidebar:
    st.header("About")
    st.write(
        "This tool uses a Retrieval-Augmented Generation (RAG) pipeline:\n"
        "1. Your initiative is embedded.\n"
        "2. Top-matching SDGs are retrieved from a FAISS vector store.\n"
        "3. A large language model (Gemini) analyzes impact using the retrieved context."
    )
    top_k = st.slider("Number of SDGs to retrieve", 3, 10, 5)

initiative = st.text_area(
    "Describe your sustainability initiative",
    height=140,
    placeholder="Example: A solar microgrid project powering rural schools and health clinics in Bihar.",
)

col1, col2 = st.columns([1, 4])
run = col1.button("Analyze", type="primary")

if run:
    if len(initiative.strip()) < 10:
        st.warning("Please describe the initiative in at least 10 characters.")
        st.stop()

    with st.spinner("Retrieving relevant SDGs and asking the LLM ..."):
        try:
            analysis = analyze_initiative(initiative, top_k=top_k)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    result = analysis["result"]

    st.success("Analysis complete!")

    # Primary SDG
    st.subheader("🎯 Primary SDG")
    p = result["primary_sdg"]
    st.markdown(f"### SDG {p['sdg_id']} — {p['title']}")
    st.write(p["reason"])

    # Secondary
    if result.get("secondary_sdgs"):
        st.subheader("🔗 Secondary SDGs")
        for s in result["secondary_sdgs"]:
            st.markdown(f"- **SDG {s['sdg_id']} — {s['title']}**: {s['reason']}")

    # Impact
    st.subheader("📊 Impact Analysis")
    impact = result["impact_analysis"]
    badge = {"Low": "🟡", "Medium": "🟠", "High": "🟢"}.get(impact["impact_level"], "⚪")
    st.markdown(f"**Impact Level:** {badge} `{impact['impact_level']}`")
    st.write(f"*{impact['justification']}*")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Positive Outcomes**")
        for x in impact.get("positive_outcomes", []):
            st.markdown(f"- {x}")
    with c2:
        st.markdown("**⚠️ Risks & Trade-offs**")
        for x in impact.get("risks_and_tradeoffs", []):
            st.markdown(f"- {x}")

    st.subheader("📈 Suggested KPIs")
    for k in result.get("suggested_kpis", []):
        st.markdown(f"- {k}")

    # RAG evidence
    with st.expander("🔍 Retrieved SDG context (RAG evidence)"):
        for item in analysis["retrieved"]:
            m = item["metadata"]
            st.markdown(f"- **SDG {m['sdg_id']} — {m['title']}** · similarity `{item['score']:.3f}`")

    # Save + download
    md = build_markdown(initiative, analysis)
    path = save_report(initiative, analysis)
    st.download_button(
        "📥 Download Markdown Report",
        data=md,
        file_name=path.name,
        mime="text/markdown",
    )
    st.caption(f"Report saved to `{path}`")