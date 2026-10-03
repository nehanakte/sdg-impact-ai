"""Run the SDG Impact AI from the terminal."""

import json
from src.sdg_analyzer import analyze_initiative
from src.report_generator import save_report


def main():
    print("=" * 60)
    print(" SDG Impact AI — AI-Powered Sustainability Impact Analysis")
    print("=" * 60)
    print("Describe your sustainability initiative (or 'quit' to exit).\n")

    while True:
        try:
            initiative = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if initiative.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not initiative:
            continue

        try:
            analysis = analyze_initiative(initiative)
        except Exception as e:
            print(f"[error] {e}\n")
            continue

        result = analysis["result"]
        print("\n--- PRIMARY SDG ---")
        print(f"SDG {result['primary_sdg']['sdg_id']}: {result['primary_sdg']['title']}")
        print(result["primary_sdg"]["reason"])

        print("\n--- SECONDARY SDGs ---")
        for s in result.get("secondary_sdgs", []):
            print(f"SDG {s['sdg_id']}: {s['title']} — {s['reason']}")

        print("\n--- IMPACT ---")
        print("Level:", result["impact_analysis"]["impact_level"])
        print(result["impact_analysis"]["justification"])

        print("\n--- KPIs ---")
        for k in result.get("suggested_kpis", []):
            print(" -", k)

        path = save_report(initiative, analysis)
        print(f"\n[report saved] {path}\n")


if __name__ == "__main__":
    main()