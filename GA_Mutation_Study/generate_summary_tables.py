
import pandas as pd
from pathlib import Path

STUDY_ROOT = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study")
SUMMARY_FILE = STUDY_ROOT / "analysis" / "analysis_summary.csv"
TABLES_DIR = STUDY_ROOT / "analysis"
TABLES_DIR.mkdir(exist_ok=True)

def generate_summary_tables():
    if not SUMMARY_FILE.exists():
        print(f"❌ Error: {SUMMARY_FILE} not found.")
        return

    df = pd.read_csv(SUMMARY_FILE)
    if df.empty:
        print("❌ Error: Summary file is empty.")
        return

    # 1. Part B.1: Genetic Recombination (Factorial)
    print("Generating Part B.1 Summary Table...")
    b1 = df[df['part'] == 'PartB1']
    if not b1.empty:
        table_b1 = b1.groupby(['dataset', 'condition'])[['final_snr', 'final_entropy', 'final_nfidr', 'final_acc']].agg(['mean', 'std']).round(4)
        table_b1.to_csv(TABLES_DIR / "table_part_b1.csv")
        table_b1.to_markdown(TABLES_DIR / "table_part_b1.md")

    # 2. Part B.2: Mutation Impact (Dose-Response)
    print("Generating Part B.2 Summary Table...")
    b2 = df[df['part'] == 'PartB2']
    if not b2.empty:
        table_b2 = b2.groupby(['dataset', 'condition'])[['final_snr', 'final_entropy', 'final_nfidr', 'final_acc']].agg(['mean', 'std']).round(4)
        table_b2.to_csv(TABLES_DIR / "table_part_b2.csv")
        table_b2.to_markdown(TABLES_DIR / "table_part_b2.md")

    # 3. Part C: Interaction
    print("Generating Part C Summary Table...")
    c = df[df['part'] == 'PartC']
    if not c.empty:
        table_c = c.groupby(['dataset', 'condition'])[['final_snr', 'final_entropy', 'final_nfidr', 'final_acc']].agg(['mean', 'std']).round(4)
        table_c.to_csv(TABLES_DIR / "table_part_c.csv")
        table_c.to_markdown(TABLES_DIR / "table_part_c.md")

    print(f"✅ Summary tables saved to {TABLES_DIR}")

if __name__ == "__main__":
    generate_summary_tables()
