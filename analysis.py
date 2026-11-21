"""
Project 2: Student Academic Performance Analytics
Student: Piyush Kumar Soni | Admission No.: 2026B01010999
ABES Engineering College, Ghaziabad
Educator: Avinay Sir | Submission: 21 November 2026

Run from the project root:
    python src/analysis.py
Dependencies: pandas, numpy, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "02_student_performance.csv"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_inspect(path):
    """Load the student dataset and print its basic structure."""
    data = pd.read_csv(path)
    print("\n--- DATASET OVERVIEW ---")
    print("Shape:", data.shape)
    print("\nFirst five rows:\n", data.head())
    print("\nData types:\n", data.dtypes)
    print("\nDescriptive statistics:\n", data.describe(include="all"))
    return data

def clean_data(data):
    """Report data quality and apply documented cleaning decisions."""
    df = data.copy()
    print("\n--- DATA QUALITY ---")
    print("Missing values by column:\n", df.isna().sum())
    print("Duplicate Student_ID values:", df["Student_ID"].duplicated().sum())

    # Attendance is a continuous percentage. Median imputation is robust to
    # extreme values and retains all student records.
    missing_attendance = int(df["Attendance_Percent"].isna().sum())
    attendance_median = df["Attendance_Percent"].median()
    df["Attendance_Percent"] = df["Attendance_Percent"].fillna(attendance_median)
    print(f"Filled {missing_attendance} missing attendance values with median "
          f"({attendance_median:.2f}%).")

    # Remove duplicate IDs only if they exist; keep the first occurrence.
    df = df.drop_duplicates(subset="Student_ID", keep="first")

    score_columns = ["Python_Marks", "Maths_Marks", "Physics_Marks"]
    for col in score_columns:
        invalid = ~df[col].between(0, 100)
        print(f"{col}: {int(invalid.sum())} value(s) outside 0-100.")
    # Do not silently alter marks: flag invalid values and exclude such rows
    # from score-derived calculations if any are found.
    valid_scores = df[score_columns].apply(lambda s: s.between(0, 100)).all(axis=1)
    df["Valid_Scores"] = valid_scores
    return df

def analyze(df):
    score_cols = ["Python_Marks", "Maths_Marks", "Physics_Marks"]
    valid = df[df["Valid_Scores"]].copy()

    # Student-level score calculations
    valid["Total_Marks_Calculated"] = valid[score_cols].sum(axis=1)
    valid["Average_Marks"] = valid[score_cols].mean(axis=1).round(2)

    # Grade bands are based on the average of the three subject marks.
    bins = [-0.001, 39.99, 49.99, 59.99, 69.99, 79.99, 89.99, 100]
    labels = ["F", "E", "D", "C", "B", "A", "A+"]
    valid["Grade"] = pd.cut(valid["Average_Marks"], bins=bins, labels=labels)

    # Department-wise summary
    department_summary = (valid.groupby("Department", observed=False)
                          ["Average_Marks"].agg(["count", "mean", "median"])
                          .round(2).sort_values("mean", ascending=False))
    print("\n--- DEPARTMENT-WISE AVERAGE MARKS ---\n", department_summary)

    # Top ten students
    top10 = valid.nlargest(10, "Average_Marks")[
        ["Student_ID", "Department", "Average_Marks", "Grade"]]
    print("\n--- TOP 10 STUDENTS ---\n", top10.to_string(index=False))

    # Students below 40% average marks
    at_risk = valid[valid["Average_Marks"] < 40].copy()
    print("\nStudents below 40% average marks:", len(at_risk))
    print(at_risk[["Student_ID", "Department", "Average_Marks",
                   "Attendance_Percent", "Study_Hours_Per_Day"]].to_string(index=False))

    # Attendance bands
    attendance_bins = [-0.001, 59.99, 74.99, 89.99, 100]
    attendance_labels = ["Below 60%", "60-74.99%", "75-89.99%", "90-100%"]
    valid["Attendance_Band"] = pd.cut(
        valid["Attendance_Percent"], bins=attendance_bins,
        labels=attendance_labels)
    attendance_summary = (valid.groupby("Attendance_Band", observed=False)
                          ["Average_Marks"].agg(["count", "mean"]).round(2))
    print("\n--- ATTENDANCE BAND COMPARISON ---\n", attendance_summary)

    # Study-hours relationship: Pearson correlation is descriptive, not causal.
    study_corr = valid["Study_Hours_Per_Day"].corr(valid["Average_Marks"])
    print(f"\nCorrelation: study hours vs average marks = {study_corr:.3f}")

    # Internet access comparison
    internet_summary = (valid.groupby("Internet_Access", observed=False)
                        ["Average_Marks"].agg(["count", "mean", "median"]).round(2))
    print("\n--- INTERNET ACCESS COMPARISON ---\n", internet_summary)

    # NumPy statistics for Python marks
    python_marks = valid["Python_Marks"].to_numpy(dtype=float)
    q1, median, q3 = np.percentile(python_marks, [25, 50, 75])
    std_dev = np.std(python_marks, ddof=1)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = valid[(valid["Python_Marks"] < lower_fence) |
                     (valid["Python_Marks"] > upper_fence)]
    print("\n--- PYTHON MARKS: NUMPY STATISTICS ---")
    print(f"Median={median:.2f}, Sample SD={std_dev:.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"IQR fences: {lower_fence:.2f} to {upper_fence:.2f}")
    print("Potential Python_Marks outliers:", len(outliers))
    print(outliers[["Student_ID", "Python_Marks"]].to_string(index=False))

    # Save the full student-level dataset with calculated totals, averages and grades.
    valid.to_csv(OUTPUT_DIR/"student_performance_analyzed.csv", index=False)

    # Save analysis tables
    department_summary.to_csv(OUTPUT_DIR/"department_summary.csv")
    top10.to_csv(OUTPUT_DIR/"top_10_students.csv", index=False)
    at_risk.to_csv(OUTPUT_DIR/"at_risk_students.csv", index=False)
    attendance_summary.to_csv(OUTPUT_DIR/"attendance_band_summary.csv")
    internet_summary.to_csv(OUTPUT_DIR/"internet_access_summary.csv")

    # Grade distribution chart
    grade_counts = valid["Grade"].value_counts().reindex(labels, fill_value=0)
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind="bar")
    plt.title("Student Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"grade_distribution.png", dpi=160)
    plt.close()

    # Department chart
    plt.figure(figsize=(8, 5))
    department_summary["mean"].sort_values().plot(kind="barh")
    plt.title("Department-wise Average Marks")
    plt.xlabel("Average Marks")
    plt.ylabel("Department")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"department_average.png", dpi=160)
    plt.close()

    # Attendance band chart
    plt.figure(figsize=(8, 5))
    attendance_summary["mean"].plot(kind="bar")
    plt.title("Average Marks by Attendance Band")
    plt.xlabel("Attendance Band")
    plt.ylabel("Average Marks")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"attendance_vs_marks.png", dpi=160)
    plt.close()

    # Study hours scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(valid["Study_Hours_Per_Day"], valid["Average_Marks"], alpha=0.65)
    plt.title("Study Hours per Day vs Average Marks")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Average Marks")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"study_hours_vs_marks.png", dpi=160)
    plt.close()

    print("\nCharts and CSV summaries saved in:", OUTPUT_DIR)
    return valid

def main():
    raw = load_and_inspect(DATA_FILE)
    cleaned = clean_data(raw)
    analyzed = analyze(cleaned)
    print("\nAnalysis completed. Students analyzed:", len(analyzed))

if __name__ == "__main__":
    main()
