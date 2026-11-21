# Student Academic Performance Analytics

**Institution:** ABES Engineering College, Ghaziabad  
**Student:** Piyush Kumar Soni  
**Admission Number:** 2026B01010999  
**Educator:** Avinay Sir  
**Submission Date:** 21 November 2026

## Project overview
A Python, NumPy and Pandas project that analyzes attendance, study habits, subject marks and internet access to identify academic performance patterns.

## Dataset
`data/02_student_performance.csv` — 500 records, 11 columns.

## Project structure
```text
Student_Academic_Performance_Analytics/
├── data/
│   └── 02_student_performance.csv
├── src/
│   └── analysis.py
├── reports/
│   ├── student_performance_analyzed.csv
│   ├── department_summary.csv
│   ├── top_10_students.csv
│   ├── at_risk_students.csv
│   ├── attendance_band_summary.csv
│   ├── internet_access_summary.csv
│   └── *.png
├── Student_Academic_Performance_Analytics.ipynb
├── SRS_Student_Academic_Performance_Analytics.docx
└── README.md
```

## Requirements
- Python 3
- pandas
- numpy
- matplotlib

Install:
```bash
pip install pandas numpy matplotlib
```

## Run
From the project root:
```bash
python src/analysis.py
```
Or open the notebook in Jupyter and run its analysis cell.

## Data-cleaning decisions
- The 10 missing attendance values are replaced with the median attendance (78.2%).
- Duplicate Student_ID values are checked; none were found in the supplied dataset.
- All subject marks are checked for the 0–100 range; no invalid values were found.
- Student average is the mean of Python, Maths and Physics marks.
- At-risk threshold is average marks below 40%.

## Important interpretation note
This is descriptive analysis of a synthetic training dataset. Correlation does not imply causation, and automated flags should be reviewed by faculty before any student-level decision.

## GitHub submission
Create a new public or instructor-accessible GitHub repository, upload this complete folder, and paste the actual repository URL into the LMS. Do not submit a made-up URL.
