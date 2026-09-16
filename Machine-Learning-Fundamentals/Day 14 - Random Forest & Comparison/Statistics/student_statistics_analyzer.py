import numpy as np


# ============================================
# STUDENT STATISTICS ANALYZER
# ============================================

students = ["Tim", "Gim", "Mim", "Kim", "Sim"]

math = np.array([90, 70, 60, 40, 95])
science = np.array([85, 80, 65, 45, 90])
english = np.array([88, 75, 70, 50, 95])


# ============================================
# FUNCTION TO DISPLAY SUBJECT STATISTICS
# ============================================

def display_statistics(subject, scores):

    print(f"\n{'=' * 40}")
    print(f"{subject} Statistics")
    print(f"{'=' * 40}")

    print(f"Mean:                {np.mean(scores):.2f}")
    print(f"Median:              {np.median(scores):.2f}")
    print(f"Maximum:             {np.max(scores):.2f}")
    print(f"Minimum:             {np.min(scores):.2f}")
    print(f"Variance:            {np.var(scores):.2f}")
    print(f"Standard Deviation:  {np.std(scores):.2f}")


# ============================================
# DISPLAY STATISTICS
# ============================================

display_statistics("Math", math)
display_statistics("Science", science)
display_statistics("English", english)


# ============================================
# SUBJECT AVERAGES
# ============================================

subject_averages = {
    "Math": np.mean(math),
    "Science": np.mean(science),
    "English": np.mean(english)
}


print(f"\n{'=' * 40}")
print("Subject Average Comparison")
print(f"{'=' * 40}")

for subject, average in subject_averages.items():
    print(f"{subject}: {average:.2f}")


# ============================================
# FIND HIGHEST-AVERAGE SUBJECT
# ============================================

highest_subject = max(
    subject_averages,
    key=subject_averages.get
)

highest_average = subject_averages[highest_subject]

print(f"\nHighest Average Subject: {highest_subject}")
print(f"Average Score: {highest_average:.2f}")


# ============================================
# CORRELATION ANALYSIS
# ============================================

study_hours = np.array([1, 2, 3, 4, 5])
overall_scores = np.array([50, 55, 65, 75, 90])


correlation = np.corrcoef(
    study_hours,
    overall_scores
)[0, 1]


print(f"\n{'=' * 40}")
print("Study Hours vs. Scores")
print(f"{'=' * 40}")

print(f"Correlation: {correlation:.2f}")


# ============================================
# INTERPRET CORRELATION
# ============================================

if correlation > 0.7:
    interpretation = "Strong positive relationship"

elif correlation > 0.3:
    interpretation = "Moderate positive relationship"

elif correlation > -0.3:
    interpretation = "Weak relationship"

else:
    interpretation = "Negative relationship"


print(f"Interpretation: {interpretation}")


# ============================================
# END OF ANALYSIS
# ============================================

print(f"\n{'=' * 40}")
print("Analysis Complete")
print(f"{'=' * 40}")