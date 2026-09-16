# Student Data Visualization and Correlation Analysis

## Description

This project is my Day 16 project in my Artificial Intelligence and Machine Learning learning journey.

The goal of this project is to explore and analyze a student dataset using data visualization and correlation analysis.

In this project, I use Python, Pandas, Matplotlib, and Seaborn to visualize student performance and identify relationships between different academic features.

The project focuses on:

- Scatter plots
- Histograms
- Box plots
- Correlation analysis
- Correlation matrices
- Correlation heatmaps
- Comparing students who passed and failed

Data visualization helps me understand patterns and relationships in the dataset before building machine learning models.

---

## Dataset

The project uses a student dataset stored in:

```text
student_data.csv
```

The dataset contains information about student academic performance.

The main columns used in this project are:

- `Study_Hours`
- `Attendance`
- `Assignment_Score`
- `Midterm_Score`
- `Exam_Score`
- `Final_Result`

The `Final_Result` column contains two categories:

```text
Pass
Fail
```

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn

---

## Libraries Used

### Pandas

Pandas is used to load and analyze the student dataset.

```python
import pandas as pd
```

The dataset is loaded using:

```python
df = pd.read_csv("student_data.csv")
```

### Matplotlib

Matplotlib is used to create:

- Scatter plots
- Histograms
- Box plots

```python
import matplotlib.pyplot as plt
```

### Seaborn

Seaborn is used to create the correlation heatmap.

```python
import seaborn as sns
```

---

# Data Visualization

Data visualization is the process of representing data using graphs and charts.

Visualization makes it easier to identify:

- Patterns
- Relationships
- Distributions
- Differences between groups
- Possible outliers

In this project, several visualization techniques are used to understand student performance.

---

# Scatter Plots

Scatter plots are used to visualize the relationship between two numerical variables.

This project creates scatter plots for:

- Study Hours vs Exam Score
- Attendance vs Exam Score
- Assignment Score vs Exam Score
- Midterm Score vs Exam Score

---

## Study Hours vs Exam Score

The first scatter plot compares study hours with exam scores.

```python
plt.scatter(
    df["Study_Hours"],
    df["Exam_Score"]
)
```

This visualization helps examine whether students who study more tend to have higher exam scores.

---

## Attendance vs Exam Score

The second scatter plot compares attendance with exam scores.

```python
plt.scatter(
    df["Attendance"],
    df["Exam_Score"]
)
```

This helps examine whether higher attendance is associated with higher exam performance.

---

## Assignment Score vs Exam Score

The third scatter plot compares assignment scores with exam scores.

```python
plt.scatter(
    df["Assignment_Score"],
    df["Exam_Score"]
)
```

This helps examine whether students who perform well on assignments also tend to perform well on exams.

---

## Midterm Score vs Exam Score

The fourth scatter plot compares midterm scores with exam scores.

```python
plt.scatter(
    df["Midterm_Score"],
    df["Exam_Score"]
)
```

This helps examine whether students who perform well on the midterm also tend to perform well on the final exam.

---

# Histograms

A histogram shows the distribution of numerical data.

This project creates histograms for:

- Exam Scores
- Study Hours

---

## Exam Score Distribution

The project uses:

```python
plt.hist(
    df["Exam_Score"],
    bins=5
)
```

This helps visualize how exam scores are distributed among students.

It can show whether scores are concentrated in a particular range.

---

## Study Hours Distribution

The project also creates a histogram for study hours.

```python
plt.hist(
    df["Study_Hours"],
    bins=5
)
```

This helps visualize how study hours are distributed among students.

---

# Box Plots

Box plots are useful for understanding the distribution and spread of numerical data.

They can help show:

- Minimum value
- Lower quartile
- Median
- Upper quartile
- Maximum value
- Possible outliers

---

## Exam Score Box Plot

The project creates a box plot for exam scores:

```python
plt.boxplot(
    df["Exam_Score"]
)
```

This helps understand the distribution of exam scores and identify possible unusual values.

---

## Exam Scores by Final Result

The project compares exam scores between students who passed and students who failed.

First, the scores are separated:

```python
pass_scores = df[
    df["Final_Result"] == "Pass"
]["Exam_Score"]

fail_scores = df[
    df["Final_Result"] == "Fail"
]["Exam_Score"]
```

The two groups are then displayed using a box plot:

```python
plt.boxplot(
    [fail_scores, pass_scores],
    tick_labels=["Fail", "Pass"]
)
```

This allows the exam-score distributions of the two groups to be compared visually.

---

## Study Hours by Final Result

The project also compares study hours between students who passed and students who failed.

The study hours are separated into:

```python
pass_hours
fail_hours
```

A box plot is then used to compare the two groups.

```python
plt.boxplot(
    [fail_hours, pass_hours],
    tick_labels=["Fail", "Pass"]
)
```

This helps examine whether students who passed generally studied more than students who failed.

---

# Correlation Analysis

Correlation measures the strength and direction of a linear relationship between two numerical variables.

Correlation values generally range from:

```text
-1 to +1
```

A simplified interpretation is:

```text
+1 → Strong positive relationship
 0 → Little or no linear relationship
-1 → Strong negative relationship
```

A positive correlation means that two variables tend to increase together.

A negative correlation means that when one variable increases, the other tends to decrease.

---

# Correlation Matrix

The project calculates a correlation matrix using:

```python
correlation = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
].corr()
```

The matrix shows the correlation between the numerical features.

It allows relationships such as the following to be examined:

```text
Study Hours ↔ Exam Score

Attendance ↔ Exam Score

Assignment Score ↔ Exam Score

Midterm Score ↔ Exam Score
```

---

# Correlation Heatmap

The correlation matrix is visualized using a Seaborn heatmap.

```python
sns.heatmap(
    correlation,
    annot=True
)
```

The `annot=True` parameter displays the numerical correlation values inside the heatmap.

The heatmap makes it easier to visually identify stronger and weaker relationships between features.

---

# Important Concept: Correlation Does Not Mean Causation

One of the important concepts I learned from this project is:

```text
Correlation does not mean causation.
```

For example, if study hours and exam scores have a strong positive correlation, this does not automatically prove that studying more causes higher exam scores.

Other factors may also influence student performance.

Correlation describes a relationship between variables, but it does not by itself prove cause and effect.

---

# Questions Explored

This project helps explore the following questions:

1. Is study time related to exam performance?
2. Is attendance related to exam performance?
3. Are assignment scores related to exam scores?
4. Are midterm scores related to exam scores?
5. Do students who pass generally have higher exam scores?
6. Do students who pass generally study more?
7. Which features have stronger relationships with exam scores?
8. What does the correlation matrix reveal?
9. Are there any potential outliers?
10. What patterns can be identified from the visualizations?

---

# Machine Learning Connection

Data visualization is an important part of the machine learning workflow.

Before building a machine learning model, it is useful to understand the dataset.

A simplified workflow is:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Data Visualization
     ↓
Identify Patterns
     ↓
Analyze Relationships
     ↓
Feature Selection
     ↓
Machine Learning Model
     ↓
Model Evaluation
```

Visualization can help identify useful features and potential problems in the dataset before model training.

---

# What I Learned

Through this project, I learned how to use data visualization to explore a dataset.

I learned how to create scatter plots to visualize relationships between two numerical variables.

I learned how histograms can be used to understand the distribution of numerical data.

I learned how box plots can be used to understand the spread of data, compare groups, and identify possible outliers.

I also learned how to calculate a correlation matrix using Pandas and visualize it using a Seaborn heatmap.

Another important concept I learned is that correlation does not necessarily mean causation.

This project helped me understand how data visualization can be used before applying machine learning algorithms.

---

# Previous Skills Used

This project builds on skills learned in previous projects, including:

- Python
- NumPy
- Pandas
- Data Analysis
- Statistics
- Matplotlib
- Logistic Regression
- Decision Trees
- Random Forest
- Machine Learning
- Model Evaluation

---

# Project Structure

```text
Day 16
│
├── student_data_visualization.py
├── student_data.csv
└── README.md
```

---

# Installation

Make sure Python is installed on your computer.

Install the required libraries using:

```bash
pip install pandas matplotlib seaborn
```

---

# How to Run

Open a terminal in the project directory and run:

```bash
python student_data_visualization.py
```

The program will:

1. Load the student dataset.
2. Display the first five rows.
3. Create scatter plots.
4. Create histograms.
5. Create box plots.
6. Compare students who passed and failed.
7. Calculate a correlation matrix.
8. Display a correlation heatmap.

---

# Matplotlib Compatibility Note

For newer versions of Matplotlib, the box plot category names are specified using:

```python
tick_labels=["Fail", "Pass"]
```

instead of the older:

```python
labels=["Fail", "Pass"]
```

Using `tick_labels` prevents the following error in newer Matplotlib versions:

```text
TypeError: boxplot() got an unexpected keyword argument 'labels'
```

---

# Future Improvements

In future projects, I plan to:

- Create more advanced visualizations
- Add more student data
- Explore larger datasets
- Visualize machine learning predictions
- Visualize confusion matrices
- Create ROC curves
- Compare multiple machine learning models
- Perform more advanced exploratory data analysis
- Build an end-to-end machine learning project

---

# Conclusion

This project helped me understand how data visualization can be used to explore and analyze a dataset before applying machine learning algorithms.

I practiced creating:

- Scatter plots
- Histograms
- Box plots
- Correlation matrices
- Correlation heatmaps

I also explored relationships between study hours, attendance, assignment scores, midterm scores, and exam scores.

Most importantly, I learned that visual exploration can reveal patterns and relationships that may not be obvious from raw data alone.

This project represents another step in my journey toward Artificial Intelligence and Machine Learning.