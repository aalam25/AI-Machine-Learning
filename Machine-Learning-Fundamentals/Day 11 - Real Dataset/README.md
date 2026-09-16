# Student Data Analysis Using Pandas

## Description

This project is my Day 11 data analysis project and focuses on working with a student dataset using Python and Pandas.

In my previous projects, I created small datasets manually using NumPy and used them for machine learning models. In this project, I started working with data stored in a CSV file.

The goal of this project is to learn how to load, explore, analyze, and filter a dataset using Pandas before using the data for machine learning.

The project focuses mainly on data exploration and basic data analysis rather than training a machine learning model.

---

## Dataset

The project uses a sample dataset containing information about 50 students.

The dataset contains the following columns:

- Student ID
- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score
- Final Result

### Dataset Example

| Student ID | Study Hours | Attendance | Assignment Score | Midterm Score | Exam Score | Final Result |
|------------|------------:|-----------:|-----------------:|--------------:|-----------:|--------------|
| S001 | 2 | 68 | 55 | 52 | 50 | Fail |
| S002 | 3 | 72 | 60 | 58 | 55 | Fail |
| S003 | 4 | 75 | 65 | 62 | 60 | Fail |
| S004 | 5 | 80 | 70 | 68 | 65 | Pass |
| S005 | 6 | 85 | 75 | 72 | 70 | Pass |

The complete dataset contains 50 student records and is stored in:

```text
student_data.csv
```

---

## Technologies Used

- Python
- Pandas
- CSV

---

## Python Library

The main library used in this project is:

```python
import pandas as pd
```

Pandas is used for loading, exploring, filtering, and analyzing the student dataset.

---

## Loading the Dataset

The CSV file is loaded using:

```python
df = pd.read_csv("student_data.csv")
```

This converts the CSV data into a Pandas DataFrame.

---

## Data Exploration

The project uses several Pandas functions to understand the dataset.

### Displaying the First Rows

```python
df.head()
```

This displays the first five rows of the dataset.

---

### Checking Dataset Size

```python
df.shape
```

This shows the number of rows and columns.

The dataset contains:

```text
50 rows
7 columns
```

---

### Checking Column Names

```python
df.columns
```

This displays all the column names in the DataFrame.

---

### Checking Data Types

```python
df.dtypes
```

This shows the data type of each column.

For example, numerical columns can contain integer values while the `Final_Result` column contains text values.

---

### Dataset Information

```python
df.info()
```

This provides information about:

- Number of rows
- Column names
- Data types
- Non-null values
- Memory usage

---

### Statistical Summary

```python
df.describe()
```

This provides statistical information about the numerical columns, including:

- Count
- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

---

## Missing Value Analysis

The project checks for missing values using:

```python
df.isnull().sum()
```

This counts the number of missing values in each column.

The student dataset used in this project does not contain missing values.

Checking for missing values is an important step before performing data analysis or machine learning.

---

# Student Performance Analysis

After exploring the dataset, the project calculates several statistics about student performance.

## Average Study Hours

The average number of study hours is calculated using:

```python
df["Study_Hours"].mean()
```

This helps identify the typical amount of time students spend studying.

---

## Average Attendance

The average attendance is calculated using:

```python
df["Attendance"].mean()
```

This provides an overview of the attendance level across the students.

---

## Average Exam Score

The average exam score is calculated using:

```python
df["Exam_Score"].mean()
```

This shows the overall average exam performance of the students.

---

## Highest Exam Score

The highest exam score is found using:

```python
df["Exam_Score"].max()
```

This identifies the highest score achieved in the dataset.

---

## Lowest Exam Score

The lowest exam score is found using:

```python
df["Exam_Score"].min()
```

This identifies the lowest score in the dataset.

---

## Finding the Highest-Scoring Student

The project uses:

```python
df.loc[df["Exam_Score"].idxmax()]
```

This finds the complete record of the student who achieved the highest exam score.

The `idxmax()` function identifies the index of the highest value, while `loc[]` retrieves the corresponding row.

---

# Pass and Fail Analysis

The dataset contains a `Final_Result` column with two possible values:

```text
Pass
Fail
```

## Counting Pass and Fail Students

The number of students in each category is calculated using:

```python
df["Final_Result"].value_counts()
```

This allows the project to determine how many students passed and how many failed.

---

## Overall Pass Rate

The project calculates the overall pass rate using:

```python
pass_rate = (df["Final_Result"] == "Pass").mean()
```

The result is then displayed as a percentage:

```python
print(f"{pass_rate:.2%}")
```

This provides an easy-to-understand measurement of the percentage of students who passed.

---

# Comparing Passing and Failing Students

The project also calculates the average exam score for students who passed.

```python
df[df["Final_Result"] == "Pass"]["Exam_Score"].mean()
```

Similarly, the average exam score for students who failed is calculated using:

```python
df[df["Final_Result"] == "Fail"]["Exam_Score"].mean()
```

These calculations allow the performance of passing and failing students to be compared.

---

# Filtering Student Data

One of the important Pandas concepts practiced in this project is filtering data based on multiple conditions.

The project finds students who:

- Study at least 7 hours
- Have attendance of at least 90%

The code is:

```python
df[
    (df["Study_Hours"] >= 7) &
    (df["Attendance"] >= 90)
]
```

The `&` operator means that both conditions must be satisfied.

This type of filtering is useful when analyzing specific groups of students.

---

# Pandas Concepts Learned

During this project, I practiced several important Pandas operations.

### `read_csv()`

Used to load a CSV file:

```python
pd.read_csv("student_data.csv")
```

### `head()`

Displays the first rows:

```python
df.head()
```

### `shape`

Returns the number of rows and columns:

```python
df.shape
```

### `columns`

Displays the column names:

```python
df.columns
```

### `dtypes`

Displays the data types:

```python
df.dtypes
```

### `info()`

Provides information about the DataFrame:

```python
df.info()
```

### `describe()`

Provides statistical information:

```python
df.describe()
```

### `isnull()`

Checks for missing values:

```python
df.isnull().sum()
```

### `mean()`

Calculates the average:

```python
df["Exam_Score"].mean()
```

### `max()`

Finds the highest value:

```python
df["Exam_Score"].max()
```

### `min()`

Finds the lowest value:

```python
df["Exam_Score"].min()
```

### `value_counts()`

Counts the occurrences of different values:

```python
df["Final_Result"].value_counts()
```

### `idxmax()`

Finds the index of the maximum value:

```python
df["Exam_Score"].idxmax()
```

### Boolean Filtering

Filters rows based on conditions:

```python
df[df["Exam_Score"] >= 80]
```

### Multiple Conditions

Filters using multiple conditions:

```python
df[
    (df["Study_Hours"] >= 7) &
    (df["Attendance"] >= 90)
]
```

---

# What I Learned

Through this project, I learned how to work with a dataset stored in a CSV file instead of manually creating data using NumPy.

I learned how to load a CSV file into a Pandas DataFrame and explore its structure using functions such as `head()`, `shape`, `columns`, `dtypes`, `info()`, and `describe()`.

I also learned how to check for missing values and perform basic statistical analysis.

Another important part of this project was learning how to filter data based on conditions. I practiced finding students based on their exam scores, study hours, attendance, and final results.

I also learned how to calculate the overall pass rate and compare the average exam scores of passing and failing students.

---

# Important Data Analysis Concepts

This project helped me understand that machine learning does not begin with training a model.

Before training a machine learning model, data usually needs to be:

```text
Collected
   ↓
Loaded
   ↓
Explored
   ↓
Checked for Problems
   ↓
Cleaned
   ↓
Analyzed
   ↓
Prepared
   ↓
Machine Learning
```

Understanding the dataset is an important part of the machine learning process.

---

# Project Purpose

This project is part of my learning journey toward Artificial Intelligence and Machine Learning.

My previous projects focused on NumPy, Pandas, statistics, visualization, Linear Regression, Logistic Regression, multiple features, and feature scaling.

In this project, I focused on working with a CSV dataset and learning the basic data analysis workflow using Pandas.

This project will provide the foundation for working with real datasets in future machine learning projects.

---

# Future Improvements

In future projects, I plan to:

- Use larger datasets
- Work with missing values
- Encode categorical variables
- Perform more advanced data cleaning
- Create visualizations
- Analyze relationships between features
- Prepare datasets for machine learning
- Train machine learning models using real-world data
- Compare different machine learning algorithms
- Evaluate models using multiple metrics

---

# Conclusion

This project helped me develop practical experience with Pandas and CSV datasets.

I learned how to load a dataset, inspect its structure, identify missing values, calculate statistics, find high and low values, analyze pass and fail results, and filter students based on multiple conditions.

Most importantly, I learned that understanding and preparing data is an important step before building a machine learning model.

This project represents the next step in my journey from learning individual machine learning algorithms to working with complete data science and machine learning workflows.