# Student Exploratory Data Analysis (EDA)

## Description

This project is my Day 12 project in my Artificial Intelligence and Machine Learning learning journey.

The goal of this project is to perform Exploratory Data Analysis (EDA) on a student dataset using Python and Pandas.

In the previous project, I learned how to load a CSV dataset and perform basic analysis. In this project, I went a step further by investigating relationships between different student performance features, comparing passing and failing students, and preparing the data for a future machine learning model.

The project focuses on understanding the dataset before training a machine learning model.

---

## Dataset

The project uses a student performance dataset stored in a CSV file:

```text
student_data.csv
```

The dataset contains information about students, including:

- Student ID
- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score
- Final Result

The `Final_Result` column contains two categories:

```text
Pass
Fail
```

---

## Technologies Used

- Python
- Pandas
- CSV Dataset

---

## Project Goals

The main goals of this project are to:

1. Load a student dataset using Pandas.
2. Explore relationships between variables.
3. Calculate correlations.
4. Create a correlation matrix.
5. Compare passing and failing students.
6. Identify important features.
7. Create machine learning features (`X`).
8. Create the target variable (`y`).
9. Encode categorical results into numerical values.
10. Prepare the dataset for future machine learning models.

---

# Exploratory Data Analysis

## What is EDA?

Exploratory Data Analysis (EDA) is the process of examining and understanding a dataset before building a machine learning model.

EDA can help identify:

- Relationships between variables
- Patterns in the data
- Important features
- Differences between groups
- Potential problems with the dataset

The general workflow used in this project is:

```text
Dataset
   ↓
Load Data
   ↓
Explore Data
   ↓
Calculate Correlations
   ↓
Compare Groups
   ↓
Identify Important Features
   ↓
Create Features (X)
   ↓
Create Target (y)
   ↓
Prepare for Machine Learning
```

---

# Correlation Analysis

Correlation measures the relationship between two numerical variables.

A correlation value generally ranges from:

```text
-1 to +1
```

A value close to:

```text
+1
```

indicates a strong positive relationship.

A value close to:

```text
0
```

indicates a weak or limited linear relationship.

A value close to:

```text
-1
```

indicates a strong negative relationship.

---

## Study Hours vs Exam Score

The project calculates the correlation between study hours and exam scores:

```python
df["Study_Hours"].corr(df["Exam_Score"])
```

This helps determine whether students who study more tend to achieve higher exam scores.

---

## Attendance vs Exam Score

The project calculates:

```python
df["Attendance"].corr(df["Exam_Score"])
```

This helps examine the relationship between attendance and exam performance.

---

## Assignment Score vs Exam Score

The project calculates:

```python
df["Assignment_Score"].corr(df["Exam_Score"])
```

This helps determine whether students with higher assignment scores tend to achieve higher exam scores.

---

## Midterm Score vs Exam Score

The project calculates:

```python
df["Midterm_Score"].corr(df["Exam_Score"])
```

This helps examine whether midterm performance is related to final exam performance.

---

# Correlation Matrix

The project also creates a correlation matrix for the main numerical features.

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

The correlation matrix allows multiple relationships to be examined at the same time.

The features analyzed are:

- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score

This is useful for identifying which variables may be important predictors for future machine learning models.

---

# Group Analysis

The project compares student performance based on their final result.

The two groups are:

```text
Pass
Fail
```

---

## Average Study Hours by Result

The project calculates:

```python
df.groupby("Final_Result")["Study_Hours"].mean()
```

This allows the average study time of passing and failing students to be compared.

---

## Average Attendance by Result

The project calculates:

```python
df.groupby("Final_Result")["Attendance"].mean()
```

This helps determine whether attendance differs between passing and failing students.

---

## Average Assignment Score by Result

The project calculates:

```python
df.groupby("Final_Result")["Assignment_Score"].mean()
```

This allows assignment performance to be compared between the two groups.

---

## Average Midterm Score by Result

The project calculates:

```python
df.groupby("Final_Result")["Midterm_Score"].mean()
```

This helps compare midterm performance between passing and failing students.

---

## Average Exam Score by Result

The project calculates:

```python
df.groupby("Final_Result")["Exam_Score"].mean()
```

This shows the difference in average exam performance between students who passed and failed.

---

# Feature Preparation

One of the most important parts of this project is preparing the dataset for machine learning.

## Features (X)

The following variables are used as features:

```text
Study Hours
Attendance
Assignment Score
Midterm Score
Exam Score
```

They are stored in `X`:

```python
X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
]
```

Features are the input variables that a machine learning model can use to make predictions.

---

# Target Variable (y)

The target variable is:

```text
Final Result
```

The target represents what the machine learning model will eventually try to predict.

The original values are:

```text
Pass
Fail
```

These values are converted into numerical values:

```text
Pass → 1
Fail → 0
```

The conversion is performed using:

```python
y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})
```

This process is called **encoding**.

---

# X and y

After preparing the data:

```text
X = Features
y = Target
```

In this project:

```text
X:
- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score

y:
- 1 = Pass
- 0 = Fail
```

The shapes of `X` and `y` are also checked:

```python
print(X.shape)
print(y.shape)
```

This helps verify that the features and target contain the expected number of student records.

---

# Dataset Summary

The project also calculates the average of each numerical feature:

```python
df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
].mean()
```

This provides a quick overview of the typical values in the dataset.

---

# Pandas Concepts Practiced

During this project, I practiced several important Pandas concepts.

### Loading a CSV

```python
pd.read_csv()
```

### Correlation

```python
df["column1"].corr(df["column2"])
```

### Correlation Matrix

```python
df[columns].corr()
```

### Grouping Data

```python
df.groupby()
```

### Calculating Averages

```python
.mean()
```

### Selecting Columns

```python
df[["Column1", "Column2"]]
```

### Creating Features

```python
X = df[features]
```

### Creating a Target

```python
y = df["Final_Result"]
```

### Encoding Categories

```python
.map({
    "Pass": 1,
    "Fail": 0
})
```

### Checking Data Shape

```python
X.shape
y.shape
```

---

# What I Learned

Through this project, I learned the importance of Exploratory Data Analysis before building a machine learning model.

I learned how to investigate relationships between variables using correlation and how to use a correlation matrix to examine several variables at once.

I also learned how to compare different groups of data using `groupby()`.

Another important lesson was understanding the difference between **features** and the **target**.

Features are the information that a machine learning model uses as input, while the target is the outcome that the model is trying to predict.

I also learned how to convert categorical values such as `Pass` and `Fail` into numerical values using encoding.

---

# Machine Learning Preparation

This project does not train a machine learning model.

Instead, it prepares the data for a future model.

The preparation process is:

```text
Student Dataset
      ↓
Exploratory Data Analysis
      ↓
Correlation Analysis
      ↓
Group Analysis
      ↓
Feature Selection
      ↓
Create X
      ↓
Create y
      ↓
Encode Target
      ↓
Ready for Machine Learning
```

This is an important step because understanding the data before training a model can help determine which features may be useful.

---

# Project Purpose

This project is part of my ongoing learning journey toward Artificial Intelligence and Machine Learning.

My previous projects focused on:

- Python
- NumPy
- Pandas
- Statistics
- Data analysis
- Linear Regression
- Logistic Regression
- Multiple features
- Feature scaling

This project builds on those skills by introducing Exploratory Data Analysis and preparing a real CSV dataset for machine learning.

---

# Future Improvements

In future projects, I plan to:

- Visualize the correlation matrix
- Perform more advanced data cleaning
- Handle missing values
- Explore outliers
- Perform feature selection
- Split the dataset into training and testing sets
- Scale numerical features
- Train machine learning models
- Compare different algorithms
- Evaluate model performance
- Make predictions for new students

---

# Conclusion

This project helped me understand that machine learning is not only about training models.

Before building a model, it is important to understand the dataset, explore relationships between variables, compare different groups, select useful features, and prepare the target variable.

This project represents an important step in my transition from learning individual machine learning algorithms to understanding the complete machine learning workflow.