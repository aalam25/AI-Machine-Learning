# Random Forest ROC-AUC Classification

## Project Overview

This project demonstrates how to evaluate a **Random Forest classification model** using the **ROC Curve** and **AUC (Area Under the Curve)**.

The model uses student academic data to predict whether a student will **Pass** or **Fail**.

This is **Day 24** of my Machine Learning learning journey.

The main focus of this project is understanding how a classification model can be evaluated using prediction probabilities rather than only looking at the final predicted class.

---

## Project Goal

The main objectives of this project are to:

- Train a Random Forest classification model.
- Generate prediction probabilities using `predict_proba()`.
- Calculate the False Positive Rate (FPR).
- Calculate the True Positive Rate (TPR).
- Calculate ROC curve thresholds.
- Calculate the AUC score.
- Create and save an ROC curve visualization.
- Understand the difference between accuracy and ROC-AUC.

---

## Project Files

```text
random-forest-roc-auc/
│
├── roc_auc.py
├── student_data.csv
├── roc_curve.png
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest
- ROC Curve
- AUC
- Git & GitHub

---

## Dataset

The dataset contains **50 student records** and the following columns:

| Column | Description |
|---|---|
| `Student_ID` | Unique student identifier |
| `Study_Hours` | Number of hours studied |
| `Attendance` | Student attendance percentage |
| `Assignment_Score` | Assignment score |
| `Midterm_Score` | Midterm examination score |
| `Exam_Score` | Final examination score |
| `Final_Result` | Pass or Fail |

The model uses four features:

- `Study_Hours`
- `Attendance`
- `Assignment_Score`
- `Exam_Score`

The target variable is:

- `Pass` → 1
- `Fail` → 0

---

## Machine Learning Process

The project follows these steps:

1. Load the student dataset.
2. Select the input features.
3. Convert `Pass` and `Fail` into numerical values.
4. Split the dataset into training and testing data.
5. Train a Random Forest classifier.
6. Generate prediction probabilities.
7. Calculate the ROC curve values.
8. Calculate the AUC score.
9. Create and save the ROC curve visualization.

---

## Random Forest Model

The model was created using:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model was trained using:

```python
model.fit(
    X_train,
    y_train
)
```

The dataset was split into:

- Training data: `(40, 4)`
- Testing data: `(10, 4)`

The Random Forest model was successfully trained.

---

## Prediction Probabilities

Instead of using only:

```python
model.predict(X_test)
```

this project uses:

```python
model.predict_proba(X_test)[:, 1]
```

`predict_proba()` returns the model's probability for each class.

The `[:, 1]` selects the probability of the positive class, which is:

```text
Pass = 1
```

The actual prediction probabilities from this project were:

```text
[1.   1.   1.   0.   1.   1.   1.   0.02 1.   0.29]
```

These probabilities are used to calculate the ROC curve and AUC.

---

## ROC Curve

ROC stands for **Receiver Operating Characteristic**.

A ROC curve compares:

- **True Positive Rate (TPR)**
- **False Positive Rate (FPR)**

The ROC curve can be calculated using:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)
```

### False Positive Rate

FPR measures the proportion of actual negative cases that are incorrectly classified as positive.

The actual FPR values were:

```text
[0. 0. 0. 1.]
```

### True Positive Rate

TPR measures the proportion of actual positive cases that are correctly identified.

TPR is also known as **Recall** or **Sensitivity**.

The actual TPR values were:

```text
[0.    0.875 1.    1.   ]
```

### Thresholds

The model uses different probability thresholds when calculating the ROC curve.

The actual thresholds were:

```text
[ inf 1.   0.29 0.  ]
```

The `inf` value represents the starting threshold used by the ROC calculation.

---

## AUC Score

AUC stands for **Area Under the Curve**.

It summarizes the model's ability to distinguish between the two classes across different classification thresholds.

The AUC was calculated using:

```python
auc_score = roc_auc_score(
    y_test,
    y_probability
)
```

### Result

```text
AUC Score:
1.0000
```

For this particular test set, the model achieved an **AUC of 1.0000**.

This means the model perfectly separated the positive and negative examples in this particular evaluation set.

However, the test set contains only **10 samples**, so this result should not be treated as proof that the model would perform perfectly on a larger or different dataset.

---

## ROC Curve Visualization

The project creates an ROC curve using Matplotlib.

The graph includes:

- Random Forest ROC curve
- Random classifier reference line
- False Positive Rate on the x-axis
- True Positive Rate on the y-axis
- AUC value in the legend

The graph is saved as:

```text
roc_curve.png
```

The graph provides a visual representation of the model's classification performance across different probability thresholds.

---

## Understanding AUC

AUC can be interpreted as a measure of how well a classifier separates two classes.

In general:

| AUC | General Interpretation |
|---:|---|
| 1.0 | Perfect separation on the evaluated data |
| 0.9–0.99 | Very strong separation |
| 0.8–0.89 | Good separation |
| 0.7–0.79 | Moderate separation |
| 0.5 | Approximately random discrimination |
| Below 0.5 | Poor discrimination |

These ranges are general guidelines rather than universal rules.

For this project:

```text
AUC = 1.0000
```

The model achieved perfect separation on the 10-sample test set.

---

## Accuracy vs. ROC-AUC

Accuracy and ROC-AUC measure different aspects of a classification model.

### Accuracy

Accuracy measures the percentage of predictions that are correct at a particular classification threshold.

```text
Accuracy = Correct Predictions / Total Predictions
```

### ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between the two classes across different classification thresholds.

This means ROC-AUC uses the model's probability scores rather than relying on only one threshold.

---

## Important Lesson About the Result

Although this project produced an AUC of **1.0000**, the dataset is relatively small.

There are:

- 50 total records
- 40 training records
- 10 testing records

Because the test set contains only 10 samples, the AUC can be sensitive to individual predictions.

Therefore, the result should be interpreted as:

> The Random Forest achieved perfect class separation on this particular 10-sample test set.

It should not automatically be interpreted as perfect real-world performance.

A larger dataset and additional validation would provide a more reliable estimate of generalization performance.

---

## Key Functions Learned

### `predict_proba()`

```python
y_probability = model.predict_proba(X_test)[:, 1]
```

Returns the model's probability estimates for the classes.

---

### `roc_curve()`

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)
```

Calculates the False Positive Rate, True Positive Rate, and classification thresholds used to construct the ROC curve.

---

### `roc_auc_score()`

```python
auc_score = roc_auc_score(
    y_test,
    y_probability
)
```

Calculates the Area Under the ROC Curve.

---

## Complete Workflow

The overall workflow can be summarized as:

```text
Student Dataset
       ↓
Select Features
       ↓
Convert Pass/Fail to 1/0
       ↓
Train/Test Split
       ↓
Random Forest
       ↓
Prediction Probabilities
       ↓
ROC Curve
       ↓
AUC Score
```

---

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/aalam25/random-forest-roc-auc.git
```

### Step 2: Open the project folder

```bash
cd random-forest-roc-auc
```

### Step 3: Install the required libraries

```bash
pip install pandas scikit-learn matplotlib
```

### Step 4: Run the Python program

```bash
python roc_auc.py
```

The program will:

- Load the dataset.
- Train the Random Forest model.
- Generate prediction probabilities.
- Calculate FPR.
- Calculate TPR.
- Display thresholds.
- Calculate AUC.
- Create the ROC curve.
- Save the graph as `roc_curve.png`.

---

## Actual Output

```text
Training Data Shape:
(40, 4)

Testing Data Shape:
(10, 4)

Random Forest Model Trained Successfully.

Prediction Probabilities:
[1.   1.   1.   0.   1.   1.   1.   0.02 1.   0.29]

False Positive Rate:
[0. 0. 0. 1.]

True Positive Rate:
[0.    0.875 1.    1.   ]

Thresholds:
[ inf 1.   0.29 0.  ]

AUC Score:
1.0000
```

---

## What I Learned

Through this project, I learned how to:

- Evaluate a binary classification model.
- Generate prediction probabilities.
- Understand probability thresholds.
- Calculate False Positive Rate.
- Calculate True Positive Rate.
- Generate ROC curve values.
- Calculate AUC.
- Visualize ROC performance.
- Understand the difference between accuracy and ROC-AUC.
- Interpret a perfect AUC result carefully when using a small test set.

---

## Machine Learning Journey

This project is part of my ongoing Machine Learning portfolio.

Previous projects covered:

- Decision Tree Classification
- Random Forest Classification
- Model Evaluation
- Confusion Matrix
- Feature Importance
- Decision Tree vs. Random Forest
- Hyperparameter Tuning
- Cross-Validation
- GridSearchCV
- Random Forest Evaluation
- Random Forest Feature Importance
- Permutation Feature Importance

Day 24 builds on these concepts by introducing **ROC Curve and AUC** for classification evaluation.

---

## Future Improvements

Possible improvements for this project include:

- Use a larger dataset.
- Increase the test set size.
- Use stratified cross-validation.
- Compare ROC-AUC across multiple models.
- Calculate Precision-Recall curves.
- Compare Random Forest with Logistic Regression.
- Evaluate model performance on unseen datasets.

---

## Author

**Abone Alam**

Computer Science Student | Aspiring Machine Learning / AI Professional

GitHub: https://github.com/aalam25

---

## Conclusion

Day 24 helped me understand that classification models can be evaluated in more ways than simply checking whether their predictions are correct.

By using prediction probabilities, ROC curves, and AUC, I can examine how well a model separates different classes across multiple decision thresholds.

The Random Forest achieved an **AUC of 1.0000 on this particular 10-sample test set**. The result is encouraging, but the small test set means that further validation with more data would be necessary to understand how well the model generalizes.

This project strengthened my understanding of **classification evaluation, probability-based predictions, ROC curves, and model performance analysis**.