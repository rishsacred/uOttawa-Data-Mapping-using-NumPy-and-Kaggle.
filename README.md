# uOttawa-Data-Mapping-using-NumPy-and-Kaggle.
# ❤️ Heart Disease Prediction using Random Forest

## 📌 Project Overview

This project builds a supervised Machine Learning model to predict the presence of heart disease using structured clinical data.

The model is trained using a **Random Forest Classifier** from scikit-learn and evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

This project demonstrates:
- Data preprocessing and cleaning
- Exploratory data analysis
- Supervised classification
- Model evaluation
- Feature importance analysis
- Data visualization

> ⚠️ This project is for educational purposes only and is NOT intended for medical diagnosis.

---

## 📊 Dataset Description

The dataset contains patient health information and a binary target variable indicating heart disease presence.

### Features Included

| Feature | Description |
|----------|------------|
| Age | Age of patient |
| Sex | 1 = Male, 0 = Female |
| Chest pain type | Type of chest pain (1–4) |
| BP | Resting blood pressure |
| Cholesterol | Serum cholesterol level |
| FBS over 120 | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| EKG results | Resting electrocardiographic results |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina (1 = Yes, 0 = No) |
| ST depression | ST depression induced by exercise |
| Slope of ST | Slope of peak exercise ST segment |
| Number of vessels fluro | Number of major vessels colored by fluoroscopy |
| Thallium | Thallium stress test result |
| Heart Disease | Target variable (Presence / Absence) |

---

## 🧹 Data Preprocessing

The script performs the following preprocessing steps:

- Removes whitespace from column names
- Renames `Heart Disease` column to `target`
- Cleans target values
- Converts:
  - `"Presence"` → `1`
  - `"Absence"` → `0`
- Checks for missing values
- Splits data into training (80%) and testing (20%)

---

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Classifier  
**Library:** scikit-learn  
**Random State:** 42  

### Why Random Forest?

- Handles structured tabular data well
- Reduces overfitting compared to single decision trees
- Provides feature importance rankings
- Works well without heavy parameter tuning

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-score
- Confusion Matrix

The script also:
- Generates visualizations
- Saves output images
- Performs a sample prediction

---

## 📊 Output Visualizations

The following files are automatically generated:

- `target_distribution.png`
- `confusion_matrix.png`
- `feature_importance.png`

These are saved in the project directory after running the script.

---

# 🛠️ Installation & Setup Guide (Step-by-Step)

This section explains everything from scratch.

---

## ✅ Step 1 — Install Python

Make sure Python 3.9+ is installed.

Check version:

bash
python --version


If not installed, download from:
https://www.python.org/downloads/

---

## ✅ Step 2 — Clone or Download the Project

If using Git:

bash
git clone <your-repository-link>
cd project-folder


Or download as ZIP and extract.

---

## ✅ Step 3 — Install Required Libraries

Run this in your project directory:

bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn


This installs:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To verify installation:

```bash
python -m pip list
```

---

## ✅ Step 4 — Ensure Dataset is Present

Place:

```
heart.csv
```

in the same folder as:

```
heart_model.py
```

Folder structure should look like:

```
project-folder/
│
├── heart.csv
├── heart_model.py
└── README.md
```

---

## ✅ Step 5 — Run the Program

Execute:

```bash
python heart_model.py
```

---

## 📌 What Happens When You Run It?

The script will:

1. Load and clean the dataset
2. Train a Random Forest model
3. Evaluate performance
4. Print metrics to terminal
5. Save visualizations as PNG files

---

## 📊 Example Output

```
Accuracy: 0.84

Classification Report:
              precision    recall    f1-score   support
           0       0.82       0.86       0.84
           1       0.86       0.82       0.84

