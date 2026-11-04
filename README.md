# Dry Bean Classification Using Machine Learning

**BITS Pilani – M.Tech Artificial Intelligence and Machine Learning**

**Course:** Machine Learning  
**Assignment:** Assignment 2

---

## 1. Problem Statement

The objective of this project is to classify seven varieties of dry beans using 16 numerical geometric features.

Five machine learning classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

An interactive Streamlit application is developed to demonstrate and compare the trained models.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/602/dry+bean

### Dataset Overview

| Parameter | Value |
|---|---:|
| Original observations | 13,611 |
| Duplicate records removed | 68 |
| Final observations | 13,543 |
| Input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Missing values | 0 |

### Input Features

Area, Perimeter, MajorAxisLength, MinorAxisLength,
AspectRatio, Eccentricity, ConvexArea, EquivDiameter,
Extent, Solidity, Roundness, Compactness,
ShapeFactor1, ShapeFactor2, ShapeFactor3 and ShapeFactor4.

### Target Classes

BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

### Data Preprocessing

- Checked for missing values.
- Identified 68 excess duplicate records.
- Verified that duplicate feature combinations had no conflicting class labels.
- Removed exact duplicate records.
- Investigated feature distributions using histograms.
- Investigated feature correlations using Pearson correlation.
- Investigated potential outliers using the IQR method.
- Retained all 16 input features.

The IQR analysis flagged 3,004 observations (22.18%) with at least one potential outlier. These observations were retained because statistical outliers are not necessarily erroneous measurements.

### Train/Test Split

An 80:20 stratified train/test split was performed.

| Dataset | Observations |
|---|---:|
| Training | 10,834 |
| Testing | 2,709 |
| Total | 13,543 |

Random state: 42  
Stratification: Enabled

### Feature Scaling

StandardScaler was integrated into pipelines for:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

Decision Tree and Gaussian Naive Bayes were implemented without scaling.

Scaling parameters were learned using training data only.

---

## 3. GitHub Repository Link

**Repository URL:** https://github.com/2026aj05145-pixel/dry-bean-classification-ml

The repository contains the application source code, trained models, requirements, test dataset and documentation.

---

## 4. Models Used and Evaluation

All five models were evaluated on the same held-out test dataset containing 2,709 observations.

### Evaluation Metrics

1. Accuracy – Proportion of correctly classified observations.
2. AUC – Multiclass area under the ROC curve, calculated using One-vs-Rest and weighted averaging.
3. Precision – Reliability of predicted class labels.
4. Recall – Proportion of actual class observations correctly identified.
5. F1 Score – Harmonic mean of precision and recall.
6. Matthews Correlation Coefficient (MCC) – Overall classification quality considering correct and incorrect predictions.

### Model Configurations

**Logistic Regression**
- Solver: lbfgs
- Maximum iterations: 2,000
- StandardScaler: Applied

**Decision Tree**
- Criterion: Gini
- Random state: 42
- StandardScaler: Not applied

**KNN**
- Number of neighbors: 5
- Distance metric: Euclidean
- Weights: Uniform
- StandardScaler: Applied

**Gaussian Naive Bayes**
- Classifier: GaussianNB
- Default parameters
- StandardScaler: Not applied

**Support Vector Machine**
- Kernel: RBF
- C: 1.0
- Gamma: scale
- Probability estimation: Enabled
- StandardScaler: Applied

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Observations on Model Performance

**Logistic Regression**

Achieved 91.95% accuracy and the highest AUC of 0.9935. It demonstrated strong and balanced classification performance across the seven classes. SIRA was comparatively challenging, with an F1-score of 0.8721.

**Decision Tree**

Achieved 89.55% accuracy. It performed well for BOMBAY and HOROZ but showed more classification errors for SIRA, which had an F1-score of 0.8248. Its overall performance was lower than Logistic Regression, KNN and SVM.

**K-Nearest Neighbors**

Achieved 91.55% accuracy and 0.9811 AUC. It performed competitively with Logistic Regression and SVM. Feature scaling was important because KNN relies on distance calculations.

**Gaussian Naive Bayes**

Achieved 76.30% accuracy and the lowest MCC of 0.7143. BARBUNYA and SEKER were particularly challenging classes, with F1-scores of 0.5483 and 0.6828 respectively. Strong correlations between input features may be one possible explanation for weaker performance because Gaussian Naive Bayes assumes conditional feature independence; this dataset alone does not establish the cause.

**Support Vector Machine**

Achieved the highest accuracy of 92.10%, precision of 0.9214, recall of 0.9210, weighted F1-score of 0.9211 and MCC of 0.9045. It demonstrated the strongest overall baseline classification performance.

### Overall Winner

**Support Vector Machine (RBF Kernel)**

SVM was selected as the overall winner based on its highest accuracy, precision, recall, weighted F1-score and MCC.

Logistic Regression achieved the highest AUC score of 0.9935, marginally above SVM's 0.9933.

Both models demonstrated excellent classification performance. Based on the overall evaluation metrics, SVM is selected as the best-performing baseline classifier for this dataset.

---

## 5. Streamlit Application

The project includes an interactive web application developed using Streamlit.

### Application Features

1. Preloaded test dataset for immediate evaluation.
2. Option to upload another labelled test dataset in CSV format.
3. Evaluation of all five saved classification models.
4. Comparison of all six evaluation metrics.
5. Individual model analysis using separate tabs.
6. Confusion matrix for each classifier.
7. Classification report for each classifier.
8. Side-by-side prediction comparison across all five models.
9. Downloadable prediction results.

### Application Behaviour

The application loads previously trained models from the `model/` directory.

The uploaded dataset is used for evaluation and prediction. No model retraining takes place on uploaded test data.

A labelled CSV containing the 16 required input features and the target column `Class` is required for evaluation metrics and confusion matrices.

**Live Streamlit Application:** https://dry-bean-classification-ml-gujgsxwd8wwuhncnetvrsg.streamlit.app/

---

## 6. Project Structure

```text
dry-bean-classification-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── 2026AJ05145.ipynb
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## 7. How to Run the Application

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start Streamlit

```bash
streamlit run app.py
```

### Step 3: Open the application

Open the local URL displayed by Streamlit in your browser. The preloaded test dataset will be available automatically.

---

## 8. Conclusion

This project implemented and evaluated five machine learning classification algorithms using the UCI Dry Bean dataset.

All five models were evaluated using the same training and testing partitions and six classification metrics.

SVM achieved the best overall baseline performance, with an accuracy of 92.10% and an MCC of 0.9045.

Logistic Regression demonstrated very competitive performance and achieved the highest AUC of 0.9935.

The project demonstrates an end-to-end machine learning workflow covering data exploration, preprocessing, model training, evaluation, model serialization and deployment through an interactive Streamlit application.
