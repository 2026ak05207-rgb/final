# BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2

## 1. Project Overview
This project presents an end-to-end Machine Learning pipeline developed as part of the Machine Learning Assignment - 2 (WILP Division). It features synthetic dataset generation, competitive offline evaluation of five classification algorithms, model serialization, and interactive web deployment using Streamlit Community Cloud.

### Core Deliverables:
- **GitHub Repository** with complete source code (`app.py`), package dependencies (`requirements.txt`), comprehensive documentation (`README.md`), and test data (`test_data.csv`).
- **Live Web Application** deployed on Streamlit Community Cloud.
- **Zero-Downtime Design:** Built with a smart dynamic fallback mechanism that re-trains models on-the-fly if pre-trained binaries are absent from the host environment.

---

## 2. Dataset Characteristics
To perform robust binary classification, we generated a high-dimensional synthetic dataset matching assignment specifications:
- **Total Instances:** 1,000
- **Total Features:** 15 (10 informative features, 5 redundant features for complexity as per PDF report)
- **Test Split:** 30% stratified split (300 instances) saved as `test_data.csv`
- **Target Class:** Binary (0 or 1)

---

## 3. Machine Learning Models & Evaluation Comparison
We trained and validated 5 distinct classification models. Each model's final testing performance is summarized below:

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8467 | 0.9191 | 0.8581 | 0.8355 | 0.8467 | 0.6936 |
| **Decision Tree** | 0.8300 | 0.8304 | 0.8531 | 0.8026 | 0.8271 | 0.6614 |
| **KNN** | 0.9167 | 0.9774 | 0.9635 | 0.8684 | 0.9135 | 0.8377 |
| **Naive Bayes** | 0.8467 | 0.9029 | 0.8955 | 0.7895 | 0.8392 | 0.6988 |
| **Support Vector Machine (Winner)** | **0.9467** | **0.9902** | **0.9416** | **0.9539** | **0.9477** | **0.8934** |

### Model Performance Insights:
- **Support Vector Machine (SVM)** achieved the highest performance overall with an **Accuracy of 94.67%** and an **MCC of 0.8934**.
- **K-Nearest Neighbors (KNN)** achieved highly competitive results (91.67% Accuracy) and shared the highest **Precision (96.35%)**.

---

## 4. Web Application Features
The deployed Streamlit application incorporates the following interactive features:
1. **Dynamic File Upload (.csv):** Accepts user-uploaded test dataset for model evaluation.
2. **Model Selector Dropdown:** Allows the evaluator to toggle between any of the 5 trained models.
3. **Metrics Panel:** Displays dynamically computed metrics (Accuracy, AUC, Precision, Recall, F1, and MCC).
4. **Visual Interpretations:** Renders a Confusion Matrix heatmap and raw Classification Report on the uploaded test CSV data.

---

## 5. Deployment Instructions (Streamlit Community Cloud)
1. Fork or upload this flat directory directly to a public GitHub repository.
2. Ensure all files (`app.py`, `requirements.txt`, etc.) are placed at the **root** folder level (no nesting directories).
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
4. Select your Repository, Branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy** to publish the interactive dashboard!
