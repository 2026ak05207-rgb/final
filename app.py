import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("🧠 End-to-End Classification Model Evaluator - BITS ID: 2026AK05538")
st.write("This application loads pre-trained classification models to evaluate custom datasets. If the joblib files are missing from the root or model directory, the app dynamically fits the classifier in the background with the exact historical synthetic dataset seed.")

offline_results = [
    {"ML Model Name": "Logistic Regression", "Accuracy": 0.8467, "AUC": 0.9191, "Precision": 0.8581, "Recall": 0.8355, "F1": 0.8467, "MCC": 0.6936},
    {"ML Model Name": "Decision Tree", "Accuracy": 0.8300, "AUC": 0.8304, "Precision": 0.8531, "Recall": 0.8026, "F1": 0.8271, "MCC": 0.6614},
    {"ML Model Name": "KNN", "Accuracy": 0.9167, "AUC": 0.9774, "Precision": 0.9635, "Recall": 0.8684, "F1": 0.9135, "MCC": 0.8377},
    {"ML Model Name": "Naive Bayes", "Accuracy": 0.8467, "AUC": 0.9029, "Precision": 0.8955, "Recall": 0.7895, "F1": 0.8392, "MCC": 0.6988},
    {"ML Model Name": "Support Vector Machine (Winner)", "Accuracy": 0.9467, "AUC": 0.9902, "Precision": 0.9416, "Recall": 0.9539, "F1": 0.9477, "MCC": 0.8934}
]

st.sidebar.header("Configuration")

model_option = st.sidebar.selectbox(
    'Choose Classifier Model',
    ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
)

st.sidebar.subheader("Offline Training Benchmarks")
st.sidebar.dataframe(pd.DataFrame(offline_results), hide_index=True)

base_filename = f"{model_option.lower().replace(' ', '_')}.joblib"
model_path_root = base_filename
model_path_folder = f"model/{base_filename}"
model = None

if os.path.exists(model_path_root):
    try:
        model = joblib.load(model_path_root)
        st.sidebar.success(f"Loaded {model_option} from root folder!")
    except Exception as e:
        st.sidebar.error(f"Failed loading root model: {e}")
elif os.path.exists(model_path_folder):
    try:
        model = joblib.load(model_path_folder)
        st.sidebar.success(f"Loaded {model_option} from model/ directory!")
    except Exception as e:
        st.sidebar.error(f"Failed loading subdirectory model: {e}")

if model is None:
    st.sidebar.warning(f"{model_option} file not found or corrupted. Fitting model dynamically with seed parameters...")
    X_synth, y_synth = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.30, random_state=42, stratify=y_synth)
    
    fallback_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    model = fallback_models[model_option]
    model.fit(X_train, y_train)
    st.sidebar.success(f"Dynamically fitted {model_option} successfully!")

uploaded_file = st.file_uploader("Upload Test CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(test_data.head())

        if 'target' not in test_data.columns:
            st.error("The uploaded CSV must contain a column named 'target'.")
        else:
            X_val = test_data.drop(columns=['target'])
            y_val = test_data['target']

            expected_features = 15
            if X_val.shape[1] != expected_features:
                st.warning(f"Feature count mismatch! Expected {expected_features} columns but received {X_val.shape[1]}. Slicing or padding columns to ensure runtime safety...")
                if X_val.shape[1] < expected_features:
                    for idx in range(X_val.shape[1], expected_features):
                        X_val[f'feature_{idx}'] = 0.0
                else:
                    X_val = X_val.iloc[:, :expected_features]
            
            X_val.columns = [f'feature_{i}' for i in range(expected_features)]

            preds = model.predict(X_val)
            probas = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds

            acc = accuracy_score(y_val, preds)
            auc = roc_auc_score(y_val, probas)
            prec = precision_score(y_val, preds)
            rec = recall_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            mcc = matthews_corrcoef(y_val, preds)

            st.subheader(f"│̄̅ Evaluation Metrics for {model_option}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC Score", f"{auc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col2.metric("Recall", f"{rec:.4f}")
            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC Score", f"{mcc:.4f}")

            st.subheader(" ̄̅ Confusion Matrix & Classification Report")
            cc1, cc2 = st.columns(2)

            with cc1:
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_val, preds)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                st.pyplot(fig)

            with cc2:
                report = classification_report(y_val, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"An error occurred while parsing the dataset: {e}")
else:
    st.info("Please upload a test CSV dataset to begin evaluation.")
