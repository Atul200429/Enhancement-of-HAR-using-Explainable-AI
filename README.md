# Enhancement of Human Activity Recognition Using Explainable AI

A machine learning system that classifies human activities (Walking, Sitting, Standing, Laying, Walking Upstairs/Downstairs) from smartphone sensor data, with built-in explainability using SHAP and LIME.

## 🎯 Project Overview

This project goes beyond standard activity classification by answering **why** a model makes a prediction — not just **what** it predicts. Using SHAP and LIME, every prediction can be traced back to the specific sensor features that drove it.

## 📊 Dataset

- **UCI HAR Dataset** (sourced via Kaggle)
- 7,352 training samples, 2,947 test samples
- 562 statistical features extracted from accelerometer & gyroscope signals
- 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

## 🤖 Models Implemented

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~96% |
| Decision Tree | ~86% |
| Random Forest | ~93% |
| SVM (RBF) | ~96% |
| AdaBoost | ~83% |
| **Voting Classifier (LR + SVC)** | **~97%** |

## 🔍 Explainability (XAI)

- **SHAP**: Global feature importance + local waterfall explanations using TreeExplainer
- **LIME**: Per-prediction local explanations showing which features support/oppose a prediction
- **Permutation & Gini Importance**: Cross-validated feature ranking
- **t-SNE**: 2D visualization of activity clusters

## 📈 Evaluation

- Confusion matrix analysis
- 5-fold cross-validation
- Multi-class AUC-ROC curves (all classes achieve AUC > 0.99)

## 🌐 Web Application

A Flask-based web app allows real-time predictions:
- Upload a CSV row of sensor data, or
- Try a random sample from the test set
- View predicted activity with confidence scores across all 6 classes

### Running locally
```bash
pip install flask scikit-learn pandas numpy joblib
python app.py
```
Then open `http://127.0.0.1:5000`

## 🛠️ Tech Stack

Python, scikit-learn, pandas, NumPy, SHAP, LIME, Flask, Matplotlib, Seaborn

## 👤 Author

**Atul Krishna**
B.Tech Computer Science and Information Technology, COER University, Roorkee