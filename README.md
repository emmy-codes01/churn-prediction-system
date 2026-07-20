# ML AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecommunications customer is likely to **churn (leave)** or **stay** based on customer demographics, subscription details, and billing information.

The project demonstrates the complete Machine Learning workflow—from data exploration and preprocessing to model training, evaluation, deployment, and a user-friendly web application built with Streamlit.

---

## Live Demo

**Application:** https://your-streamlit-app.streamlit.app

**GitHub Repository:** https://github.com/emmy-codes01/churn-prediction-system

---

## Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers is often more expensive than acquiring new ones.

This project leverages Machine Learning to analyze historical customer data and predict customers who are at risk of leaving the service. Such predictions help businesses take proactive retention measures before customers churn.

---

## Objectives

- Analyze customer churn patterns.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess customer data.
- Train and evaluate a Machine Learning model.
- Deploy the trained model using Streamlit.
- Provide an interactive interface for real-time churn prediction.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information including:

- Customer demographics
- Internet services
- Contract type
- Monthly charges
- Total charges
- Customer tenure
- Payment method
- Technical support
- Online security
- Streaming services
- Churn status

### Dataset Statistics

- Total Records: **7,043**
- Features: **21**
- Target Variable: **Churn**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## Exploratory Data Analysis

Several insights were obtained from the dataset:

- Most customers remained with the company.
- Customers with higher monthly charges showed a higher tendency to churn.
- The dataset contains both categorical and numerical variables requiring preprocessing.
- Customer churn is an imbalanced classification problem.

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Converted `TotalCharges` to numeric values.
- Filled missing values.
- Removed `customerID`.
- Encoded categorical variables using One-Hot Encoding.
- Converted churn labels into binary values.
- Split the dataset into training and testing sets.

---

## Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

The Random Forest model was trained to classify customers into:

- **0 → Customer Stays**
- **1 → Customer Churns**

---

## Model Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.48%** |
| Precision | **67.72%** |
| Recall    | **45.99%** |
| F1-Score  | **54.78%** |

---

## Confusion Matrix

| Actual | Predicted Stay | Predicted Churn |
| ------ | -------------: | --------------: |
| Stay   |            953 |              82 |
| Churn  |            202 |             172 |

---

## Streamlit Application

The deployed application allows users to:

- Enter customer information.
- Predict customer churn(exit).
- View churn probability.
- Receive an instant prediction result.

---

## Screenshots

### Home Page

```text
images/home.png
```

### Prediction Result

```text
images/prediction.png
```

---

## Project Structure

```text
churn-prediction-system/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/emmy-codes01/churn-prediction-system.git
```

Navigate into the project:

```bash
cd churn-prediction-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Improve recall through hyperparameter tuning.
- Add more customer input fields.
- Compare additional models such as XGBoost and LightGBM.
- Integrate customer retention recommendations.
- Store prediction history in a database.

---

## Author

**Moyinoluwa Emmanuel Ayeni**

Machine Learning Student | Software Dev | Brand Identity Designer

**GitHub:** https://github.com/emmy-codes01

**LinkedIn:** https://linkedin.com/in/your-emmanuel-ayeni01

---

## License

This project is released under the MIT License and is intended for educational and portfolio purposes.
