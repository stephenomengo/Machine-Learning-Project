# Machine-Learning-Project
# Financial Health Prediction — ML Project

An end-to-end machine learning project that predicts an individual's **financial status trajectory** (*Improved*, *Stayed the same*, or *Worsened*) using financial inclusion, demographic, and socioeconomic data from the **2024 FinAccess Household Survey**.

---

## 📌 Project Overview

Financial health is a multidimensional concept influenced by income, savings behavior, access to formal financial services, mobile money usage, and various socioeconomic barriers. This project leverages the FinAccess 2024 dataset to build and evaluate multiple classification models capable of predicting whether a person's financial situation has improved, stayed the same, or worsened.

The final model was deployed with **Streamlit** for interactive predictions.

---

## 🎯 Objectives

- Explore and clean a large, real-world financial inclusion dataset (~20,000+ records).
- Identify key drivers of financial status through exploratory data analysis (EDA) and hypothesis testing.
- Build and compare multiple machine learning classification models.
- Deploy the best-performing model via an interactive Streamlit app.

---

## 📂 Dataset

**Source:** `finaccess2024_datasprint.xlsx` (FinAccess Household Survey 2024 — Kenya)

- **Rows:** ~20,871 (before cleaning)
- **Columns:** 28
- **Target Variable:** `financial_status` (3 classes: `Improved`, `Stayed the same`, `Worsened`)

### Feature Categories

| Category | Example Features |
|---|---|
| **Demographics** | `county`, `location_type`, `Sex`, `Age`, `household_size`, `education_level`, `marital_status`, `has_disability` |
| **Financial Inclusion** | `Savings_formal`, `Savings_informal`, `Loan_formal`, `Loan_informal`, `formal_service_use`, `defaulted` |
| **Digital Finance** | `mobile_money_access`, `mobile_ownership_1`, `barriers_mobile_money` |
| **Financial Literacy & Resilience** | `fl_score`, `nfhi_11`, `nfhi_12`, `nfhi_13`, `accessto_13k_1month`, `not_difficult` |
| **Income & Shocks** | `monthly_income`, `prodsum1`, `experienced_shock`, `barriers_bank` |

---

## 🛠️ Tech Stack

- **Python 3.13**
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical Testing:** `scipy.stats` (Chi-Square, Kruskal-Wallis)
- **Machine Learning:** `scikit-learn`
- **Model Persistence:** `joblib`
- **Deployment:** `streamlit`

---

## 🔍 Project Workflow

### 1. Data Loading & Overview
- Loaded dataset, checked shape, dtypes, duplicates, and missing values.
- Identified categorical vs. numerical features.

### 2. Data Cleaning
- Dropped duplicates.
- Standardized `education_level` categories (removed quotes, merged technical training variants).
- Cleaned `marital_status` (removed "Refused to Answer" / "Don't Know").
- Handled missing values in `barriers_bank` using **mode imputation**.
- Replaced numeric `0` in `barriers_mobile_money` with `"None"`.

### 3. Exploratory Data Analysis (EDA)
- Boxplots and histograms for numerical features.
- Count plots comparing `financial_status` against `Age`, `education_level`, `has_disability`, and `Savings_formal`.
- Correlation heatmap (Spearman) among numerical features.

### 4. Hypothesis Testing

**Categorical vs. Target (Chi-Square Test):**
> All 24 categorical features showed statistically significant associations with `financial_status` (p ≤ 0.05).

**Numerical vs. Target (Kruskal-Wallis Test):**
> `household_size`, `monthly_income`, and `prodsum1` all showed significant distributional differences across financial status groups.

### 5. Preprocessing
- **Train/Test Split:** 70/30 with stratification.
- **Ordinal Encoding:** `Age` (ordered: `16-17` → `Above 55`).
- **One-Hot Encoding:** All other categorical features (`drop_first=True`).
- **Label Encoding:** Target variable.
- **Scaling:** `RobustScaler` on numerical features.
- **Column Alignment:** Ensured train/test feature consistency.

### 6. Model Training & Evaluation

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| **Logistic Regression** | **0.57** | **0.61** | 0.46 |
| K-Nearest Neighbors (K=5) | 0.46 | 0.47 | 0.41 |
| Random Forest (200 trees, depth=10) | 0.56 | 0.64 | 0.39 |
| Gradient Boosting | 0.57 | 0.62 | 0.44 |
| AdaBoost | — | — | — |

> **Note:** The dataset is imbalanced (Worsened ≈ 73%), so weighted F1 and macro F1 are more informative than raw accuracy.

### 7. Model Deployment
- The **Logistic Regression** model was saved with `joblib` alongside the `LabelEncoder`.
- A **Streamlit app** loads the model and predicts financial status based on user inputs.
- Feature alignment is handled programmatically to match `model.feature_names_in_`.

---

## 📁 Project Structure

```
financess-ml-project/
│
├── Financess ML-Project.ipynb        # Main notebook (EDA, modeling, evaluation)
├── financial_health_best_model.pkl   # Saved model + label encoder
├── app.py                            # Streamlit deployment script
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/financess-ml-project.git
cd financess-ml-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Explore the notebook
Open `Financess ML-Project.ipynb` in Jupyter or VS Code.

---

## 📊 Key Insights

- **Income, household size, and productive assets** (`prodsum1`) significantly influence financial trajectory.
- **Access to formal financial services** (savings, loans, mobile money) is strongly associated with improved financial status.
- **Barriers to banking** (affordability, awareness, trust) are prevalent and correlate with worsening financial outcomes.
- **Education level and age** show clear patterns: higher education and middle-age groups tend to report improvement.
- The dataset is **highly imbalanced**, which limits model performance on the minority classes (`Improved`, `Stayed the same`).

---

## 🔮 Future Improvements

- Address class imbalance using **SMOTE**, **class weights**, or **resampling**.
- Hyperparameter tuning with `GridSearchCV` / `Optuna`.
- Try advanced models: **XGBoost**, **LightGBM**, **CatBoost**.
- Feature engineering: interaction terms, target encoding for high-cardinality features (`county`).
- Add **SHAP** explainability for model interpretability.
- Expand the Streamlit app with batch prediction and data upload.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.

---

## 👤 Author

**Stephen Omengo**

## 🙏 Acknowledgments

- **FinAccess 2024** for providing the dataset.
- The open-source Python data science community.
