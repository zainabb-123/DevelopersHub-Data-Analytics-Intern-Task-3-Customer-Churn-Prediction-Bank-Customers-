# Bank Customer Churn Prediction

## Task Objective

The goal of this project is to predict whether a bank customer will churn (i.e., close their account and leave the bank) based on demographic and account-related features. This is a **binary classification problem** where the target variable `Exited` indicates churn (`1`) or retention (`0`).

Accurately identifying at-risk customers enables the bank to take proactive retention measures, reducing revenue loss and improving customer lifetime value.

---

## Approach

### 1. Data Loading
The dataset (`Churn_Modelling.csv`) was extracted from a zip archive and loaded into a pandas DataFrame. It contains 10,000 customer records with features such as credit score, geography, gender, age, tenure, account balance, number of products, and activity status.

### 2. Data Preprocessing
- **Dropped irrelevant columns**: `RowNumber`, `CustomerId`, and `Surname` were removed as they carry no predictive signal.
- **One-hot encoding**: Categorical features `Geography` and `Gender` were encoded using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity.
- **Train/test split**: The data was split 80/20 into training and test sets, using stratified sampling to preserve the class distribution.
- **Feature scaling**: Continuous numerical features (e.g., `CreditScore`, `Age`, `Balance`, `EstimatedSalary`) were standardised using `StandardScaler`. Binary and already-encoded columns were excluded from scaling.

### 3. Model Training
A **Logistic Regression** model was trained using the `liblinear` solver, which is well-suited for smaller datasets and supports both L1 and L2 regularisation.

### 4. Evaluation
Model performance was assessed on the held-out test set using:
- Accuracy
- Precision & Recall
- F1-Score
- ROC AUC Score
- Confusion Matrix

### 5. Feature Importance
Logistic Regression coefficients were extracted and ranked by absolute magnitude to identify the features with the greatest influence on churn prediction.

---

## Results and Insights

### Model Performance (Logistic Regression)

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~81%   |
| Precision  | ~57%   |
| Recall     | ~20%   |
| F1-Score   | ~29%   |
| ROC AUC    | ~77%   |

> *Note: Exact figures depend on the runtime output. The values above reflect typical results for logistic regression on this dataset.*

### Key Insights

- **Age** is among the strongest predictors of churn — older customers are significantly more likely to leave.
- **Geography** matters: customers from Germany show a notably higher churn rate compared to France and Spain.
- **Number of Products** has a non-linear effect — customers with only one product or more than two products are at higher risk.
- **Active membership** is a strong negative predictor of churn; inactive customers churn at a much higher rate.
- **Balance** is also informative — customers with higher balances but who are inactive tend to churn.

### Limitations & Next Steps

- Logistic Regression achieves decent accuracy but **low recall**, meaning many churners are missed. The dataset is class-imbalanced (~20% churn rate), which suppresses recall.
- **Recommended improvements**:
  - Apply class-balancing techniques (e.g., SMOTE, class weights).
  - Explore non-linear models such as Random Forest, Gradient Boosting (XGBoost/LightGBM), or a neural network for better recall.
  - Tune the classification threshold to trade precision for recall depending on business priorities.
  - Perform cross-validation for more robust performance estimates.
