# E-Commerce Customer Churn Prediction  

## Business Problem Understanding  
Customer churn analysis involves evaluating a company's customer loss rate and creating strategies to reduce it. Churn can significantly impact a business, as an increased churn rate creates challenges, making it difficult to replace lost customers and often leading to a self-perpetuating cycle.  

In the context of e-commerce, customer churn occurs when customers stop buying from the store and move to competitors. A low churn rate indicates successful targeting of the right audience, while a high churn rate suggests factors that drive customers away. While achieving a 0% churn rate is unrealistic, the aim is to maintain it lower than the business growth rate.  

Churn analysis is crucial for tracking customer behavior and understanding why customers leave, enabling companies to:  
- Improve customer retention.  
- Develop personalized marketing strategies.  
- Reduce costs by focusing on loyal customers.  
- Enhance customer experience.  
- Understand competitors' influence.  

This case study involves building a machine learning model to predict customer churn, benefiting the business by enabling targeted strategies and actionable insights.  

---

## Data Source and Description  
The dataset originates from an online retail company and consists of 3941 rows (customers) and 11 columns (features):  
- **Tenure**: Days of customer tenure with the company.  
- **WarehouseToHome**: Distance from the customer's house to the warehouse.  
- **NumberOfDeviceRegistered**: Number of registered devices (1-6).  
- **PreferedOrderCat**: Preferred order category (6 categories).  
- **SatisfactionScore**: Customer satisfaction score (1-5).  
- **MaritalStatus**: Marital status (3 categories).  
- **NumberOfAddress**: Number of registered addresses.  
- **Complaint**: Whether a complaint was raised in the last month (1: Yes, 0: No).  
- **DaySinceLastOrder**: Days since the last order.  
- **CashbackAmount**: Average cashback amount in the last month.  
- **Churn**: Target variable indicating churn (1: Churn, 0: Not Churn).  

---

## Project Goal  
Develop a machine learning model to predict customer churn in an e-commerce business.  

---

## Analytic Approach  

### Data Preparation and Cleaning  
- Removed outliers from `Tenure`, `NumberOfAddress`, `WarehouseToHome`, and `DaySinceLastOrder`.  
- Combined categories `Mobile` and `Mobile Phone` in the `PreferedOrderCat` column.  

### Data Splitting  
- **Features (X)**: Independent variables.  
- **Target (y)**: `Churn` (dependent variable).  
- Data split: 80% for training, 20% for testing (random_state=0).  

### Preprocessing Steps  
- **Missing Values**: Handled using `IterativeImputer` for numerical columns.  
- **Encoding**: One-hot encoding for categorical variables.  
- **Imbalanced Data**: Handled using SMOTE (Synthetic Minority Oversampling Technique).  
- **Scaling**: Applied `RobustScaler`.  

### Modeling Approach  
1. Evaluated several models with cross-validation:  
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  
   - Random Forest  
   - AdaBoost  
   - Gradient Boosting  
   - Extreme Gradient Boosting (XGBoost)  

2. Selected the top 2 models based on performance metrics for hyperparameter tuning.  

3. Tested and compared model performance on test data before and after hyperparameter tuning.  

---

## Metrics for Evaluation  
- **Precision**  
- **Recall**  
- **F1-score**  
- **ROC-AUC Curve**  
- **Precision-Recall Curve**  

---

## Feature Importance  
The analysis identifies features contributing significantly to churn, helping the company focus its efforts on engaging customers effectively.  

---

## Tools and Technologies Used  

### Programming Language  
- Python  

### Standard Libraries  
- `numpy`, `pandas`: For data manipulation and analysis.  
- `matplotlib`, `seaborn`: For data visualization.  

### Machine Learning and Preprocessing  
- `scikit-learn`:  
  - `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`: Model training and validation.  
  - `ColumnTransformer`, `Pipeline`: Data preprocessing.  
  - `OneHotEncoder`, `RobustScaler`: Encoding and scaling.  
  - `IterativeImputer`: Handling missing values.  
  - Evaluation metrics: `accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `classification_report`, `roc_auc_score`, `PrecisionRecallDisplay`, `make_scorer`.  
- `imbalanced-learn`: Handling imbalanced data using `SMOTE` and other techniques.  
- `xgboost`: For building the Extreme Gradient Boosting classifier.  

### Classifiers Used  
- `LogisticRegression`  
- `KNeighborsClassifier`  
- `DecisionTreeClassifier` (with `plot_tree` for visualization)  
- `RandomForestClassifier`  
- `AdaBoostClassifier`  
- `GradientBoostingClassifier`  
- `XGBClassifier`  

### Utilities  
- `pickle`: Saving and loading the trained model.  
- `os`: System-level file handling.  
- `warnings`: For managing warning messages.  

---

## Repository Structure  
- **`churn_analysis.ipynb`**: Jupyter Notebook containing data preprocessing, analysis, and modeling code.  
- **`Model_RF.sav`**: Saved Random Forest model.  
- **`README.md`**: Project documentation.  

---
