# E-Commerce Customer Purchase Behaviour Analysis

## Project Overview
This project analyzes a comprehensive e-commerce dataset to identify the key factors contributing to customer churn. By leveraging statistical hypothesis testing and machine learning, the project aims to provide actionable insights for improving customer retention strategies and sales growth.

The analysis focuses on predicting whether a customer is likely to churn based on behaviour metrics such as cashback amount, tenure, days since last order, and complaint history.

## Key Results & Insights
* **Predictive Performance**: Achieved 95% accuracy in predicting customer churn using a Random Forest Classifier.
* **Churn Drivers**: Statistical analysis identified Tenure, Cashback Amount, Days Since Last Order, and Complaints as the most significant predictors of churn.
* **Customer Behaviour**:
    * Customers with fewer registered devices and shorter tenure are significantly more likely to churn.
    * Complaints showed a strong correlation with churn likelihood (confirmed via Mann-Whitney U Test).
    * Contrary to expectations, Coupon Usage was found to have no statistically significant association with churn behaviour.

## Technologies Used
* **Language**: Python
* **Data Manipulation**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Statistical Analysis**: SciPy (T-tests, Mann-Whitney U, Chi-Square)
* **Machine Learning**: Scikit-Learn (Random Forest, K-Neighbors, Gaussian Naive Bayes)

## Project Structure
The analysis is broken down into independent scripts that handle specific stages of the data science pipeline:

* **cleaning.py**: Handles missing values (imputation/dropping), outlier removal, and data formatting. Outputs `cleandata.csv`.
* **dataVisualizations.py**: Generates exploratory visualizations (bar plots, pair plots) to identify trends and correlations.
* **normTests.py**: Conducts normality testing to determine appropriate statistical tests.
* **statsTest.py**: Performs hypothesis testing (T-Test, Mann-Whitney U, Chi-Square) to validate relationships between variables and churn.
* **modelling.py**: Builds, trains, and evaluates ML models (KNN, Random Forest, GaussianNB).

## Methodology

### 1. Data Cleaning
* **Outlier Removal**: Removed extreme outliers in Tenure, WarehouseToHome, and NumberOfAddress.
* **Imputation**:
    * Mode imputation for categorical/discrete variables like HourSpendOnApp, CouponUsed, and OrderCount.
    * Mean imputation for continuous variables like WarehouseToHome.
    * Selective dropping of rows for Tenure and DaySinceLastOrder to preserve correlation integrity.

### 2. Statistical Analysis
Since many variables did not follow a normal distribution (p < 0.05), non-parametric tests were largely utilized:
* **Mann-Whitney U Test**: Confirmed significant differences in Tenure and DaySinceLastOrder between churned and non-churned customers.
* **Chi-Square Test**: Established strong relationships between MaritalStatus, PreferredOrderCat, and Churn.
* **T-Test**: Used on transformed CashbackAmount data to confirm its impact on retention.

### 3. Machine Learning Modeling
Three classification models were trained and evaluated on a 75/25 train-test split:
* **Random Forest Classifier**: 95% Accuracy (Best Performer) - Handled the mix of categorical and numerical data effectively.
* **K-Neighbors Classifier**: 84% Accuracy - Performed well due to distinct clustering in variables like Tenure.
* **Gaussian Naive Bayes**: 42% Accuracy - Underperformed due to the non-normal distribution of the underlying data.

## How to Run
The files are designed to be run in the following sequential order to reproduce the full analysis pipeline:

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn sklearn scipy
    ```
2.  **Run the Pipeline**:
    ```bash
    python cleaning.py           # Prepares the data
    python dataVisualizations.py # Generates plots
    python normTests.py          # Checks distribution
    python statsTest.py          # Runs statistical tests
    python modelling.py          # Trains and evaluates models
    ```

## Data Source
The dataset used for this analysis is sourced from Kaggle:
[E-Commerce Dataset](https://www.kaggle.com/datasets/anaghapaul/e-commerce-dataset?resource=download)
