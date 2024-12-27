# AlphaCare Insurance Analytics

## Project Overview
AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa. This project aims to analyze historical insurance claim data to optimize marketing strategies and identify low-risk targets for reduced premiums, enabling client acquisition.

## Business Objectives
The primary goals of this project are:
- Optimize marketing strategies through data-driven insights.
- Identify "low-risk" client segments for premium reductions.
- Develop predictive models to tailor insurance products to meet customer needs effectively.

## Motivation
This project sharpens skills in Data Engineering (DE), Predictive Analytics (PA), and Machine Learning Engineering (MLE). It provides hands-on experience with:
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Machine learning model development and evaluation

These tasks simulate challenges typical in financial analytics and enhance problem-solving, data management, and creativity in tackling business challenges.

---

## Deliverables
### Task 1: Git and GitHub Setup
- **Tasks**:
  - Create and initialize a Git repository with a detailed README.
  - Setup Git version control with branches and commits.
  - Implement CI/CD pipelines using GitHub Actions.
- **KPIs**:
  - Proficiency in version control and CI/CD.

### Task 2: Exploratory Data Analysis (EDA)
- **Tasks**:
  - **Data Summarization**:
    - Compute descriptive statistics for numerical features (e.g., `TotalPremium`, `TotalClaim`).
    - Review data types to ensure proper formatting of categorical variables and dates.
  - **Data Quality Assessment**:
    - Identify and handle missing values.
  - **Univariate Analysis**:
    - Plot histograms and bar charts to explore data distributions.
  - **Bivariate/Multivariate Analysis**:
    - Analyze correlations between features like `TotalPremium` and `TotalClaims`.
    - Visualize geographic trends in insurance metrics.
  - **Outlier Detection**:
    - Use box plots to detect outliers in numerical data.
  - **Visualization**:
    - Produce three insightful plots capturing key findings from EDA.

### Task 3: A/B Hypothesis Testing
- **Hypotheses**:
  1. There are no risk differences across provinces.
  2. There are no risk differences between zip codes.
  3. There are no significant margin differences between zip codes.
  4. There are no significant risk differences between women and men.
- **Tasks**:
  - Select key performance indicators (KPIs).
  - Perform data segmentation into control and test groups.
  - Conduct statistical tests (e.g., chi-squared, t-tests, z-tests).
  - Analyze p-values and interpret results.

### Task 4: Statistical Modeling
- **Tasks**:
  - **Data Preparation**:
    - Handle missing data.
    - Engineer relevant features.
    - Encode categorical data numerically.
    - Split data into training and testing sets.
  - **Model Building**:
    - Implement linear regression, random forests, and XGBoost models.
  - **Model Evaluation**:
    - Assess models using metrics like accuracy, precision, recall, and F1-score.
  - **Model Interpretation**:
    - Analyze feature importance using SHAP or LIME.
    - Compare model performance and provide recommendations.

---

## Technologies Used
- **Version Control**: Git, GitHub
- **Data Analysis**: Python, Pandas, Matplotlib, Seaborn
- **Statistical Testing**: SciPy, StatsModels
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, SHAP
- **CI/CD**: GitHub Actions

---

## Getting Started
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/1Light/kaim-week-3.git
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup
1. Add your data files to the `data/` directory.
2. Initialize DVC:
   ```bash
   dvc init
   ```
3. Track your data with DVC:
   ```bash
   dvc add data/data.csv
   ```
4. Commit DVC files to Git:
   ```bash
   git add data/data.csv.dvc .gitignore
   git commit -m "Add data file and DVC tracking"
   ```

---

## Contribution Guidelines
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push your branch and submit a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
Thanks to AlphaCare Insurance Solutions for providing the opportunity to work on this exciting project. Special thanks to all contributors and team members for their support and dedication.