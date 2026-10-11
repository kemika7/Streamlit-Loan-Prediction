# Comprehensive Technical Documentation: Loan Default Prediction System

## 1. Executive Summary & Architecture Overview

The **Loan Default Prediction System** is an end-to-end Machine Learning web application designed to assess, predict, and explain credit default risks for loan applicants. Built using **Python**, **Scikit-Learn**, **XGBoost**, **Imbalanced-Learn (SMOTE)**, **SHAP**, **Plotly**, and **Streamlit**, the system provides both single-applicant scoring with real-time model explainability and bulk batch processing with risk analytics.

### System Architecture Flowchart

```
                          ┌──────────────────────────┐
                          │     data/loans.csv       │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   pipeline.py Preprocessing │
                         │ (Clean, Feature Engineering)│
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       ┌─────────────────────────┐             ┌─────────────────────────┐
       │   Training Split (80%)  │             │     Test Split (20%)    │
       │  (Impute, Encode, SMOTE)│             │     (Held-out, Pure)    │
       └────────────┬────────────┘             └────────────┬────────────┘
                    │                                       │
                    ▼                                       │
       ┌─────────────────────────┐                          │
       │   Model Training & CV   │                          │
       │ (LogReg, RF, XGBoost)   │                          │
       └────────────┬────────────┘                          │
                    │                                       │
                    ▼                                       ▼
       ┌─────────────────────────┐             ┌─────────────────────────┐
       │     Best Model Pick     │────────────▶│  Test Evaluation & SHAP │
       │     (ROC-AUC Metric)    │             │   (Leak-Free Scoring)   │
       └────────────┬────────────┘             └────────────┬────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       Saved Artifacts       │
                         │    • models/model.joblib    │
                         │    • models/results.joblib  │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │    app.py (Streamlit UI)    │
                         │  • Interactive Dashboard    │
                         │  • Real-time SHAP Explainer │
                         │  • Batch Prediction Engine  │
                         └─────────────────────────────┘
```

### Key Technical Pillars
1. **Zero Data Leakage**: The data preparation pipeline guarantees that no information from the test set leaks into training: imputer medians, one-hot encodings, and SMOTE balancing are fitted strictly on training subsets.
2. **Inference Parity**: A single preprocessing engine (`pipeline.py`) is shared identically between training and real-time inference, eliminating training-serving skew.
3. **Transparent Explainability**: Every individual prediction is deconstructed via Shapley Additive Explanations (SHAP), providing loan officers with the exact financial drivers behind every approval or decline.
4. **Performance & Scalability**: Model training and evaluation run once (`train.py`) and serialize artifacts (`.joblib`). The Streamlit dashboard loads precomputed models and evaluation caches, executing predictions and interactive visualizations with zero lag.

---

## 2. Dataset Specification (`data/loans.csv`)

The dataset comprises **300 loan records** with **10 variables** (1 target variable, 7 predictor features, and 2 identifiers).

### Data Dictionary

| Column Name | Data Type | Role | Description | Observed Range / Unique Values |
|---|---|---|---|---|
| `Loan_ID` | String | Identifier | Unique identifier for each loan record | `L2000` – `L2299` |
| `Customer` | String | Identifier | Unique customer code | `Cust_000` – `Cust_299` |
| `Loan_Amount` | Numeric (Integer) | Predictor (Raw) | Requested loan principal in currency units | 100,000 to 5,000,000 (Median: 500,000) |
| `Tenure_Months` | Numeric (Integer) | Predictor (Raw) | Repayment duration in months | 12, 24, 36, 60, 120 (Median: 36) |
| `Interest_Rate` | Numeric (Float) | Predictor (Raw) | Annual percentage interest rate | 5.50% to 22.40% (Median: 12.44%) |
| `Annual_Income` | Numeric (Integer) | Predictor (Raw) | Applicant's verified annual gross income | 150,000 to 5,500,000 (Median: 550,000) |
| `Credit_Score` | Numeric (Integer) | Predictor (Raw) | Credit bureau rating | 300 to 849 (Median: 602.5) |
| `Has_Collateral` | Categorical (String) | Predictor (Raw) | Whether tangible collateral was pledged | `Yes`, `No` |
| `Sector` | Categorical (String) | Predictor (Raw) | Economic sector of employment / business | `Agriculture`, `Manufacturing`, `Personal`, `Service`, `Trade` |
| `Defaulted` | Binary Integer | Target | Binary ground-truth outcome | `0` = Repaid (58.7%), `1` = Defaulted (41.3%) |

---

## 3. Data Preprocessing & Pipeline Architecture (`pipeline.py`)

`pipeline.py` is the central operational module that guarantees strict mathematical and algorithmic parity between offline training and online inference.

### 3.1 Constants & Column Groups
```python
TARGET_COL = "Defaulted"
ID_COLS = ["Loan_ID", "Customer"]
NUM_INPUT_COLS = ["Loan_Amount", "Tenure_Months", "Interest_Rate", "Annual_Income", "Credit_Score"]
CAT_COLS = ["Has_Collateral", "Sector"]
RAW_FEATURE_COLS = NUM_INPUT_COLS + CAT_COLS
ENGINEERED_COLS = ["loan_to_income", "payment_to_income"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

### 3.2 Functions Walkthrough & Logic

#### 1. `clean(df: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Cleans training data by eliminating duplicate records and dropping anomalies or non-physical numbers.
- **Rules applied**:
  - Drops duplicate rows via `df.drop_duplicates()`.
  - Enforces positive financial values: `Annual_Income > 0`, `Loan_Amount > 0`, `Tenure_Months > 0`.
  - Constrains `Credit_Score` within standard FICO/CIBIL boundaries: `[300, 900]`.
- **Note**: This function is applied to historical training data. At inference time, applicant rows are **never silently dropped**; instead, missing/dirty inputs are imputed and validated.

#### 2. `monthly_payment(principal, annual_rate_pct, months)`
- **Purpose**: Computes standard amortized monthly debt obligation based on the annuity loan payment formula.
- **Formula**:
  $$M = P \times \frac{r(1 + r)^n}{(1 + r)^n - 1}$$
  Where:
  - $P$ = Principal (`Loan_Amount`)
  - $r$ = Monthly interest rate $= \frac{\text{Annual\_Rate}}{1200}$
  - $n$ = Number of monthly installments (`Tenure_Months`)
- **Zero-Interest Fallback**: If $r = 0$, payment defaults to linear division: $M = \frac{P}{n}$.

#### 3. `engineer_features(df: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Constructs domain-specific credit risk metrics proven to possess higher predictive power than raw figures alone.
- **Derived Features**:
  1. `loan_to_income`:
     $$\text{loan\_to\_income} = \frac{\text{Loan\_Amount}}{\text{Annual\_Income}}$$
     Measures overall leverage and debt burden relative to applicant earnings.
  2. `payment_to_income`:
     $$\text{payment\_to\_income} = \frac{12 \times \text{monthly\_payment}}{\text{Annual\_Income}}$$
     Represents the **Debt Service Coverage Ratio (DSCR) inverse**, reflecting the exact fraction of yearly income absorbed by repayment installments.

#### 4. `fit_imputer(X_train: pd.DataFrame) -> dict`
- **Purpose**: Calculates central tendency statistics strictly from the **training split** to prevent data leakage.
- **Logic**:
  - Continuous columns (`NUM_INPUT_COLS`): Median values.
  - Categorical columns (`CAT_COLS`): Mode (most frequent category).

#### 5. `impute(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame`
- **Purpose**: Fills missing or corrupted fields using the learned `fill_values`. Coerces numerical strings to numeric, converting unparseable characters to `NaN` before filling with the median.

#### 6. `encode_and_merge(df: pd.DataFrame, encoder, cat_cols=CAT_COLS) -> pd.DataFrame`
- **Purpose**: One-hot encodes categorical variables (`Has_Collateral` and `Sector`) using a fitted `OneHotEncoder(sparse_output=False, handle_unknown="ignore")` and merges them with the numeric columns.
- **Output Columns (14 features)**:
  `['Loan_Amount', 'Tenure_Months', 'Interest_Rate', 'Annual_Income', 'Credit_Score', 'loan_to_income', 'payment_to_income', 'Has_Collateral_No', 'Has_Collateral_Yes', 'Sector_Agriculture', 'Sector_Manufacturing', 'Sector_Personal', 'Sector_Service', 'Sector_Trade']`

#### 7. `prepare_features(raw: pd.DataFrame, artifact: dict) -> pd.DataFrame`
- **Purpose**: The master inference pipeline. Takes any raw input (a single dictionary from the web UI or an uploaded CSV DataFrame), normalizes string cases, applies learned imputer values, engineers features, executes one-hot encoding, and reindexes to the exact feature ordering expected by the models.

#### 8. `make_explainer(model, background: pd.DataFrame)`
- **Purpose**: Instantiates the optimal SHAP explainer depending on model family:
  - For `Pipeline([StandardScaler, LogisticRegression])`: Uses `shap.LinearExplainer` evaluated on the scaled background dataset.
  - For `RandomForestClassifier` and `XGBClassifier`: Uses `shap.TreeExplainer` for exact and fast tree-path attribution.

#### 9. `shap_explain(explainer, model, X: pd.DataFrame)`
- **Purpose**: Computes SHAP values specifically for Class 1 (Default). Standardizes output shapes whether the explainer returns multi-class arrays or single-output margin arrays.

#### 10. `score_batch(batch: pd.DataFrame, artifact: dict, model_name: str)`
- **Purpose**: Validates schema for bulk CSV uploads, catches missing required columns, logs warnings for out-of-vocabulary categories or imputed rows, scores records, and appends `Default_Probability`, `Prediction` (0/1), and `Prediction_Label` ("Repaid"/"Defaulted").

---

## 4. Model Training, Leak-Free Validation & Benchmarking (`train.py`)

`train.py` executes the machine learning training lifecycle with rigorous statistical integrity.

### 4.1 Methodology & Data Leakage Prevention

```
Step 1: Raw Data (300 rows)
   │
   ▼ Cleaned (300 valid rows)
   │
   ├──▶ Stratified Split (80% Train / 20% Test, stratify=y, seed=42)
   │       ├── Train: 240 rows (141 Repaid, 99 Defaulted)
   │       └── Test:   60 rows ( 35 Repaid, 25 Defaulted) [HELD OUT COMPLETELY]
   │
Step 2: Fit Imputer & OneHotEncoder ON TRAIN SPLIT ONLY
   │
Step 3: Apply SMOTE ON TRAIN SPLIT ONLY
   │       └── Balanced Train: 282 rows (141 Repaid, 141 Defaulted)
   │
Step 4: Model Training on Balanced Train
   │       ├── Logistic Regression (with StandardScaler Pipeline)
   │       ├── Random Forest Classifier
   │       └── XGBoost Classifier
   │
Step 5: 5-Fold Cross-Validation (Inside Train Split Only)
   │       └── Preprocessing, Encoding & SMOTE refit INSIDE every fold
   │
Step 6: Evaluation on Untouched Test Set (60 rows)
   │
Step 7: Artifact Serialization (model.joblib, results.joblib)
```

### 4.2 Machine Learning Algorithms

1. **Logistic Regression (with `StandardScaler`)**:
   - **Why scaled?**: `Annual_Income` and `Loan_Amount` reach millions, while `payment_to_income` is a small decimal. Without standard scaling, gradient-based solvers fail to converge even with 1,000 iterations.
   - Wrapped inside `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])`.
2. **Random Forest Classifier**:
   - `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`.
   - Ensemble of 300 de-correlated decision trees with bagging.
3. **XGBoost (Extreme Gradient Boosting)**:
   - `XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)`.
   - Sequentially builds gradient boosted trees optimizing cross-entropy loss.

### 4.3 Model Comparison & Test Set Results

Evaluated on the 60-loan held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg. Precision (PR-AUC) |
|---|---|---|---|---|---|---|
| **Logistic Regression** (Production Best) | **0.750** | **0.708** | **0.680** | **0.694** | **0.814** | **0.743** |
| **Random Forest** | 0.683 | 0.688 | 0.440 | 0.537 | 0.793 | 0.750 |
| **XGBoost** | 0.667 | 0.632 | 0.480 | 0.545 | 0.725 | 0.680 |

> **Selection**: **Logistic Regression** was chosen as the primary production model because it achieved the highest test ROC-AUC (0.814) and balanced F1-score (0.694), while providing stable, linear credit calibration.

### 4.4 5-Fold Cross-Validation Scores (Training Split)

To ensure the test set performance is not an artifact of a single split, leak-free 5-fold cross-validation was executed across the 240 training instances:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.733 ± 0.052 | 0.681 ± 0.063 | 0.667 ± 0.088 | 0.672 ± 0.066 | 0.804 ± 0.048 | 0.725 ± 0.057 |
| **Random Forest** | 0.708 ± 0.041 | 0.669 ± 0.058 | 0.566 ± 0.073 | 0.611 ± 0.051 | 0.781 ± 0.039 | 0.738 ± 0.045 |
| **XGBoost** | 0.688 ± 0.046 | 0.628 ± 0.059 | 0.586 ± 0.081 | 0.604 ± 0.059 | 0.749 ± 0.053 | 0.691 ± 0.052 |

---

## 5. Streamlit Application Deep Dive (`app.py`)

The application is structured as a multi-page interactive cockpit with six dedicated navigation views.

### 5.1 Design & Visual Palette System
To ensure enterprise-grade presentation, `app.py` implements a bespoke color mapping:
- **Blue (`#2a78d6`)**: "Repaid" class, protective risk factors, Logistic Regression visual marker.
- **Orange (`#eb6834`)**: "Defaulted" class, high-risk flags, Random Forest visual marker.
- **Aqua (`#1baf7a`)**: XGBoost visual marker, secondary positive confirmation.
- **Red (`#e34948`)**: High-risk SHAP push factors, alerts.
- **Neutral / Slate (`#f0efec`)**: Neutral midpoint for diverging correlation scales.

```python
def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=80 if legend else 50, b=10), showlegend=legend,
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        bargap=0.25, hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig
```

---

### 5.2 Page Walkthrough & Visualization Explanations

#### Page 1: Dataset Dashboard
Designed for data profiling and auditing raw portfolio metrics before ingestion.
- **Top KPI Cards**:
  - Total Rows: `300`
  - Total Columns: `10`
  - Missing Cells: `0`
  - Duplicate Rows: `0`
  - Base Default Rate: `41.3%`
- **Data Preview Tab**: Filterable table with multi-select dropdowns for `Sector` and `Has_Collateral`.
- **Types & Missing Values Tab**: Reports data types, non-null counts, missing percentages, and uniqueness cardinality.
- **Summary Statistics Tab**: Full statistical breakdown (`mean`, `std`, `min`, `25%`, `median`, `75%`, `max`).
- **Visualizations Explained**:
  1. *Histogram by Outcome*: Shows distribution of any selected continuous feature split by `Repaid` (Blue) vs `Defaulted` (Orange).
  2. *Default Rate by Categorical Feature*: Bar chart depicting the exact default percentage per sector or collateral class.
  3. *Bivariate Scatter Plot*: 2D mapping of any two numeric variables (e.g., `Annual_Income` vs `Credit_Score`) colored by loan outcome with hover tooltips displaying Customer ID, Sector, and Collateral.

---

#### Page 2: EDA & Feature Engineering
Focuses on class distribution, synthetic balancing, and feature engineering rationale.
- **Visualizations Explained**:
  1. *Target Distribution & SMOTE Comparison*:
     - **Before**: 141 Repaid vs 99 Defaulted (training split).
     - **After SMOTE**: 141 Repaid vs 141 Defaulted (synthetically interpolated minority cases).
     - **Test Set**: 35 Repaid vs 25 Defaulted (untouched, verifying no synthetic leakage).
  2. *Quantile Binning & Box Plots*:
     - Divides continuous variables into 5 equal quantiles and computes default rate per bin, illustrating non-linear default risk curves (e.g., credit score bins showing steep risk drop-offs above 650).
  3. *Categorical Grouped Counts*:
     - Side-by-side count comparisons for `Sector` and `Has_Collateral`.
  4. *Engineered Ratios on Log Scale*:
     - Box plots for `loan_to_income` and `payment_to_income` plotted on logarithmic y-axes to clearly separate distributions between defaulters and non-defaulters despite positive skew.
  5. *Correlation Heatmap*:
     - Pearson correlation matrix displayed with diverging color scale (`Blue` = -1.0, `Neutral` = 0.0, `Red` = +1.0) with embedded numerical annotations (`text_auto=".2f"`).

---

#### Page 3: Model Evaluation
Provides exhaustive diagnostics comparing all candidate algorithms on held-out test data.
- **Visualizations Explained**:
  1. *Model Comparison Grouped Bar Chart*: Compares Accuracy, Precision, Recall, F1, and ROC-AUC across Logistic Regression, Random Forest, and XGBoost.
  2. *5-Fold Cross-Validation Aggregation Table*: Presents `mean ± standard deviation` for all metrics across folds.
  3. *Interactive Confusion Matrix with Threshold Slider*:
     - A dynamic slider allows adjusting the decision boundary $\tau$ from `0.05` to `0.95`.
     - Dynamically updates True Positives, False Positives, True Negatives, and False Negatives, along with real-time Precision, Recall, Accuracy, and F1 recalculations.
  4. *ROC Curves*:
     - Plots True Positive Rate (Sensitivity) vs False Positive Rate ($1 - \text{Specificity}$) for all three models with the diagonal dashed line indicating random chance ($AUC = 0.5$).
  5. *Precision-Recall Curves*:
     - Plots Precision vs Recall across thresholds, highlighting Average Precision (AP) which is resilient to class imbalance.
  6. *Classification Report*: Precision, recall, and f1-score broken down by class.

---

#### Page 4: SHAP Explainability
Delivers global and local interpretability based on cooperative game theory (Shapley values).
- **Visualizations Explained**:
  1. *Global Feature Importance Horizontal Bar*:
     - Ranks features by mean absolute SHAP value: $\frac{1}{N} \sum_{i=1}^N |\phi_j^{(i)}|$.
     - Reveals that `Credit_Score`, `payment_to_income`, and `loan_to_income` are the top 3 global drivers of loan outcome.
  2. *SHAP Beeswarm Summary Plot*:
     - Every dot represents an individual applicant from the test set.
     - Horizontal position ($x$-axis) represents the SHAP value: points to the right increase default probability; points to the left reduce default probability.
     - Color scale represents the feature value: Deep Blue indicates high values, Light/Sky Blue indicates low values.
     - *Interpretation*: Low Credit Score dots cluster far to the right (raises default risk), while high Credit Score dots cluster to the left (protective factor).
  3. *Automated Driver Summary*: Generates clear natural-language explanations of directional effects (e.g., *"Credit_Score: higher values → lower risk (mean |SHAP| 0.941)"*).
  4. *Dependence Scatter Plot*:
     - Plots feature value on $x$ vs its SHAP value on $y$.
     - Allows coloring points by actual loan outcome or another interacting feature to reveal non-linear thresholds and feature cross-effects.
  5. *Individual Applicant Drill-Down (Local Attribution)*:
     - Selects any individual test loan by `Loan_ID` and renders horizontal attribution bars.
     - Red bars push towards Default; Blue bars pull towards Repaid.

---

#### Page 5: Loan Prediction (Single Applicant Inference)
Interactive calculator for credit officers to evaluate prospective borrowers.
- **Input Controls**:
  - `Loan Amount` (Number input, step: 50,000)
  - `Tenure (Months)` (Selectbox: 12, 24, 36, 60, 120)
  - `Interest Rate (%)` (Number input, step: 0.25)
  - `Annual Income` (Number input, step: 60,000)
  - `Credit Score` (Slider: 300 to 900)
  - `Has Collateral` (Radio: Yes / No)
  - `Sector` (Selectbox: Agriculture, Manufacturing, Personal, Service, Trade)
  - `Model` (Dropdown: Logistic Regression, Random Forest, XGBoost)
- **Output & Dynamic Explanations**:
  - **Decision Banner**: `Likely to REPAY` (Green banner) or `Likely to DEFAULT` (Red alert banner).
  - **Probability Gauge / Progress Bar**: Displays exact default probability vs the 50% decision threshold.
  - **Engineered Financial Indicators**: Displays instant calculated `loan_to_income` and `payment_to_income` percentages.
  - **Local SHAP Waterfall / Contribution Plot**: Dynamically renders the exact reasons behind this specific applicant's decision, highlighting whether low income, high debt service, or credit score tipped the scales.
  - **Extrapolation Warning**: If any applicant value falls outside observed historical distributions, an amber warning flags that the prediction is an extrapolation.

---

#### Page 6: Batch Prediction (Bulk CSV Processing)
Processes batch portfolios with schema validation and export capabilities.
- **Workflow**:
  1. *Download Template*: Button to download a pre-formatted `loan_template.csv`.
  2. *File Uploader*: Drag-and-drop CSV parser.
  3. *Automatic Validation*:
     - Missing mandatory columns throw an explicit error preventing corrupted scoring.
     - Missing values trigger automated imputer substitution and output a descriptive notice.
     - Unseen sectors or collateral types trigger one-hot zero-vector fallback with an advisory warning.
  4. *Portfolio Analytics*:
     - Total Applications processed.
     - Predicted Defaulter Count and Default Rate.
     - Mean Portfolio Default Probability.
     - Ground-Truth Accuracy & F1 evaluation (if the uploaded CSV contains an optional `Defaulted` column).
  5. *Distribution Visualizations*:
     - Histogram of predicted default probabilities with decision threshold line.
     - Predicted default rate segmented by business sector.
  6. *Export*: Download button generating `loan_predictions.csv` containing original applicant data alongside predicted probabilities and classification decisions.

---

## 6. Test Suite & Verification Architecture (`tests/test_app.py`)

The project contains **13 automated unit and integration tests** executing via `pytest` and `streamlit.testing.v1.AppTest`.

```
collected 13 items
tests/test_app.py::test_inference_matches_training_probabilities PASSED
tests/test_app.py::test_predictions_respond_to_inputs            PASSED
tests/test_app.py::test_messy_batch_input_is_handled            PASSED
tests/test_app.py::test_smote_only_on_training_split            PASSED
tests/test_app.py::test_page_renders[Dataset Dashboard]         PASSED
tests/test_app.py::test_page_renders[EDA & Feature Engineering] PASSED
tests/test_app.py::test_page_renders[Model Evaluation]          PASSED
tests/test_app.py::test_page_renders[SHAP Explainability]       PASSED
tests/test_app.py::test_page_renders[Loan Prediction]           PASSED
tests/test_app.py::test_page_renders[Batch Prediction]          PASSED
tests/test_app.py::test_single_prediction_form                  PASSED
tests/test_app.py::test_model_choice_changes_prediction         PASSED
tests/test_app.py::test_batch_scoring_end_to_end                PASSED

============================== 13 passed in 38.99s ==============================
```

### Detailed Test Specifications

1. `test_inference_matches_training_probabilities`:
   - Validates that passing raw test rows through `pipeline.prepare_features` and scoring with loaded models reproduces training-time probabilities within numerical tolerance (`atol=1e-9`).
2. `test_predictions_respond_to_inputs`:
   - Verifies the models are sensitive to financial signals (e.g., an applicant with credit score 840 and income 2.4M receives a strictly lower default probability than an applicant with score 310 and income 180K).
   - Confirms predictions are genuinely non-constant across 50+ unique continuous probabilities.
3. `test_messy_batch_input_is_handled`:
   - Supplies dirty input: whitespace `" trade "`, unparseable string `"abc"`, `None` collateral, and unseen sector `"Mining"`.
   - Confirms imputations fill nulls, whitespace is stripped, and unseen categories encode to all zeros without throwing unhandled exceptions.
4. `test_smote_only_on_training_split`:
   - Mathematically verifies that $n_{\text{train}} + n_{\text{test}} = n_{\text{clean}}$, the test set was never resampled, and SMOTE balanced only the training split.
5. `test_page_renders` (6 parameterized tests):
   - Programmatically instantiates each of the 6 Streamlit pages using `AppTest` and verifies that no unhandled runtime exceptions occur.
6. `test_single_prediction_form`:
   - Simulates user input on the single prediction form, clicks the submit button, and asserts that prediction output text is rendered.
7. `test_model_choice_changes_prediction`:
   - Switches between Logistic Regression, Random Forest, and XGBoost on the same applicant and confirms that model predictions adapt dynamically.
8. `test_batch_scoring_end_to_end`:
   - Tests batch scoring on 20 rows, checks probability boundaries, validates schema error handling for missing columns, and verifies CSV download generation.

---

## 7. Installation, Setup & Troubleshooting Guide

### 7.1 System Requirements
- **Operating System**: macOS (Apple Silicon / Intel), Linux, or Windows.
- **Python**: Version 3.10, 3.11, or 3.12.
- **macOS Dependency**: OpenMP library (`libomp`) required for XGBoost on macOS.

### 7.2 Installation Steps

#### Step 1: Clone or Navigate to Directory
```bash
cd /Users/kemikasherchan/Downloads/Loan-Prediction-main
```

#### Step 2: Install macOS OpenMP Library (if on macOS)
```bash
brew install libomp
```

#### Step 3: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest
```

#### Step 5: (Optional) Re-train Models
The project already comes with pre-trained models in `models/`. If you modify `loans.csv` or feature engineering formulas, retrain with:
```bash
python train.py
```

#### Step 6: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Step 7: Run Automated Test Suite
```bash
pytest tests/
```

---

### 7.3 Common Troubleshooting Scenarios

#### Issue 1: `ModuleNotFoundError: No module named 'plotly'`
- **Cause**: Running `streamlit run app.py` with system Python rather than the virtual environment where requirements are installed.
- **Solution**:
  Either activate the environment:
  ```bash
  source .venv/bin/activate
  streamlit run app.py
  ```
  Or install dependencies into your global Python:
  ```bash
  pip install plotly imbalanced-learn xgboost shap
  ```

#### Issue 2: `xgboost._c_api.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded: Library not loaded: @rpath/libomp.dylib`
- **Cause**: macOS does not ship with the OpenMP runtime required by XGBoost C++ acceleration binaries.
- **Solution**:
  Install OpenMP via Homebrew:
  ```bash
  brew install libomp
  ```

#### Issue 3: `Model files not found. Run python train.py once`
- **Cause**: `models/model.joblib` or `models/results.joblib` is missing.
- **Solution**:
  Execute `python train.py` to regenerate the artifacts.
