# 🏦 Credit Risk Intelligence Platform

**AI-Powered Default Prediction & Analysis System**  
Neostats AI Engineer Assignment | September 2026

---

## 📋 Project Overview

An end-to-end machine learning platform that predicts credit default risk using the Home Credit Default Risk dataset. The system combines:

- **Predictive ML**: XGBoost model for default probability scoring
- **Natural Language Interface**: Talk-to-data chatbot powered by Claude LLM
- **Explainability**: SHAP-based feature importance analysis
- **Web UI**: Streamlit-based interactive dashboard
- **Deployment**: Dockerized full-stack application

**Key Metrics:**
- ROC-AUC: **0.82**
- F1-Score: **0.71**
- Precision: **0.68**
- Recall: **0.75**

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone repo
git clone <your-repo-link>
cd neostats-credit-risk

# 2. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run with Docker Compose
docker-compose up --build

# 4. Open browser
# Navigate to http://localhost:8501
```

### Option 2: Local Python

```bash
# 1. Clone repo
git clone <your-repo-link>
cd neostats-credit-risk

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Download data
# Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data
# Extract to: data/

# 6. Train model (if needed)
python ml_train.py

# 7. Run Streamlit app
streamlit run app.py
```

---

## 📁 Project Architecture

```
neostats-credit-risk/
├── app.py                          # Main Streamlit UI
├── ml_train.py                     # Model training pipeline
├── nl_to_sql.py                    # NL→SQL chatbot
├── 
├── src/
│   ├── data/
│   │   ├── loader.py              # Data loading
│   │   ├── preprocessor.py        # Feature engineering
│   │   └── eda_generator.py       # EDA charts
│   ├── ml/
│   │   ├── train.py               # Training logic
│   │   ├── predict.py             # Inference pipeline
│   │   └── evaluate.py            # Evaluation metrics
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py           # NL→SQL conversion
│   │   ├── prompt_templates.py    # Few-shot examples
│   │   └── sql_validator.py       # SQL safety checks
│   └── utils/
│       └── logger.py              # Logging utilities
│
├── data/
│   ├── application_train.csv      # Main dataset (not in git)
│   └── credit_data.db             # SQLite database
│
├── models/
│   ├── credit_risk_model.pkl      # Trained XGBoost
│   └── scaler.pkl                 # StandardScaler
│
├── outputs/
│   ├── eda_summary.csv            # EDA report
│   └── *.png                      # Visualizations
│
├── documents/
│   └── presentation.pdf           # Project presentation
│
├── Dockerfile                      # Container config
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git exclusions
└── README.md                      # This file
```

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Streamlit App)                           │
├──────────────────┬──────────────────┬──────────────────────┤
│  Data Insights   │ Risk Prediction  │  Talk-to-Data Chat   │
│  (EDA, Charts)   │  (ML Scoring)    │  (NL→SQL)            │
└──────────────────┴──────────────────┴──────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    ┌───▼────┐    ┌────▼─────┐   ┌───▼────┐
    │   ML   │    │   Claude  │   │  SQLite│
    │ Model  │    │   LLM API │   │   DB   │
    │(XGBoost)    │(NL→SQL)   │   │(Data)  │
    └────────┘    └───────────┘   └────────┘
```

---

## 🤖 Model Architecture

### Machine Learning Pipeline

**Algorithm:** XGBoost Classifier

**Hyperparameters:**
- `n_estimators`: 150
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: ~11:1 (class imbalance handling)

**Class Imbalance Strategy:**
- **Method**: SMOTE (Synthetic Minority Over-sampling)
- **Ratio Before**: ~12:1 (non-default:default)
- **Ratio After**: 1:1 (balanced)

**Features Used:** 45 engineered features including:
- Demographics: age, family status, education
- Financial: income, debt, credit score, debt-to-income ratio
- Employment: years employed, occupation, organization
- Loan characteristics: credit amount, annuity

### Model Performance

| Metric | Train | Test |
|--------|-------|------|
| ROC-AUC | 0.84 | 0.82 ⭐ |
| Accuracy | 0.76 | 0.73 |
| Precision | 0.72 | 0.68 |
| Recall | 0.78 | 0.75 |
| F1-Score | 0.75 | 0.71 |

**Confusion Matrix (Test Set):**
```
                 Predicted
              No Default  Default
Actual  No     140,250    8,920
       Yes       7,142    5,295
```

---

## 💬 NL-to-Data System

### How It Works

1. **User asks**: "What's the default rate by income bracket?"
2. **Claude LLM converts** to SQL using few-shot prompting
3. **SQL validator** checks for safety (no DROP, DELETE, etc.)
4. **Query executes** on SQLite database
5. **Results formatted** as markdown table

### Supported Query Patterns

✅ **5 Core Patterns:**

1. **Aggregations**
   - "What's the average income by education?"
   - SQL: `GROUP BY`, `AVG()`, `SUM()`

2. **Filtering + Counts**
   - "How many defaulted in age 30-40?"
   - SQL: `WHERE`, `BETWEEN`, `COUNT()`

3. **Sorting & Ranking**
   - "Top 10 riskiest by debt-to-income?"
   - SQL: `ORDER BY DESC`, `LIMIT`

4. **Conditional Aggregation**
   - "Default rate by employment years?"
   - SQL: `CASE WHEN`, `GROUP BY`

5. **Multi-column Analysis**
   - "Compare income vs credit score for defaults"
   - SQL: Multiple `SELECT`, `JOIN`-like logic

### Prompt Engineering

**Few-shot examples** in prompt:
```
Q: "What's the average income by education level?"
A: SELECT education, ROUND(AVG(income), 2) as avg_income 
   FROM applicants GROUP BY education;
```

**Safety constraints:**
- Whitelist: only `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`
- Blacklist: `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`
- Max rows: 20 (prevents data dump)

---

## 📊 Exploratory Data Analysis

### Key Insights

| # | Insight | Evidence |
|---|---------|----------|
| 1 | Default ↑ with Debt Burden | 18.2% vs 4.3% (high vs low debt-to-income) |
| 2 | Young Adults at Risk | 12.1% default rate (age 25-35) |
| 3 | Employment Stability Matters | 15.3% (< 1 yr) vs 6.5% (5+ yrs) |
| 4 | Income is Strong Predictor | 16.8% (bottom) vs 3.2% (top quartile) |
| 5 | Data Quality Acceptable | 2.3% missing, no critical gaps |

### Dataset Characteristics

- **Size**: 307,511 applicants
- **Target**: 25,417 defaults (8.27%)
- **Features**: 122 raw → 45 engineered
- **Missing Values**: 2.3% overall, 12.4% in education
- **Class Imbalance**: ~12:1 (addressed via SMOTE)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.10 |
| **ML Model** | XGBoost |
| **Imbalance Handling** | Imbalanced-Learn (SMOTE) |
| **LLM** | Claude (Anthropic) |
| **Database** | SQLite |
| **Explainability** | SHAP |
| **Deployment** | Docker + Docker Compose |
| **API Communication** | HTTP (Claude API) |

---

## ⚙️ Configuration

### Environment Variables

See `.env.example` for full list. Key variables:

```env
ANTHROPIC_API_KEY=sk-ant-...  # Required for NL→SQL
DATABASE_PATH=data/credit_data.db
MODEL_PATH=models/credit_risk_model.pkl
LOG_LEVEL=INFO
```

### Docker Configuration

**Port Mapping:**
- `8501` → Streamlit web UI

**Resource Limits:**
- CPU: 2 cores (max), 1 core (reserved)
- Memory: 4 GB (max), 2 GB (reserved)

**Health Check:**
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

---

## 📈 Performance & Optimization

### Model Optimization

**Why XGBoost?**
- ✅ Excellent for tabular data
- ✅ Handles imbalance well with `scale_pos_weight`
- ✅ Fast training & inference
- ✅ Feature importance built-in
- ✅ Proven on Kaggle competitions

**Hyperparameter Tuning:**
- Depth limited to 7 (prevent overfitting)
- Learning rate: 0.05 (conservative, stable)
- SMOTE balancing (addresses class imbalance)

### Prompt Optimization

**Token Usage:**
- Few-shot examples: ~200 tokens
- Schema definition: ~100 tokens
- User question: ~50 tokens
- **Total per query**: ~350 tokens (cost: $0.001-0.002)

**Latency Reduction:**
- Cached schema + examples
- Concurrent query execution
- Fallback hardcoded queries (backup)

---

## 🔍 Explainability Approach

### SHAP (SHapley Additive exPlanations)

**Global Explanation:**
- Feature importance across all predictions
- Shows which factors drive default most often

**Local Explanation (Per-Prediction):**
- Shows contribution of each feature to a specific prediction
- SHAP waterfall: base value + feature deltas = final prediction

**Example Output:**
```
Base value (avg default):     0.083
+ Debt-to-Income (0.40):     +0.150
+ Employment (7 years):      -0.050
+ Credit Score (720):        -0.045
+ Income (500k):             -0.040
+ Age (35):                  +0.020
= Final Prediction:           0.118 (LOW RISK)
```

---

## ⚠️ Known Limitations

1. **Static Income Assumption**
   - Model uses annual income snapshot
   - Real-world: income fluctuates, time-series needed

2. **NL→SQL Limited Patterns**
   - Only 5 core patterns programmed
   - Complex joins/CTEs not supported
   - Fallback: hardcoded common queries

3. **Dataset Age**
   - Data from 2005-2015 (11 years old)
   - Current credit landscape may differ

4. **UI is Demo Grade**
   - Not production-ready
   - Single user (no multi-tenancy)
   - No audit logging

5. **Model Bias**
   - Trained on historical data (may contain bias)
   - No fairness constraints applied
   - Recommend fairness audit before production

---

## 🚀 Future Improvements

### Short-term
- [ ] Add time-series features (income trends)
- [ ] Implement more NL→SQL patterns
- [ ] Add model explainability dashboard (SHAP plots)
- [ ] Implement result caching

### Medium-term
- [ ] Fairness audit & bias mitigation
- [ ] A/B testing framework
- [ ] API gateway (REST/GraphQL)
- [ ] Audit logging & monitoring

### Long-term
- [ ] Real-time data pipeline (Kafka)
- [ ] Model retraining automation
- [ ] Multi-model ensemble
- [ ] Production monitoring (Prometheus/Grafana)

---

## 📝 Submission Checklist

- [x] GitHub repo public
- [x] `docker-compose up` runs without errors
- [x] Streamlit app loads at `localhost:8501`
- [x] ML model predicts with risk score + band
- [x] NL→SQL responds to 5+ question types
- [x] README covers setup + architecture + metrics
- [x] `.env.example` has all variables
- [x] No virtual env, dataset, or large files in git
- [x] Presentation PDF in `/documents/`
- [x] Form submission link provided

---

## 📞 Support & Troubleshooting

### Docker Issues

```bash
# Check logs
docker logs credit-risk-platform

# Rebuild image
docker-compose build --no-cache

# Stop and restart
docker-compose down && docker-compose up
```

### Python Issues

```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear cache
pip cache purge
```

### Model Issues

```bash
# Retrain model
python ml_train.py

# Check model file exists
ls -la models/credit_risk_model.pkl
```

---

## 📄 License & Attribution

- **Dataset**: Home Credit Default Risk (Kaggle)
- **Framework**: Streamlit, XGBoost, Claude API
- **Assignment**: Neostats AI Engineer Internship

---

## ✍️ Author

**Vinmatha**  
M.Sc. Computer Science & Applications  
Christ University, Bengaluru  
September 2026

---
