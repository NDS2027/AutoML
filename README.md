# 🎯 ChurnGuard AI - Intelligent Customer Retention Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://mongodb.com)

An **end-to-end Machine Learning platform** that predicts customer churn, identifies at-risk customers, and provides AI-powered actionable insights to maximize revenue retention.

![Dashboard Preview](docs/dashboard_preview.png)

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **🤖 AutoML Engine** | Automatically trains and compares 5 ML models (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM) |
| **📊 Smart Dashboard** | Business-focused insights with risk segmentation and revenue-at-risk metrics |
| **💡 AI Chat Assistant** | LLM-powered Q&A about your data using Ollama (local) or OpenAI |
| **🎮 What-If Simulator** | ROI calculator to simulate retention campaigns before spending |
| **🔍 Customer Explorer** | Drill down into individual customer risk profiles |
| **📈 Feature Engineering** | 24+ engineered features including RFM, behavioral, and temporal patterns |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│    MongoDB      │
│   Frontend      │     │    Backend      │     │    Atlas        │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  AutoML  │ │  SHAP    │ │  Ollama  │
              │  Engine  │ │  Explainer│ │   LLM    │
              └──────────┘ └──────────┘ └──────────┘
```

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI (REST API)
- MongoDB Atlas (Database)
- Scikit-learn, XGBoost, LightGBM (ML Models)
- SMOTE (Imbalanced Data Handling)
- SHAP (Model Explainability)

**Frontend:**
- Streamlit (Interactive Dashboard)
- Plotly (Data Visualization)

**AI/ML:**
- AutoML with 5 model comparison
- 24+ engineered features (RFM, Temporal, Behavioral)
- Ollama/OpenAI LLM Integration

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free tier works)
- Ollama (optional, for AI chat)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/churnguard-ai.git
cd churnguard-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set environment variables
set MONGODB_URL=your_mongodb_connection_string

# Start backend (Terminal 1)
cd backend
uvicorn app:app --reload --port 8000

# Start frontend (Terminal 2)
cd frontend
streamlit run app.py
```

### Optional: Enable AI Chat
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
```

---

## 📊 How It Works

1. **Upload** → Upload CSV/Excel with transaction data
2. **Configure** → Map columns (Customer ID, Date, Amount, Product)
3. **Train** → AutoML trains 5 models, selects the best
4. **Explore** → View dashboard with risk segments and insights
5. **Act** → Use What-If simulator to plan retention campaigns
6. **Ask** → Chat with AI about your data

---

## 📁 Project Structure

```
ChurnGuard-AI/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── core/             # ML pipeline modules
│   │   ├── automl_engine.py      # Model training & selection
│   │   ├── feature_engineer.py   # Feature engineering
│   │   ├── data_profiler.py      # Data cleaning
│   │   ├── explainability.py     # SHAP explanations
│   │   ├── insight_generator.py  # Business insights
│   │   └── llm_service.py        # LLM integration
│   ├── models/           # Pydantic schemas
│   └── utils/            # Helper functions
├── frontend/
│   ├── app.py            # Main Streamlit app
│   └── pages/            # Dashboard pages
├── data/samples/         # Sample datasets
└── requirements.txt
```

---

## 🎯 Sample Results

| Metric | Value |
|--------|-------|
| Customers Analyzed | 500+ |
| Models Trained | 5 |
| Champion Model | Random Forest |
| Features Engineered | 24 |
| High-Risk Identified | 18% |

---

## 📄 License

MIT License - feel free to use for learning and projects.

---

## 👨‍💻 Author

Built by [Your Name] as a demonstration of end-to-end ML system design.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset
- Streamlit for the amazing dashboard framework
- FastAPI for the blazing-fast backend
