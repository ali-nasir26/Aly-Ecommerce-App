# Aly — AI-Enabled E-Commerce Application

Beginner-to-intermediate AI/ML portfolio project built with:
- **Python**
- **Streamlit** (web UI)
- **Pandas / NumPy** (data handling)
- **scikit-learn** (recommendation ML)
- **Plotly** (analytics charts)
- **Joblib** (saving/loading ML artifacts)
- **SHAP** (explainability — later phases)

This project is built in **phases**.

This repository currently contains:
- **Phase 1 (Project Setup)**
- **Phase 2 (Dataset creation)**
- **Phase 3 (Streamlit frontend)**
- **Phase 4 (Product recommendation system)**
- **Phase 5 (Analytics dashboard)**
- **Phase 6 (AI explainability — SHAP)**
- **Phase 7 (Admin features)**
- **Phase 8 (SQLite integration)**
- **Phase 9 (Cloud deployment guidance)**
- **Phase 10 (Interview preparation notes)**


---

## Folder Structure (Phase 1)

```text
Aly/
  app/
  data/
  models/
  notebooks/
  utils/
  assets/
  saved_models/
  requirements.txt
  README.md
```

### Why this structure matters
- **app/**: Streamlit pages and runtime UI code.
- **data/**: CSV datasets (initially) and later any local data used.
- **models/**: Training scripts and model-related utilities.
- **notebooks/**: Exploratory analysis and experiments.
- **utils/**: Helper functions reused across the app.
- **assets/**: Images/icons for a nicer UI.
- **saved_models/**: Persisted ML artifacts (vectorizers/models).

---

## Phase 1 — Setup

### 1) Create a virtual environment
Windows (PowerShell or CMD):

```bash
python -m venv .venv
```

Activate it:
- PowerShell:
  ```bash
  .venv\Scripts\Activate.ps1
  ```
- CMD:
  ```bash
  .venv\Scripts\activate.bat
  ```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the Streamlit app
```bash
streamlit run app/main.py
```

If the app reports missing datasets, open the **Admin / Data** tab and click **Generate datasets**, or run:
```bash
python data/generate_datasets.py --validate
```

### 4) (Optional) Train recommendation artifacts
```bash
python models/train_model.py --validate
```

### 5) (Optional) Switch data backend to SQLite
Generate CSV datasets first, then migrate:
```bash
python data/migrate_to_sqlite.py --no-backup --clear
```
Or run the same migration from the **Admin / Data** page using **Migrate CSV to SQLite**.

When `data/Aly.db` exists and has data, the app automatically uses SQLite.  
Otherwise it falls back to CSV files.

---

## Git / GitHub commands

```bash
git init
git add .
git commit -m "chore: phase 1 setup"
```

When ready to push to GitHub:
```bash
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

---

## Streamlit Cloud Deployment (Phase 9)

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **New app**.
3. Select repository/branch and set entrypoint to:
   - `app/main.py`
4. Ensure `requirements.txt` is present at the repo root.
5. Deploy and verify:
   - Home page loads
   - Catalog and Analytics render without errors
   - Admin page can generate datasets (if CSVs are missing)

Recommended pre-deploy checks:
```bash
pip install -r requirements.txt
python data/generate_datasets.py --validate
python models/train_model.py --validate
streamlit run app/main.py
```

---

## Interview Preparation (Phase 10)

### Architecture (talk track)
- **Presentation layer**: Streamlit app (`app/main.py`) with modules for Home, Catalog, Analytics, Recommendations, Admin.
- **Data layer**: CSV-first workflow with optional SQLite backend (`utils/database.py`).
- **ML layer**: TF-IDF content-based recommender (`models/train_model.py`, `models/recommendation_engine.py`).
- **Explainability layer**: recommendation rationale using feature contributions (`models/explainability.py`).

### Resume-ready project description
Built an AI-enabled e-commerce analytics and recommendation app using Python + Streamlit.  
Implemented synthetic data generation, TF-IDF/cosine product recommendations, explainability signals, admin inventory CRUD, and an optional SQLite migration path from flat CSV storage.

### Common interview Q&A prompts
- **Why TF-IDF for recommendations?**  
  It is simple, fast, interpretable, and works well for cold-start item similarity without user-history depth.
- **How did you ensure modularity?**  
  Split responsibilities into data generation, model training, inference, explainability, and UI/admin modules.
- **How would you scale this project?**  
  Move to managed DB + API layer, add user-personalized ranking models, implement caching, and introduce CI tests/deployment checks.
- **What are current limitations?**  
  Synthetic data assumptions, lightweight explainability approximation, and Streamlit-centric architecture for prototype velocity.

---

## How we progress
Each phase is designed to be **fully runnable** before moving to the next.
- ✅ Phase 1: project setup + base skeleton
- ✅ Phase 2: dataset creation
- ✅ Phase 3: Streamlit frontend
- ✅ Phase 4: recommendation engine
- ✅ Phase 5: analytics dashboard
- ✅ Phase 6: explainability
- ✅ Phase 7: admin features
- ✅ Phase 8: SQLite integration
- ✅ Phase 9: cloud deployment guide
- ✅ Phase 10: interview preparation pack


