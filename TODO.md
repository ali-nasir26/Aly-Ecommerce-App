# Aly — TODO Tracker

## Phase 1: Project Setup (current)
- [x] Create project folder skeleton
- [x] Add README.md (phase overview)
- [x] Add requirements.txt
- [x] Add .gitignore
- [x] Add minimal app entrypoint (Phase 1 sanity check)
- [x] Add instructions to run Phase 1

## Phase 2: Dataset creation
- [x] Add synthetic dataset generation script (`data/generate_datasets.py`)
- [x] Update docs with how to regenerate datasets

## Phase 3: Streamlit frontend
- [x] Implement multi-section Streamlit UI (Home / Catalog / Analytics / Admin)
- [x] Add dataset loading + "Generate datasets" button when CSVs are missing
- [x] Wire UI to `products.csv`, `users.csv`, `interactions.csv`
- [x] Update run instructions for Phase 3


## Phase 4: Product recommendation system
- [x] Add TF-IDF + cosine similarity based content recommendation
- [x] Implement `models/train_model.py`
- [x] Implement `models/recommendation_engine.py`

## Phase 5: Analytics dashboard
- [x] Sales-like analytics using interactions dataset
- [x] Category distribution, rating distribution, price analysis

## Phase 6: AI explainability (SHAP)
- [x] Add SHAP explainability for recommendations (where feasible)

## Phase 7: Admin features
- [x] Add/remove/update inventory (CSV-backed for now)
- [x] Upload CSV via Streamlit form

## Phase 8: SQLite integration
- [x] Add `database.py` and migration scripts
- [x] Implement CRUD against SQLite
- [x] Switch Streamlit inventory CRUD to SQLite
- [x] Switch dataset loading to SQLite when available

## Phase 9: Cloud deployment
- [x] Streamlit Cloud deployment steps
- [x] GitHub push + requirements verification

## Phase 10: Interview preparation
- [x] Architecture explanation
- [x] Resume-ready project description
- [x] Common interview questions + talking points

