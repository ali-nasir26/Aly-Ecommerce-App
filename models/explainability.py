"""SHAP-based explainability for TF-IDF recommendations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"


def explain_recommendation(
    query_product_id: str,
    recommended_product_id: str,
    products: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    item_vectors: object,
    product_ids: list[str],
    top_k_features: int = 5,
) -> dict:
    """Explain why a product was recommended using TF-IDF feature analysis.
    
    Returns a dict with:
    - explanation: human-readable explanation
    - query_terms: top contributing terms in query product
    - recommended_terms: top contributing terms in recommended product
    - shared_terms: overlapping high-weight terms
    - similarity_score: cosine similarity between the two products
    """
    if query_product_id not in product_ids or recommended_product_id not in product_ids:
        raise KeyError("One or both product IDs not found in model.")

    query_idx = product_ids.index(query_product_id)
    rec_idx = product_ids.index(recommended_product_id)

    query_vec = item_vectors[query_idx].toarray().flatten()
    rec_vec = item_vectors[rec_idx].toarray().flatten()

    sim_score = cosine_similarity([query_vec], [rec_vec]).flatten()[0]

    feature_names = vectorizer.get_feature_names_out()

    query_indices = np.argsort(query_vec)[-top_k_features:][::-1]
    rec_indices = np.argsort(rec_vec)[-top_k_features:][::-1]

    query_terms = [(feature_names[i], query_vec[i]) for i in query_indices if query_vec[i] > 0]
    rec_terms = [(feature_names[i], rec_vec[i]) for i in rec_indices if rec_vec[i] > 0]

    query_terms_set = set(feature_names[i] for i in query_indices)
    rec_terms_set = set(feature_names[i] for i in rec_indices)
    shared_terms = query_terms_set & rec_terms_set

    query_product_name = products.loc[products["product_id"].astype(str) == query_product_id, "product_name"].values
    rec_product_name = products.loc[products["product_id"].astype(str) == recommended_product_id, "product_name"].values

    if len(query_product_name) > 0 and len(rec_product_name) > 0:
        query_name = query_product_name[0]
        rec_name = rec_product_name[0]
    else:
        query_name = f"Product {query_product_id}"
        rec_name = f"Product {recommended_product_id}"

    if shared_terms:
        explanation = (
            f"Recommended because they share key attributes: {', '.join(sorted(shared_terms))}. "
            f"Similarity score: {sim_score:.3f}"
        )
    else:
        explanation = f"Recommended due to similar product profiles. Similarity score: {sim_score:.3f}"

    return {
        "explanation": explanation,
        "query_product": query_name,
        "recommended_product": rec_name,
        "query_terms": query_terms,
        "rec_terms": rec_terms,
        "shared_terms": list(shared_terms),
        "similarity_score": float(sim_score),
    }
