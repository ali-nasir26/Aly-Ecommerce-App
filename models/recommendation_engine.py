"""Recommendation helper utilities for EKAM item-to-item similarity."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"


def load_model_artifacts(model_dir: Path | str = DEFAULT_MODEL_DIR) -> tuple[TfidfVectorizer, object, list[str]]:
    model_dir = Path(model_dir)
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    item_vectors = joblib.load(model_dir / "item_vectors.joblib")
    product_ids = joblib.load(model_dir / "product_ids.joblib")
    return vectorizer, item_vectors, product_ids


def _build_product_feature_text(products: pd.DataFrame) -> pd.Series:
    features = []
    for _, row in products.iterrows():
        values = [
            str(row.get("product_name", "")),
            str(row.get("category", "")),
            str(row.get("brand", "")),
        ]
        features.append(" ".join(value for value in values if value))
    return pd.Series(features, index=products.index)


def recommend_similar_products(
    product_id: int | str,
    products: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    item_vectors: object,
    product_ids: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    product_id = str(product_id)
    if product_id not in product_ids:
        raise KeyError(f"Product ID {product_id} not found in saved recommendation model.")

    index = product_ids.index(product_id)
    scores = cosine_similarity(item_vectors[index], item_vectors).flatten()
    sorted_indices = np.argsort(scores)[::-1]
    filtered = [i for i in sorted_indices if i != index]
    selected = filtered[:top_n]

    recommendations = products.reset_index(drop=True).iloc[selected].copy()
    recommendations["similarity_score"] = scores[selected]
    return recommendations


def recommend_by_query(
    query: str,
    products: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    item_vectors: object,
    top_n: int = 10,
) -> pd.DataFrame:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, item_vectors).flatten()
    sorted_indices = np.argsort(scores)[::-1][:top_n]
    recommendations = products.reset_index(drop=True).iloc[sorted_indices].copy()
    recommendations["similarity_score"] = scores[sorted_indices]
    return recommendations
