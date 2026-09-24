"""Train a TF-IDF item similarity model for EKAM product recommendations.

This script consumes `data/products.csv` and persists artifacts into `saved_models/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "saved_models"


def _build_product_feature_text(products: pd.DataFrame) -> pd.Series:
    features = []
    for _, row in products.iterrows():
        fields = [
            str(row.get("product_name", "")),
            str(row.get("category", "")),
            str(row.get("brand", "")),
        ]
        features.append(" ".join(field for field in fields if field))
    return pd.Series(features, index=products.index)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_recommendation_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_df: int = 1,
) -> tuple[TfidfVectorizer, np.ndarray, list[str], pd.DataFrame]:
    products_path = data_dir / "products.csv"
    products = pd.read_csv(products_path)

    if "product_id" not in products.columns:
        raise ValueError("products.csv must contain a 'product_id' column.")

    index = products["product_id"].astype(str).tolist()
    feature_text = _build_product_feature_text(products)

    vectorizer = TfidfVectorizer(min_df=min_df, stop_words="english")
    matrix = vectorizer.fit_transform(feature_text)
    matrix = normalize(matrix, norm="l2", axis=1)

    _ensure_output_dir(output_dir)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, output_dir / "item_vectors.joblib")
    joblib.dump(index, output_dir / "product_ids.joblib")

    return vectorizer, matrix, index, products


def validate_model_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    vectorizer = joblib.load(output_dir / "tfidf_vectorizer.joblib")
    item_vectors = joblib.load(output_dir / "item_vectors.joblib")
    product_ids = joblib.load(output_dir / "product_ids.joblib")

    if item_vectors.shape[0] != len(product_ids):
        raise ValueError("Saved vectors and product IDs length mismatch.")
    if item_vectors.shape[1] != len(vectorizer.get_feature_names_out()):
        raise ValueError("Saved vector dimension does not match the vectorizer vocabulary.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF based product recommendation model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to the data folder containing products.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to save recommendation artifacts.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for TF-IDF terms.")
    parser.add_argument("--validate", action="store_true", help="Validate artifacts after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectorizer, item_vectors, product_ids, products = train_recommendation_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_df=args.min_df,
    )

    print(f"Saved recommendation artifacts to {args.output_dir}")
    print(f"- vectorizer vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"- item vectors: {item_vectors.shape}")
    print(f"- product IDs: {len(product_ids)}")

    if args.validate:
        validate_model_artifacts(args.output_dir)
        print("Model artifact validation passed.")


if __name__ == "__main__":
    main()
