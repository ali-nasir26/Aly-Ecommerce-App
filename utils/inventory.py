"""CSV-based inventory management utilities for EKAM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_products(csv_path: Path | str) -> pd.DataFrame:
    """Load products from CSV."""
    return pd.read_csv(csv_path)


def save_products(df: pd.DataFrame, csv_path: Path | str) -> None:
    """Save products to CSV."""
    df.to_csv(csv_path, index=False)


def add_product(
    df: pd.DataFrame,
    product_name: str,
    category: str,
    brand: str,
    price: float,
    rating: float,
    availability: str,
) -> tuple[pd.DataFrame, int]:
    """Add a new product to the inventory.
    
    Returns the updated DataFrame and the new product_id.
    """
    new_product_id = int(df["product_id"].max() + 1) if len(df) > 0 else 1
    new_row = pd.DataFrame([{
        "product_id": new_product_id,
        "product_name": product_name,
        "category": category,
        "brand": brand,
        "price": price,
        "rating": rating,
        "availability": availability,
    }])
    return pd.concat([df, new_row], ignore_index=True), new_product_id


def update_product(
    df: pd.DataFrame,
    product_id: int,
    **kwargs,
) -> pd.DataFrame:
    """Update fields for a specific product.
    
    Args:
        df: Product DataFrame
        product_id: Product ID to update
        **kwargs: Fields to update (e.g., price=99.99, availability="in_stock")
    
    Returns the updated DataFrame.
    """
    if product_id not in df["product_id"].values:
        raise ValueError(f"Product ID {product_id} not found.")
    
    idx = df[df["product_id"] == product_id].index[0]
    for key, value in kwargs.items():
        if key in df.columns:
            df.loc[idx, key] = value
    return df


def remove_product(df: pd.DataFrame, product_id: int) -> pd.DataFrame:
    """Remove a product from the inventory.
    
    Returns the updated DataFrame.
    """
    if product_id not in df["product_id"].values:
        raise ValueError(f"Product ID {product_id} not found.")
    return df[df["product_id"] != product_id].reset_index(drop=True)


def validate_product_data(**kwargs) -> list[str]:
    """Validate product data.
    
    Returns a list of validation errors (empty if valid).
    """
    errors = []
    
    if "product_name" in kwargs and not kwargs["product_name"]:
        errors.append("Product name cannot be empty.")
    
    if "category" in kwargs and not kwargs["category"]:
        errors.append("Category cannot be empty.")
    
    if "brand" in kwargs and not kwargs["brand"]:
        errors.append("Brand cannot be empty.")
    
    if "price" in kwargs:
        try:
            price = float(kwargs["price"])
            if price < 0:
                errors.append("Price must be non-negative.")
        except (ValueError, TypeError):
            errors.append("Price must be a valid number.")
    
    if "rating" in kwargs:
        try:
            rating = float(kwargs["rating"])
            if not (1.0 <= rating <= 5.0):
                errors.append("Rating must be between 1.0 and 5.0.")
        except (ValueError, TypeError):
            errors.append("Rating must be a valid number.")
    
    if "availability" in kwargs:
        valid_availability = {"in_stock", "low_stock", "out_of_stock"}
        if kwargs["availability"] not in valid_availability:
            errors.append(f"Availability must be one of: {', '.join(valid_availability)}")
    
    return errors
