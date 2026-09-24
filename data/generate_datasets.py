"""EKAM Phase 2 — Dataset generation

Creates synthetic e-commerce datasets used by later phases.
Outputs CSV files into this same folder:
- products.csv
- users.csv
- interactions.csv

Run:
    python data/generate_datasets.py

Optional validation:
    python data/generate_datasets.py --validate

Notes:
- Deterministic via --seed.
- No external data needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REAL_CATALOG = {
    "Smartphones": {
        "brands": ["Samsung", "Apple", "OnePlus", "Xiaomi", "Google"],
        "names": ["Galaxy Pro", "iPhone Series", "Nord Edge", "Redmi Note", "Pixel Core"],
        "price": (180, 1400),
        "images": [
            "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1598327105666-5b89351aff97?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Laptops": {
        "brands": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
        "names": ["Inspiron", "Pavilion", "ThinkBook", "MacBook Air", "VivoBook"],
        "price": (420, 2400),
        "images": [
            "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1517336714739-489689fd1ca8?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Headphones": {
        "brands": ["Sony", "JBL", "Boat", "Sennheiser", "Bose"],
        "names": ["Bass Boost", "Noise Cancel Pro", "Wireless Fit", "Studio Sound", "Air Beats"],
        "price": (20, 420),
        "images": [
            "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1484704849700-f032a568e944?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Sneakers": {
        "brands": ["Nike", "Adidas", "Puma", "Reebok", "New Balance"],
        "names": ["Air Runner", "Street Flex", "Ultra Motion", "Cloud Step", "Velocity Knit"],
        "price": (35, 260),
        "images": [
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Watches": {
        "brands": ["Titan", "Fossil", "Casio", "Apple", "Noise"],
        "names": ["Chrono Classic", "Urban Smart", "Sport Pulse", "Digital Pro", "Metal Edge"],
        "price": (25, 950),
        "images": [
            "https://images.unsplash.com/photo-1522312346375-d1a52e2b99b3?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1434056886845-dac89ffe9b56?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Backpacks": {
        "brands": ["Wildcraft", "Skybags", "American Tourister", "Nike", "Puma"],
        "names": ["City Pack", "Travel Pro", "Trail Max", "Campus Lite", "Commuter Smart"],
        "price": (20, 180),
        "images": [
            "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Home Appliances": {
        "brands": ["LG", "Samsung", "Whirlpool", "Philips", "Panasonic"],
        "names": ["AirCool Plus", "Smart Mixer", "Power Vacuum", "Steam Iron", "Aqua Purifier"],
        "price": (45, 800),
        "images": [
            "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1556911220-bda9f7f7597e?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Beauty": {
        "brands": ["Maybelline", "Lakme", "L'Oreal", "Nykaa", "Mamaearth"],
        "names": ["Glow Serum", "Matte Lip Kit", "Hydra Cream", "Skin Tint", "Care Essentials"],
        "price": (8, 140),
        "images": [
            "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1596462502278-27bfdc403348?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Utilities": {
        "brands": ["Syska", "Havells", "Anchor", "Wipro", "Philips"],
        "names": ["LED Bulb Pack", "Extension Board", "Emergency Light", "Water Filter Cartridge", "Power Strip"],
        "price": (6, 180),
        "images": [
            "https://images.unsplash.com/photo-1519710164239-da123dc03ef4?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1585771724684-38269d6639fd?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Groceries": {
        "brands": ["Tata", "Aashirvaad", "Fortune", "Saffola", "Dabur"],
        "names": ["Basmati Rice", "Whole Wheat Atta", "Sunflower Oil", "Peanut Butter", "Organic Honey"],
        "price": (4, 120),
        "images": [
            "https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1610348725531-843dff563e2c?auto=format&fit=crop&w=900&q=80",
        ],
    },
    "Home Essentials": {
        "brands": ["Scotch-Brite", "Lizol", "Vim", "Surf Excel", "Harpic"],
        "names": ["Floor Cleaner", "Dishwash Gel", "Microfiber Cloth Set", "Laundry Detergent", "Toilet Cleaner"],
        "price": (3, 90),
        "images": [
            "https://images.unsplash.com/photo-1581578731548-c64695cc6952?auto=format&fit=crop&w=900&q=80",
            "https://images.unsplash.com/photo-1563453392212-326f5e854473?auto=format&fit=crop&w=900&q=80",
        ],
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 42
    num_users: int = 500
    num_products: int = 300
    num_categories: int = 20

    # Interactions are generated as implicit feedback.
    min_interactions_per_user: int = 10
    max_interactions_per_user: int = 80

    # Time window for interactions
    days_back: int = 180


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_products(cfg: DatasetConfig, rng: np.random.Generator) -> pd.DataFrame:
    catalog_categories = list(REAL_CATALOG.keys())
    categories = rng.choice(catalog_categories, size=cfg.num_products, replace=True)

    availability = rng.choice(["in_stock", "low_stock", "out_of_stock"], size=cfg.num_products, p=[0.72, 0.22, 0.06])
    ratings = np.round(rng.uniform(2.8, 5.0, size=cfg.num_products), 1)

    product_names: list[str] = []
    brands: list[str] = []
    prices: list[float] = []
    image_urls: list[str] = []
    serial = rng.integers(100, 999, size=cfg.num_products)

    for i, cat in enumerate(categories):
        meta = REAL_CATALOG[cat]
        brand = str(rng.choice(meta["brands"]))
        base_name = str(rng.choice(meta["names"]))
        product_names.append(f"{brand} {base_name} {serial[i]}")
        brands.append(brand)
        low, high = meta["price"]
        prices.append(float(np.round(rng.uniform(low, high), 2)))
        image_urls.append(str(rng.choice(meta["images"])))

    df = pd.DataFrame(
        {
            "product_id": np.arange(1, cfg.num_products + 1, dtype=int),
            "product_name": product_names,
            "category": categories,
            "brand": brands,
            "price": prices,
            "rating": ratings,
            "availability": availability,
            "image_url": image_urls,
        }
    )
    return df


def generate_users(cfg: DatasetConfig, rng: np.random.Generator) -> pd.DataFrame:
    # Synthetic demographics (lightweight — useful for later analytics)
    cities = [
        "San Jose",
        "Austin",
        "Seattle",
        "New York",
        "Chicago",
        "Denver",
        "Boston",
        "Los Angeles",
        "Atlanta",
        "Miami",
    ]
    genders = ["Female", "Male", "Non-binary"]

    ages = rng.integers(18, 70, size=cfg.num_users)
    signup_days_ago = rng.integers(0, cfg.days_back, size=cfg.num_users)

    now = _utc_now()
    signup_at = [now - timedelta(days=int(d)) for d in signup_days_ago]

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, cfg.num_users + 1, dtype=int),
            "age": ages,
            "gender": rng.choice(genders, size=cfg.num_users, p=[0.48, 0.49, 0.03]),
            "city": rng.choice(cities, size=cfg.num_users, replace=True),
            "signup_at": pd.to_datetime(signup_at).astype(str),
        }
    )
    return df


def generate_interactions(
    cfg: DatasetConfig, rng: np.random.Generator, products: pd.DataFrame, users: pd.DataFrame
) -> pd.DataFrame:
    """Backward-compatible wrapper.

    Kept for readability; actual implementation is in
    `build_interactions_from_components`.
    """
    return build_interactions_from_components(cfg, rng, products, users)



def build_interactions_from_components(
    cfg: DatasetConfig, rng: np.random.Generator, products: pd.DataFrame, users: pd.DataFrame
) -> pd.DataFrame:
    now = _utc_now()

    categories = products["category"].unique().tolist()


    def age_bucket(age: int) -> int:
        if age < 30:
            return 0
        if age < 45:
            return 1
        if age < 60:
            return 2
        return 3

    cat_idx = {c: i for i, c in enumerate(categories)}
    weights = np.zeros((4, len(categories)), dtype=float)
    for c in categories:
        i = cat_idx[c]
        base = 1.0 + (i % 9) * 0.07
        weights[0, i] = base * (1.0 + (i % 3) * 0.10)
        weights[1, i] = base * (1.0 + ((i + 1) % 3) * 0.10)
        weights[2, i] = base * (1.0 + ((i + 2) % 3) * 0.10)
        weights[3, i] = base * (1.0 + ((i + 3) % 3) * 0.10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    cat_to_product_ids = {
        c: products.loc[products["category"] == c, "product_id"].to_numpy() for c in categories
    }

    prod_price = products.set_index("product_id")["price"].to_dict()

    event_types = np.array(["view", "cart", "purchase"], dtype=object)

    out_rows = []

    for _, u in users.iterrows():
        uid = int(u["user_id"])
        bucket = age_bucket(int(u["age"]))
        n = int(rng.integers(cfg.min_interactions_per_user, cfg.max_interactions_per_user + 1))

        chosen_cats = rng.choice(categories, size=n, replace=True, p=weights[bucket])
        chosen_products = np.array([rng.choice(cat_to_product_ids[c]) for c in chosen_cats], dtype=int)

        days_ago = rng.integers(0, cfg.days_back, size=n)
        minutes_in_day = rng.integers(0, 24 * 60, size=n)
        ts = [now - timedelta(days=int(d), minutes=int(m)) for d, m in zip(days_ago, minutes_in_day)]

        chosen_prices = np.array([float(prod_price[int(pid)]) for pid in chosen_products], dtype=float)
        price_norm = (chosen_prices - chosen_prices.min()) / (chosen_prices.max() - chosen_prices.min() + 1e-9)

        p_cart = 0.22 - 0.10 * price_norm
        p_purchase = 0.06 - 0.02 * price_norm
        p_cart = np.clip(p_cart, 0.05, 0.22)
        p_purchase = np.clip(p_purchase, 0.01, 0.08)
        p_view = 1.0 - p_cart - p_purchase
        p_view = np.clip(p_view, 0.6, 0.95)
        denom = p_view + p_cart + p_purchase
        p_view, p_cart, p_purchase = p_view / denom, p_cart / denom, p_purchase / denom

        event_idx = [
            int(rng.choice([0, 1, 2], p=[float(pv), float(pc), float(pp)]))
            for pv, pc, pp in zip(p_view, p_cart, p_purchase)
        ]
        event_type = event_types[event_idx]

        quantity = np.where(event_type == "view", rng.integers(1, 2, size=n), rng.integers(1, 4, size=n))

        for i in range(n):
            out_rows.append(
                {
                    "user_id": uid,
                    "product_id": int(chosen_products[i]),
                    "event_type": str(event_type[i]),
                    "quantity": int(quantity[i]),
                    "event_at": str(pd.to_datetime(ts[i]).to_pydatetime()),
                }
            )

    df = pd.DataFrame(out_rows)
    df = df.sort_values("event_at").reset_index(drop=True)
    return df


def validate_datasets(products_path: str, users_path: str, interactions_path: str) -> None:
    products = pd.read_csv(products_path)
    users = pd.read_csv(users_path)
    interactions = pd.read_csv(interactions_path)

    required_products_cols = {"product_id", "product_name", "category", "brand", "price", "rating", "availability"}
    required_users_cols = {"user_id", "age", "gender", "city", "signup_at"}
    required_interactions_cols = {"user_id", "product_id", "event_type", "quantity", "event_at"}

    if not required_products_cols.issubset(set(products.columns)):
        raise ValueError(f"products.csv missing cols. required={required_products_cols}, got={set(products.columns)}")
    if not required_users_cols.issubset(set(users.columns)):
        raise ValueError(f"users.csv missing cols. required={required_users_cols}, got={set(users.columns)}")
    if not required_interactions_cols.issubset(set(interactions.columns)):
        raise ValueError(
            f"interactions.csv missing cols. required={required_interactions_cols}, got={set(interactions.columns)}"
        )

    if len(products) == 0 or len(users) == 0 or len(interactions) == 0:
        raise ValueError("One or more generated datasets are empty.")

    # Basic consistency checks
    if interactions["user_id"].min() < users["user_id"].min() or interactions["user_id"].max() > users["user_id"].max():
        raise ValueError("interactions.csv has user_id values out of bounds.")
    if interactions["product_id"].min() < products["product_id"].min() or interactions["product_id"].max() > products["product_id"].max():
        raise ValueError("interactions.csv has product_id values out of bounds.")

    allowed_events = {"view", "cart", "purchase"}
    if not set(interactions["event_type"].unique()).issubset(allowed_events):
        raise ValueError(f"Unexpected event_type values: {set(interactions['event_type'].unique())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_users", type=int, default=500)
    parser.add_argument("--num_products", type=int, default=300)
    parser.add_argument("--num_categories", type=int, default=20)
    parser.add_argument("--days_back", type=int, default=180)
    parser.add_argument("--min_interactions_per_user", type=int, default=10)
    parser.add_argument("--max_interactions_per_user", type=int, default=80)
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()

    cfg = DatasetConfig(
        seed=args.seed,
        num_users=args.num_users,
        num_products=args.num_products,
        num_categories=args.num_categories,
        days_back=args.days_back,
        min_interactions_per_user=args.min_interactions_per_user,
        max_interactions_per_user=args.max_interactions_per_user,
    )

    rng = np.random.default_rng(cfg.seed)

    products = generate_products(cfg, rng)
    users = generate_users(cfg, rng)
    interactions = build_interactions_from_components(cfg, rng, products, users)

    data_dir = __import__("pathlib").Path(__file__).resolve().parent
    products_path = str(data_dir / "products.csv")
    users_path = str(data_dir / "users.csv")
    interactions_path = str(data_dir / "interactions.csv")

    products.to_csv(products_path, index=False)
    users.to_csv(users_path, index=False)
    interactions.to_csv(interactions_path, index=False)

    print("Phase 2 dataset generation complete.")
    print(f"- {products_path} ({len(products):,} rows)")
    print(f"- {users_path} ({len(users):,} rows)")
    print(f"- {interactions_path} ({len(interactions):,} rows)")

    if args.validate:
        validate_datasets(products_path, users_path, interactions_path)
        print("Validation passed.")


if __name__ == "__main__":
    main()

