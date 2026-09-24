"""EKAM Phase 3 — Streamlit frontend UI.

Implements multi-section UI:
- Home
- Catalog (products)
- Analytics (users/interactions)
- Admin/Data (dataset status + regeneration)

Data sources (Phase 2 synthetic datasets):
- data/products.csv
- data/users.csv
- data/interactions.csv

Run:
    streamlit run app/main.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))



from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from models.recommendation_engine import load_model_artifacts, recommend_similar_products
from models.explainability import explain_recommendation
from utils.database import Database
from utils.inventory import (
    load_products,
    save_products,
    add_product,
    update_product,
    remove_product,
    validate_product_data,
)



APP_TITLE = "EKAM — AI-Enabled E-Commerce"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "app.log"

PRODUCTS_CSV = DATA_DIR / "products.csv"
USERS_CSV = DATA_DIR / "users.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"
SQLITE_DB = DATA_DIR / "ekam.db"


def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ekam.app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = _setup_logging()


@dataclass(frozen=True)
class Datasets:
    products: pd.DataFrame
    users: pd.DataFrame
    interactions: pd.DataFrame


def _datasets_exist() -> bool:
    return PRODUCTS_CSV.exists() and USERS_CSV.exists() and INTERACTIONS_CSV.exists()


def _sqlite_ready() -> bool:
    if not SQLITE_DB.exists():
        return False
    try:
        db = Database(SQLITE_DB)
        products = db.get_all_products()
        users = db.get_all_users()
        interactions = db.get_all_interactions()
        return len(products) > 0 and len(users) > 0 and len(interactions) > 0
    except Exception:
        return False


def _active_data_backend() -> str:
    if _sqlite_ready():
        return "sqlite"
    if _datasets_exist():
        return "csv"
    return "none"


def _generate_datasets() -> tuple[bool, str]:
    script_path = Path(__file__).resolve().parent.parent / "data" / "generate_datasets.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--validate"],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            check=True,
        )
        LOGGER.info("Datasets generated via script.")
        return True, result.stdout.strip() or "Datasets generated successfully."
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr.strip() if exc.stderr else exc.stdout.strip()
        LOGGER.exception("Dataset generation failed.")
        return False, error_output or str(exc)


@st.cache_data(show_spinner=False)
def load_datasets() -> Datasets:
    backend = _active_data_backend()
    if backend == "sqlite":
        db = Database(SQLITE_DB)
        products = db.get_all_products()
        users = db.get_all_users()
        interactions = db.get_all_interactions()
        LOGGER.info("Loaded datasets from SQLite.")
    else:
        products = pd.read_csv(PRODUCTS_CSV)
        users = pd.read_csv(USERS_CSV)
        interactions = pd.read_csv(INTERACTIONS_CSV)
        LOGGER.info("Loaded datasets from CSV files.")

    # Parse types lightly for UI use
    if "price" in products.columns:
        products["price"] = pd.to_numeric(products["price"], errors="coerce")
    if "rating" in products.columns:
        products["rating"] = pd.to_numeric(products["rating"], errors="coerce")

    if "event_at" in interactions.columns:
        interactions["event_at"] = pd.to_datetime(interactions["event_at"], errors="coerce")

    return Datasets(products=products, users=users, interactions=interactions)


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                max-width: 1200px;
            }
            .ekam-card {
                background: #ffffff;
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 4px 14px rgba(16, 24, 40, 0.08);
                border: 1px solid #e6ecf5;
                margin-bottom: 10px;
            }
            .ekam-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #243b5a;
                margin-bottom: 4px;
            }
            .ekam-sub {
                color: #4b5f7d;
                font-size: 0.92rem;
            }
            .offer-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
            }
            .offer-card {
                position: relative;
                min-height: 220px;
                border-radius: 16px;
                overflow: hidden;
                background-size: cover;
                background-position: center;
                box-shadow: 0 6px 20px rgba(2, 6, 23, 0.25);
            }
            .offer-overlay {
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(15,23,42,0.15) 0%, rgba(15,23,42,0.78) 100%);
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                padding: 14px;
                color: #fff;
            }
            .offer-badge {
                font-size: 1.45rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .offer-caption {
                font-size: 0.9rem;
                opacity: 0.95;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            }
            [data-testid="stSidebar"] * {
                color: #e2e8f0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _navigate_to(page_name: str) -> None:
    st.session_state["nav_page"] = page_name
    st.session_state["nav_page_select"] = page_name
    st.rerun()


def _read_log_tail(max_lines: int = 120) -> str:
    if not LOG_FILE.exists():
        return "No logs yet."
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else "No logs yet."


def _ensure_datasets_or_prompt() -> Optional[Datasets]:
    backend = _active_data_backend()
    if backend in {"sqlite", "csv"}:
        return load_datasets()

    st.error(
        "Datasets not found. Expected CSV files under `data/` or a populated SQLite database (`data/ekam.db`)."
    )
    st.warning("Run Phase 2 dataset generation, or use the Admin tab to generate them.")
    return None


def _format_large_int(x: float | int) -> str:
    try:
        x = int(x)
    except Exception:
        return str(x)
    return f"{x:,}"


def render_home(datasets: Datasets) -> None:
    st.header("Home")
    st.markdown(
        '<div class="ekam-card"><div class="ekam-title">Executive Overview</div><div class="ekam-sub">Visual dashboard for quick understanding.</div></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.metric("Products", len(datasets.products))
    with col2:
        st.metric("Users", len(datasets.users))
    with col3:
        st.metric("Interactions", len(datasets.interactions))

    st.divider()

    st.subheader("Quick Navigation")
    st.caption("Image cards with one-click navigation.")
    nav1, nav2, nav3, nav4, nav5 = st.columns(5)
    with nav1:
        st.image(
            "https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?auto=format&fit=crop&w=600&q=80",
            use_container_width=True,
        )
        st.button("🛍️ Open Catalog", use_container_width=True, on_click=_navigate_to, args=("Catalog",))
    with nav2:
        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=600&q=80",
            use_container_width=True,
        )
        st.button("📈 Open Analytics", use_container_width=True, on_click=_navigate_to, args=("Analytics",))
    with nav3:
        st.image(
            "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=600&q=80",
            use_container_width=True,
        )
        st.button("🔎 Open Explorer", use_container_width=True, on_click=_navigate_to, args=("Explorer",))
    with nav4:
        st.image(
            "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=600&q=80",
            use_container_width=True,
        )
        st.button("👤 Open User Insights", use_container_width=True, on_click=_navigate_to, args=("User Insights",))
    with nav5:
        st.image(
            "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=600&q=80",
            use_container_width=True,
        )
        st.button("⚙️ Open Admin", use_container_width=True, on_click=_navigate_to, args=("Admin / Data",))

    st.divider()
    st.subheader("Offers & Highlights")
    st.markdown(
        """
        <div class="offer-grid">
          <div class="offer-card" style="background-image:url('https://images.unsplash.com/photo-1607082350899-7e105aa886ae?auto=format&fit=crop&w=1200&q=80');">
            <div class="offer-overlay">
              <div class="offer-badge">FLASH SALE<br>70% OFF</div>
              <div class="offer-caption">Limited-time mega discounts on top categories</div>
            </div>
          </div>
          <div class="offer-card" style="background-image:url('https://images.unsplash.com/photo-1556740738-b6a63e27c4df?auto=format&fit=crop&w=1200&q=80');">
            <div class="offer-overlay">
              <div class="offer-badge">MEGA DEAL<br>50% OFF</div>
              <div class="offer-caption">Exciting picks with happy-customer favorites</div>
            </div>
          </div>
          <div class="offer-card" style="background-image:url('https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1200&q=80');">
            <div class="offer-overlay">
              <div class="offer-badge">WEEKEND OFFER<br>UP TO 40% OFF</div>
              <div class="offer-caption">Trending styles and bestselling products</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    viz_left, viz_right = st.columns(2, gap="large")
    with viz_left:
        st.plotly_chart(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": ["Products", "Users", "Interactions"],
                        "y": [len(datasets.products), len(datasets.users), len(datasets.interactions)],
                        "marker": {"color": ["#3b82f6", "#10b981", "#8b5cf6"]},
                    }
                ],
                "layout": {
                    "title": "Platform Overview",
                    "height": 330,
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                },
            },
            use_container_width=True,
        )

    with viz_right:
        category_counts = datasets.products["category"].value_counts().head(8)
        st.plotly_chart(
            {
                "data": [
                    {
                        "type": "pie",
                        "labels": category_counts.index.astype(str).tolist(),
                        "values": category_counts.values.tolist(),
                        "hole": 0.45,
                    }
                ],
                "layout": {
                    "title": "Top Product Categories",
                    "height": 330,
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                },
            },
            use_container_width=True,
        )

    st.caption("Tip: Use Admin if CSV files are missing.")
    backend_label = "SQLite" if _active_data_backend() == "sqlite" else "CSV files"
    st.caption(f"Current data backend: {backend_label}")


def render_explorer(datasets: Datasets) -> None:
    st.header("Explorer")
    st.markdown("Use advanced interactive controls to drill into inventory cohorts.")
    st.image(
        "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=1400&q=80",
        caption="Search, filter, and compare products visually.",
        use_container_width=True,
    )

    products = datasets.products.copy()
    interactions = datasets.interactions.copy()

    keyword = st.text_input("Search product name / brand / category")
    selected_availability = st.multiselect(
        "Availability",
        options=sorted(products["availability"].dropna().unique().tolist()),
        default=sorted(products["availability"].dropna().unique().tolist()),
    )
    min_rating = st.slider("Minimum rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    top_n = st.slider("Top N results", min_value=5, max_value=100, value=25, step=5)

    filtered = products.copy()
    if keyword.strip():
        q = keyword.strip().lower()
        filtered = filtered[
            filtered["product_name"].astype(str).str.lower().str.contains(q)
            | filtered["brand"].astype(str).str.lower().str.contains(q)
            | filtered["category"].astype(str).str.lower().str.contains(q)
        ]
    if selected_availability:
        filtered = filtered[filtered["availability"].isin(selected_availability)]
    filtered = filtered[filtered["rating"] >= float(min_rating)]

    engagement = interactions.groupby("product_id").size().reset_index(name="events")
    filtered = filtered.merge(engagement, on="product_id", how="left")
    filtered["events"] = filtered["events"].fillna(0).astype(int)
    filtered = filtered.sort_values(["events", "rating"], ascending=[False, False]).head(top_n)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Matched products", len(filtered))
    with c2:
        st.metric("Avg rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "0.00")
    with c3:
        st.metric("Avg price", f"${filtered['price'].mean():.2f}" if len(filtered) else "$0.00")

    st.dataframe(
        filtered[
            [
                "product_id",
                "product_name",
                "category",
                "brand",
                "price",
                "rating",
                "availability",
                "events",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_user_insights(datasets: Datasets) -> None:
    st.header("User Insights")
    st.markdown("Inspect per-user activity and behavior funnel.")
    st.image(
        "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=1400&q=80",
        caption="Understand customer behavior with user-level analytics.",
        use_container_width=True,
    )

    users = datasets.users.copy()
    interactions = datasets.interactions.copy()
    products = datasets.products.copy()

    user_ids = users["user_id"].astype(int).tolist()
    selected_user = st.selectbox("Select User ID", options=user_ids)

    user_profile = users[users["user_id"] == selected_user].iloc[0]
    st.write(
        {
            "User ID": int(user_profile["user_id"]),
            "Age": int(user_profile["age"]),
            "Gender": user_profile["gender"],
            "City": user_profile["city"],
            "Signup At": str(user_profile["signup_at"]),
        }
    )

    user_events = interactions[interactions["user_id"] == selected_user].copy()
    if len(user_events) == 0:
        st.info("No interactions found for this user.")
        return

    evt_counts = user_events["event_type"].value_counts()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Views", int(evt_counts.get("view", 0)))
    with c2:
        st.metric("Carts", int(evt_counts.get("cart", 0)))
    with c3:
        st.metric("Purchases", int(evt_counts.get("purchase", 0)))

    merged = user_events.merge(
        products[["product_id", "product_name", "category", "price", "rating"]],
        on="product_id",
        how="left",
    )
    st.subheader("Recent interactions")
    st.dataframe(
        merged.sort_values("event_at", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Category preference")
    category_counts = merged["category"].value_counts().head(10)
    st.plotly_chart(
        {
            "data": [
                {
                    "type": "bar",
                    "x": category_counts.index.astype(str).tolist(),
                    "y": category_counts.values.tolist(),
                }
            ],
            "layout": {
                "title": "Top categories for selected user",
                "height": 360,
                "margin": {"l": 10, "r": 10, "t": 40, "b": 40},
            },
        },
        use_container_width=True,
    )


def render_monitoring(datasets: Datasets) -> None:
    st.header("Monitoring")
    st.markdown(
        '<div class="ekam-card"><div class="ekam-title">System + App Monitoring</div><div class="ekam-sub">Observe backend mode, data health, logs, and runtime diagnostics.</div></div>',
        unsafe_allow_html=True,
    )
    backend = _active_data_backend()
    uptime_hint = time.strftime("%Y-%m-%d %H:%M:%S")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Backend", "SQLite" if backend == "sqlite" else "CSV")
    with c2:
        st.metric("Products", len(datasets.products))
    with c3:
        st.metric("Users", len(datasets.users))
    with c4:
        st.metric("Interactions", len(datasets.interactions))

    st.subheader("Runtime details")
    st.write(
        {
            "Python": sys.version.split()[0],
            "Platform": platform.platform(),
            "Working directory": str(Path.cwd()),
            "Log file": str(LOG_FILE),
            "Refreshed at": uptime_hint,
        }
    )

    st.subheader("Data quality checks")
    dq = {
        "products_missing_price": int(datasets.products["price"].isna().sum()) if "price" in datasets.products.columns else -1,
        "products_missing_rating": int(datasets.products["rating"].isna().sum()) if "rating" in datasets.products.columns else -1,
        "users_missing_city": int(datasets.users["city"].isna().sum()) if "city" in datasets.users.columns else -1,
        "interactions_missing_event_type": int(datasets.interactions["event_type"].isna().sum()) if "event_type" in datasets.interactions.columns else -1,
    }
    st.json(dq)

    st.subheader("Application logs (tail)")
    if st.button("Refresh log view"):
        st.rerun()
    st.code(_read_log_tail(), language="text")


def render_catalog(datasets: Datasets) -> None:
    st.header("Catalog")
    st.image(
        "https://images.unsplash.com/photo-1472851294608-062f824d29cc?auto=format&fit=crop&w=1400&q=80",
        caption="Interactive catalog with rich filters and product insights.",
        use_container_width=True,
    )

    products = datasets.products.copy()

    # Filters
    with st.sidebar:
        st.subheader("Product filters")
        categories = sorted(products["category"].dropna().unique().tolist()) if "category" in products.columns else []
        brands = sorted(products["brand"].dropna().unique().tolist()) if "brand" in products.columns else []

        selected_categories = st.multiselect(
            "Categories", options=categories, default=categories[: min(5, len(categories))]
        )
        selected_brands = st.multiselect("Brands", options=brands, default=brands[: min(5, len(brands))])

        price_min, price_max = (float(products["price"].min()), float(products["price"].max()))
        price_range = st.slider("Price range", min_value=price_min, max_value=price_max, value=(price_min, price_max))

        availability_filter = st.selectbox(
            "Availability",
            options=["All", "in_stock", "low_stock", "out_of_stock"],
            index=0,
        )

    df = products
    if selected_categories:
        df = df[df["category"].isin(selected_categories)]
    if selected_brands:
        df = df[df["brand"].isin(selected_brands)]
    df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
    if availability_filter != "All" and "availability" in df.columns:
        df = df[df["availability"] == availability_filter]

    st.subheader("Product overview")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.metric("Matching products", len(df))
    with c2:
        st.metric("Avg price", f"${df['price'].mean():.2f}" if len(df) else "$0.00")
    with c3:
        st.metric("Avg rating", f"{df['rating'].mean():.2f}" if len(df) else "0.00")

    col_left, col_right = st.columns(2, gap="large")

    # Price distribution
    with col_left:
        st.plotly_chart(
            {
                "data": [
                    {
                        "x": df["price"].dropna().values,
                        "type": "histogram",
                        "nbinsx": 30,
                    }
                ],
                "layout": {
                    "title": "Price distribution",
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                    "height": 360,
                },
            },
            use_container_width=True,
        )

    # Rating distribution
    with col_right:
        st.plotly_chart(
            {
                "data": [
                    {
                        "x": df["rating"].dropna().values,
                        "type": "histogram",
                        "nbinsx": 20,
                    }
                ],
                "layout": {
                    "title": "Rating distribution",
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                    "height": 360,
                },
            },
            use_container_width=True,
        )

    st.divider()

    st.subheader("Top products")
    sort_choice = st.selectbox("Sort by", options=["rating_desc", "price_asc", "price_desc"], index=0)
    if sort_choice == "rating_desc":
        df = df.sort_values(["rating", "price"], ascending=[False, True])
    elif sort_choice == "price_asc":
        df = df.sort_values(["price", "rating"], ascending=[True, False])
    else:
        df = df.sort_values(["price", "rating"], ascending=[False, False])

    tab_cards, tab_table = st.tabs(["🛍️ Flipkart-style Catalog", "📋 Table View"])

    with tab_cards:
        st.caption("Interactive visual shopping cards")
        card_df = df.head(24).reset_index(drop=True)
        cols = st.columns(4, gap="large")
        for i, row in card_df.iterrows():
            col = cols[i % 4]
            with col:
                card_image = (
                    row["image_url"]
                    if "image_url" in row and pd.notna(row["image_url"]) and str(row["image_url"]).strip()
                    else f"https://picsum.photos/seed/ekam-product-{int(row['product_id'])}/400/300"
                )
                st.image(
                    card_image,
                    use_container_width=True,
                )
                st.markdown(f"**{row['product_name']}**")
                st.caption(f"{row['category']} • {row['brand']}")
                st.write(f"💲 {float(row['price']):.2f}   |   ⭐ {float(row['rating']):.1f}")
                st.write(f"Stock: `{row['availability']}`")
                if st.button("View Details", key=f"view_product_{int(row['product_id'])}"):
                    st.session_state["catalog_selected_product"] = int(row["product_id"])

        selected_product_id = st.session_state.get("catalog_selected_product")
        if selected_product_id is not None:
            selected = datasets.products[datasets.products["product_id"] == selected_product_id]
            if len(selected) > 0:
                sp = selected.iloc[0]
                st.divider()
                st.subheader("Selected Product")
                d1, d2 = st.columns([1, 2], gap="large")
                with d1:
                    detail_image = (
                        sp["image_url"]
                        if "image_url" in sp and pd.notna(sp["image_url"]) and str(sp["image_url"]).strip()
                        else f"https://picsum.photos/seed/ekam-product-{int(sp['product_id'])}-detail/600/420"
                    )
                    st.image(
                        detail_image,
                        use_container_width=True,
                    )
                with d2:
                    st.markdown(f"### {sp['product_name']}")
                    st.write(
                        {
                            "Product ID": int(sp["product_id"]),
                            "Category": sp["category"],
                            "Brand": sp["brand"],
                            "Price": f"${float(sp['price']):.2f}",
                            "Rating": float(sp["rating"]),
                            "Availability": sp["availability"],
                        }
                    )

    with tab_table:
        st.dataframe(
            df[["product_id", "product_name", "category", "brand", "price", "rating", "availability"]].head(50),
            use_container_width=True,
            hide_index=True,
        )


def render_analytics(datasets: Datasets) -> None:
    st.header("Analytics")
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1400&q=80",
        caption="Track conversion, sales, category performance, and trends.",
        use_container_width=True,
    )

    interactions = datasets.interactions.copy()
    users = datasets.users.copy()
    products = datasets.products.copy()

    if "event_at" in interactions.columns and interactions["event_at"].notna().any():
        min_date = interactions["event_at"].min().date()
        max_date = interactions["event_at"].max().date()
        st.caption(f"Interaction timestamps range: {min_date} → {max_date}")

    event_types = ["view", "cart", "purchase"]
    if "event_type" not in interactions.columns:
        st.error("interactions.csv is missing `event_type`. Cannot render analytics.")
        return

    # Event counts by type
    st.subheader("Event counts")
    counts = interactions["event_type"].value_counts().reindex(event_types)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.metric("Views", _format_large_int(counts.get("view", 0)))
    with c2:
        st.metric("Carts", _format_large_int(counts.get("cart", 0)))
    with c3:
        st.metric("Purchases", _format_large_int(counts.get("purchase", 0)))

    st.divider()

    # Funnel conversion (aggregate)
    st.subheader("Conversion funnel (aggregate)")

    total_views = int(counts.get("view", 0))
    total_carts = int(counts.get("cart", 0))
    total_purchases = int(counts.get("purchase", 0))

    def safe_rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    colA, colB, colC = st.columns(3, gap="large")
    with colA:
        st.metric("View", total_views)
    with colB:
        st.metric("Cart", total_carts, delta=f"{safe_rate(total_carts, total_views)*100:.2f}% of views")
    with colC:
        st.metric(
            "Purchase",
            total_purchases,
            delta=f"{safe_rate(total_purchases, total_carts)*100:.2f}% of carts",
        )

    # Sales analytics (revenue)
    st.divider()
    st.subheader("Sales analytics")
    purchases = interactions[interactions["event_type"] == "purchase"].copy()
    purchases = purchases.merge(products[["product_id", "price", "category", "rating"]], on="product_id", how="left")
    
    if len(purchases) > 0:
        total_revenue = (purchases["quantity"] * purchases["price"]).sum()
        avg_order_value = total_revenue / len(purchases)
        avg_items_per_transaction = purchases["quantity"].mean()
        
        col_s1, col_s2, col_s3 = st.columns(3, gap="large")
        with col_s1:
            st.metric("Total Revenue (est.)", f"${total_revenue:,.2f}")
        with col_s2:
            st.metric("Avg. Order Value", f"${avg_order_value:.2f}")
        with col_s3:
            st.metric("Avg. Items/Transaction", f"{avg_items_per_transaction:.1f}")
    else:
        st.info("No purchase events found to calculate sales metrics.")

    st.divider()

    # Category distribution
    st.subheader("Category distribution")
    merged_all = interactions.merge(products[["product_id", "category"]], on="product_id", how="left")
    cat_dist = merged_all["category"].value_counts()

    if len(cat_dist) > 0:
        st.plotly_chart(
            {
                "data": [
                    {
                        "type": "pie",
                        "labels": cat_dist.index.astype(str).tolist(),
                        "values": cat_dist.values.tolist(),
                    }
                ],
                "layout": {
                    "title": "Interactions by category",
                    "height": 420,
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                },
            },
            use_container_width=True,
        )
    else:
        st.info("No category data available.")

    st.divider()

    # Rating distribution (from purchases)
    st.subheader("Rating distribution (purchased products)")
    if len(purchases) > 0:
        rating_dist = purchases["rating"].dropna()
        if len(rating_dist) > 0:
            st.plotly_chart(
                {
                    "data": [
                        {
                            "x": rating_dist.values,
                            "type": "histogram",
                            "nbinsx": 15,
                        }
                    ],
                    "layout": {
                        "title": "Rating distribution of purchased products",
                        "xaxis": {"title": "Rating"},
                        "yaxis": {"title": "Count"},
                        "height": 360,
                        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                    },
                },
                use_container_width=True,
            )
        else:
            st.info("No rating data available for purchased products.")
    else:
        st.info("No purchases to analyze product ratings.")

    st.divider()

    # Price analysis by category
    st.subheader("Average price by category")
    price_by_cat = products.groupby("category")["price"].agg(["mean", "min", "max"]).reset_index()
    price_by_cat = price_by_cat.sort_values("mean", ascending=False).head(15)

    st.plotly_chart(
        {
            "data": [
                {
                    "type": "bar",
                    "x": price_by_cat["category"].astype(str).values,
                    "y": price_by_cat["mean"].values,
                    "error_y": {
                        "type": "data",
                        "array": (price_by_cat["max"] - price_by_cat["mean"]).values,
                        "visible": True,
                    },
                }
            ],
            "layout": {
                "title": "Average price by category (top 15)",
                "xaxis": {"title": "Category"},
                "yaxis": {"title": "Price ($)"},
                "height": 420,
                "margin": {"l": 10, "r": 10, "t": 40, "b": 60},
            },
        },
        use_container_width=True,
    )

    # Top categories by purchase/view/cart
    st.divider()
    st.subheader("Top categories")

    merged = interactions.merge(products[["product_id", "category"]], on="product_id", how="left")

    # Select which event to rank by
    rank_by = st.selectbox("Rank categories by event", options=event_types, index=0)
    cat_rank = (
        merged.groupby(["category", "event_type"]).size().reset_index(name="events").query("event_type == @rank_by")
    )
    cat_rank = cat_rank.sort_values("events", ascending=False).head(12)

    st.plotly_chart(
        {
            "data": [
                {
                    "type": "bar",
                    "x": cat_rank["events"].values,
                    "y": cat_rank["category"].astype(str).values,
                    "orientation": "h",
                }
            ],
            "layout": {
                "title": f"Top categories by {rank_by}",
                "height": 420,
                "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
            },
        },
        use_container_width=True,
    )

    st.divider()

    # Top products
    st.subheader("Top products")
    rank_products_by = st.selectbox("Rank products by event", options=event_types, index=2)

    prod_rank = (
        interactions.groupby(["product_id", "event_type"]).size().reset_index(name="events").query("event_type == @rank_products_by")
    )
    prod_rank = prod_rank.merge(products[["product_id", "product_name", "category", "price", "rating"]], on="product_id", how="left")
    prod_rank = prod_rank.sort_values("events", ascending=False).head(15)

    st.dataframe(
        prod_rank[["product_id", "product_name", "category", "price", "rating", "events"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Event trend over time (monthly)
    st.subheader("Event trend over time")
    time_basis = "Monthly"

    if "event_at" in interactions.columns and interactions["event_at"].notna().any():
        interactions["event_month"] = interactions["event_at"].dt.to_period("M").astype(str)
        trend = interactions.groupby(["event_month", "event_type"]).size().reset_index(name="events")

        # Pivot for chart
        pivot = trend.pivot(index="event_month", columns="event_type", values="events").fillna(0).reset_index()

        st.plotly_chart(
            {
                "data": [
                    {"type": "scatter", "x": pivot["event_month"], "y": pivot.get(evt, 0), "mode": "lines+markers", "name": evt}
                    for evt in event_types
                ],
                "layout": {
                    "title": f"{time_basis} event counts",
                    "xaxis": {"title": "Month"},
                    "yaxis": {"title": "Events"},
                    "height": 420,
                    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                },
            },
            use_container_width=True,
        )
    else:
        st.info("No valid `event_at` timestamps found; skipping time trend chart.")

    st.divider()


def render_recommendations(datasets: Datasets) -> None:
    st.header("Recommendations")
    st.markdown(
        "Explore content-based product recommendations using TF-IDF and cosine similarity. "
        "If model artifacts are missing, train the recommendation model first."
    )

    products = datasets.products.copy()
    products["product_id"] = products["product_id"].astype(str)
    product_map = products.set_index("product_id")
    model_dir = Path(__file__).resolve().parent.parent / "saved_models"
    artifacts = [
        model_dir / "tfidf_vectorizer.joblib",
        model_dir / "item_vectors.joblib",
        model_dir / "product_ids.joblib",
    ]

    if not all(path.exists() for path in artifacts):
        st.warning(
            "Recommendation artifacts are missing. Run `python models/train_model.py --validate` from the repository root."
        )
        return

    try:
        vectorizer, item_vectors, product_ids = load_model_artifacts(model_dir)
    except Exception as exc:
        st.error("Could not load recommendation artifacts.")
        st.code(str(exc))
        return

    selected_product_id = st.selectbox(
        "Select product to view similar recommendations",
        options=product_ids,
        format_func=lambda value: product_map.loc[value, "product_name"],
    )

    recommendations = recommend_similar_products(
        selected_product_id,
        products,
        vectorizer,
        item_vectors,
        product_ids,
        top_n=10,
    )

    st.subheader("Similar products")
    st.dataframe(
        recommendations[
            ["product_id", "product_name", "category", "brand", "price", "rating", "similarity_score"]
        ].rename(columns={"similarity_score": "score"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Why these recommendations?")
    st.markdown("Expand any recommendation below to see feature-based explanations for the similarity.")

    for idx, (_, rec_row) in enumerate(recommendations.iterrows()):
        rec_product_id = str(rec_row["product_id"])
        try:
            explanation_data = explain_recommendation(
                selected_product_id,
                rec_product_id,
                products,
                vectorizer,
                item_vectors,
                product_ids,
                top_k_features=5,
            )
            with st.expander(f"📊 {rec_row['product_name']} (Score: {rec_row['similarity_score']:.3f})"):
                st.write(f"**Explanation:** {explanation_data['explanation']}")
                if explanation_data["shared_terms"]:
                    st.write(f"**Shared attributes:** {', '.join(sorted(explanation_data['shared_terms']))}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Query product features:**")
                    for term, weight in explanation_data["query_terms"][:3]:
                        st.write(f"- {term}: {weight:.3f}")
                with col2:
                    st.write("**Recommended product features:**")
                    for term, weight in explanation_data["rec_terms"][:3]:
                        st.write(f"- {term}: {weight:.3f}")
        except Exception as e:
            st.warning(f"Could not generate explanation for {rec_row['product_name']}: {str(e)}")


def render_admin() -> None:
    st.header("Admin / Data")
    st.markdown(
        "Manage datasets and inventory. Generate missing datasets, add/edit/remove products, or upload a CSV."
    )
    st.image(
        "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=1400&q=80",
        caption="Control center for data operations and inventory management.",
        use_container_width=True,
    )

    backend = _active_data_backend()
    use_sqlite = backend == "sqlite"
    db = Database(SQLITE_DB) if use_sqlite else None

    # Dataset management
    st.subheader("📊 Dataset Management")
    files = [
        (PRODUCTS_CSV, "Products"),
        (USERS_CSV, "Users"),
        (INTERACTIONS_CSV, "Interactions"),
    ]
    missing = [path for path, _ in files if not path.exists()]

    for path, label in files:
        status = "✅ Exists" if path.exists() else "❌ Missing"
        st.write(f"- **{label}**: `{path.name}` — {status}")

    if missing and not use_sqlite:
        st.warning("One or more dataset files are missing. Generate datasets to restore app functionality.")
    elif use_sqlite:
        st.success("SQLite database detected. App is using SQLite as the primary data source.")
    else:
        st.success("All required dataset files are present.")

    regenerate_label = "Generate datasets" if missing else "Regenerate datasets (CSV)"
    if st.button(regenerate_label, key="gen_datasets"):
        with st.spinner("Generating datasets..."):
            success, message = _generate_datasets()

        if success:
            load_datasets.clear()
            st.success("Datasets generated successfully.")
            st.code(message)
            st.experimental_rerun()
        else:
            st.error("Dataset generation failed.")
            st.code(message)

    st.markdown("**SQLite migration**")
    st.caption("Move CSV datasets into SQLite and make SQLite the active backend.")
    if st.button("Migrate CSV to SQLite", key="migrate_sqlite"):
        migrate_script = Path(__file__).resolve().parent.parent / "data" / "migrate_to_sqlite.py"
        try:
            result = subprocess.run(
                [sys.executable, str(migrate_script), "--no-backup", "--clear"],
                cwd=str(migrate_script.parent),
                capture_output=True,
                text=True,
                check=True,
            )
            load_datasets.clear()
            st.success("Migration to SQLite completed.")
            st.code(result.stdout.strip() or "Migration complete.")
            st.experimental_rerun()
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() if exc.stderr else exc.stdout.strip()
            st.error("SQLite migration failed.")
            st.code(error_output or str(exc))

    st.divider()

    # Inventory management
    st.subheader("📦 Inventory Management")
    st.caption(f"Inventory backend: {'SQLite' if use_sqlite else 'CSV'}")
    inv_tab1, inv_tab2, inv_tab3, inv_tab4 = st.tabs(["View", "Add", "Update", "Upload CSV"])

    with inv_tab1:
        st.write("**Current Inventory**")
        try:
            products = db.get_all_products() if use_sqlite else load_products(PRODUCTS_CSV)
            st.dataframe(
                products[["product_id", "product_name", "category", "brand", "price", "rating", "availability"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Total products: {len(products)}")
        except Exception as e:
            st.error(f"Could not load products: {str(e)}")

    with inv_tab2:
        st.write("**Add New Product**")
        with st.form("add_product_form", clear_on_submit=True):
            product_name = st.text_input("Product Name")
            category = st.text_input("Category")
            brand = st.text_input("Brand")
            price = st.number_input("Price", min_value=0.0, step=0.01)
            rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1)
            availability = st.selectbox("Availability", ["in_stock", "low_stock", "out_of_stock"])
            submit = st.form_submit_button("Add Product")

            if submit:
                errors = validate_product_data(
                    product_name=product_name,
                    category=category,
                    brand=brand,
                    price=price,
                    rating=rating,
                    availability=availability,
                )
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        if use_sqlite:
                            new_id = db.add_product(product_name, category, brand, price, rating, availability)
                        else:
                            products = load_products(PRODUCTS_CSV)
                            products, new_id = add_product(
                                products, product_name, category, brand, price, rating, availability
                            )
                            save_products(products, PRODUCTS_CSV)
                        load_datasets.clear()
                        st.success(f"✅ Product added with ID {new_id}")
                    except Exception as e:
                        st.error(f"Error adding product: {str(e)}")

    with inv_tab3:
        st.write("**Update Product**")
        try:
            products = db.get_all_products() if use_sqlite else load_products(PRODUCTS_CSV)
            product_ids = products["product_id"].astype(str).tolist()
            selected_id = st.selectbox(
                "Select product to update",
                options=product_ids,
                format_func=lambda pid: f"{pid} — {products[products['product_id']==int(pid)]['product_name'].values[0]}",
                key="update_select",
            )
            selected_product = products[products["product_id"] == int(selected_id)].iloc[0]

            with st.form("update_product_form"):
                st.write(f"**Updating: {selected_product['product_name']}**")
                new_name = st.text_input("Product Name", value=selected_product["product_name"])
                new_category = st.text_input("Category", value=selected_product["category"])
                new_brand = st.text_input("Brand", value=selected_product["brand"])
                new_price = st.number_input("Price", value=float(selected_product["price"]), min_value=0.0, step=0.01)
                new_rating = st.slider("Rating", value=float(selected_product["rating"]), min_value=1.0, max_value=5.0, step=0.1)
                new_availability = st.selectbox(
                    "Availability",
                    ["in_stock", "low_stock", "out_of_stock"],
                    index=["in_stock", "low_stock", "out_of_stock"].index(selected_product["availability"]),
                )
                submit_update = st.form_submit_button("Update Product")

                if submit_update:
                    errors = validate_product_data(
                        product_name=new_name,
                        category=new_category,
                        brand=new_brand,
                        price=new_price,
                        rating=new_rating,
                        availability=new_availability,
                    )
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        try:
                            if use_sqlite:
                                db.update_product(
                                    int(selected_id),
                                    product_name=new_name,
                                    category=new_category,
                                    brand=new_brand,
                                    price=new_price,
                                    rating=new_rating,
                                    availability=new_availability,
                                )
                            else:
                                products = update_product(
                                    products,
                                    int(selected_id),
                                    product_name=new_name,
                                    category=new_category,
                                    brand=new_brand,
                                    price=new_price,
                                    rating=new_rating,
                                    availability=new_availability,
                                )
                                save_products(products, PRODUCTS_CSV)
                            load_datasets.clear()
                            st.success(f"✅ Product {selected_id} updated")
                        except Exception as e:
                            st.error(f"Error updating product: {str(e)}")
        except Exception as e:
            st.error(f"Could not load products: {str(e)}")

        st.divider()
        st.write("**Remove Product**")
        try:
            products = db.get_all_products() if use_sqlite else load_products(PRODUCTS_CSV)
            product_ids = products["product_id"].astype(str).tolist()
            remove_id = st.selectbox(
                "Select product to remove",
                options=product_ids,
                format_func=lambda pid: f"{pid} — {products[products['product_id']==int(pid)]['product_name'].values[0]}",
                key="remove_select",
            )
            if st.button("🗑️ Remove Product", key="remove_btn"):
                try:
                    if use_sqlite:
                        db.delete_product(int(remove_id))
                    else:
                        products = remove_product(products, int(remove_id))
                        save_products(products, PRODUCTS_CSV)
                    load_datasets.clear()
                    st.success(f"✅ Product {remove_id} removed")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error removing product: {str(e)}")
        except Exception as e:
            st.error(f"Could not load products: {str(e)}")

    with inv_tab4:
        st.write("**Upload Products CSV**")
        st.markdown("Replace the current products inventory with an uploaded CSV file.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                new_products = pd.read_csv(uploaded_file)
                required_cols = {"product_id", "product_name", "category", "brand", "price", "rating", "availability"}
                if not required_cols.issubset(set(new_products.columns)):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                else:
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(new_products.head(10), use_container_width=True, hide_index=True)
                    if st.button("✅ Confirm and Replace", key="upload_confirm"):
                        if use_sqlite:
                            db.clear_table("products")
                            db.import_from_dataframe("products", new_products)
                        else:
                            save_products(new_products, PRODUCTS_CSV)
                        load_datasets.clear()
                        st.success(f"✅ Inventory updated with {len(new_products)} products")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")



def main() -> None:
    start_ts = time.time()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_dashboard_theme()
    st.title(APP_TITLE)

    options = [
        "Home",
        "Catalog",
        "Explorer",
        "Analytics",
        "User Insights",
        "Recommendations",
        "Monitoring",
        "Admin / Data",
    ]
    if "nav_page_select" not in st.session_state or st.session_state["nav_page_select"] not in options:
        st.session_state["nav_page_select"] = "Home"
    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in options:
        st.session_state["nav_page"] = st.session_state["nav_page_select"]

    page = st.sidebar.selectbox(
        "Navigation",
        options,
        key="nav_page_select",
    )
    st.session_state["nav_page"] = page
    LOGGER.info("Navigation page selected: %s", page)

    if page == "Admin / Data":
        render_admin()
        return

    datasets = _ensure_datasets_or_prompt()
    if datasets is None:
        return

    if page == "Home":
        render_home(datasets)
    elif page == "Catalog":
        render_catalog(datasets)
    elif page == "Explorer":
        render_explorer(datasets)
    elif page == "Analytics":
        render_analytics(datasets)
    elif page == "User Insights":
        render_user_insights(datasets)
    elif page == "Recommendations":
        render_recommendations(datasets)
    elif page == "Monitoring":
        render_monitoring(datasets)

    elapsed_ms = int((time.time() - start_ts) * 1000)
    LOGGER.info("Page rendered in %sms | page=%s", elapsed_ms, page)


if __name__ == "__main__":
    main()

