"""SQLite database module for EKAM with schema, migrations, and CRUD operations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd


class Database:
    """SQLite database manager for EKAM."""

    def __init__(self, db_path: Path | str = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. Defaults to `data/ekam.db`
        """
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "ekam.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create all tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Products table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    brand TEXT NOT NULL,
                    price REAL NOT NULL,
                    rating REAL NOT NULL,
                    availability TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    city TEXT NOT NULL,
                    signup_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    event_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            """)

            # Create indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_product ON interactions(product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_event_type ON interactions(event_type)")

            conn.commit()

    # ===== Products CRUD =====

    def get_all_products(self) -> pd.DataFrame:
        """Get all products as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql("SELECT * FROM products ORDER BY product_id", conn)

    def get_product(self, product_id: int) -> dict | None:
        """Get a single product by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_product(
        self,
        product_name: str,
        category: str,
        brand: str,
        price: float,
        rating: float,
        availability: str,
    ) -> int:
        """Add a new product. Returns the product_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO products (product_name, category, brand, price, rating, availability)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (product_name, category, brand, price, rating, availability),
            )
            conn.commit()
            return cursor.lastrowid

    def update_product(self, product_id: int, **kwargs) -> bool:
        """Update product fields. Returns True if product existed."""
        if not self.get_product(product_id):
            return False

        allowed_fields = {"product_name", "category", "brand", "price", "rating", "availability"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return True

        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [product_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE products SET {set_clause} WHERE product_id = ?", values)
            conn.commit()
        return True

    def delete_product(self, product_id: int) -> bool:
        """Delete a product. Returns True if product existed."""
        if not self.get_product(product_id):
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
            conn.commit()
        return True

    # ===== Users CRUD =====

    def get_all_users(self) -> pd.DataFrame:
        """Get all users as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql("SELECT * FROM users ORDER BY user_id", conn)

    def get_user(self, user_id: int) -> dict | None:
        """Get a single user by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_user(self, user_id: int, age: int, gender: str, city: str, signup_at: str) -> bool:
        """Add a new user. Returns True if successful."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO users (user_id, age, gender, city, signup_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, age, gender, city, signup_at),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user fields. Returns True if user existed."""
        if not self.get_user(user_id):
            return False

        allowed_fields = {"age", "gender", "city"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return True

        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
            conn.commit()
        return True

    # ===== Interactions CRUD =====

    def get_all_interactions(self) -> pd.DataFrame:
        """Get all interactions as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql("SELECT * FROM interactions ORDER BY interaction_id", conn)

    def get_interactions_by_user(self, user_id: int) -> pd.DataFrame:
        """Get all interactions for a user."""
        with self.get_connection() as conn:
            return pd.read_sql(
                "SELECT * FROM interactions WHERE user_id = ? ORDER BY event_at",
                conn,
                params=(user_id,),
            )

    def get_interactions_by_product(self, product_id: int) -> pd.DataFrame:
        """Get all interactions for a product."""
        with self.get_connection() as conn:
            return pd.read_sql(
                "SELECT * FROM interactions WHERE product_id = ? ORDER BY event_at",
                conn,
                params=(product_id,),
            )

    def add_interaction(
        self, user_id: int, product_id: int, event_type: str, quantity: int, event_at: str
    ) -> int:
        """Add a new interaction. Returns the interaction_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO interactions (user_id, product_id, event_type, quantity, event_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, product_id, event_type, quantity, event_at),
            )
            conn.commit()
            return cursor.lastrowid

    # ===== Bulk Operations =====

    def import_from_dataframe(self, table: str, df: pd.DataFrame) -> int:
        """Import data from DataFrame into table. Returns number of rows inserted."""
        with self.get_connection() as conn:
            df.to_sql(table, conn, if_exists="append", index=False)
            conn.commit()
            return len(df)

    def export_to_dataframe(self, table: str) -> pd.DataFrame:
        """Export entire table as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    def clear_table(self, table: str) -> int:
        """Clear all rows from a table. Returns number of rows deleted."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table}")
            conn.commit()
            return cursor.rowcount

    # ===== Analytics =====

    def get_event_counts(self) -> dict:
        """Get counts of each event type."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT event_type, COUNT(*) as count FROM interactions GROUP BY event_type"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_top_products(self, event_type: str = None, limit: int = 10) -> pd.DataFrame:
        """Get top products by interaction count."""
        query = """
            SELECT p.*, COUNT(i.interaction_id) as interaction_count
            FROM products p
            LEFT JOIN interactions i ON p.product_id = i.product_id
        """
        if event_type:
            query += f" WHERE i.event_type = '{event_type}'"
        query += " GROUP BY p.product_id ORDER BY interaction_count DESC LIMIT ?"

        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=(limit,))

    def get_top_categories(self, event_type: str = None, limit: int = 10) -> pd.DataFrame:
        """Get top categories by interaction count."""
        query = """
            SELECT p.category, COUNT(i.interaction_id) as interaction_count
            FROM products p
            LEFT JOIN interactions i ON p.product_id = i.product_id
        """
        if event_type:
            query += f" WHERE i.event_type = '{event_type}'"
        query += " GROUP BY p.category ORDER BY interaction_count DESC LIMIT ?"

        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=(limit,))
