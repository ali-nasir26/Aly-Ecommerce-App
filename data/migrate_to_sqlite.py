"""Migration script to move data from CSV files to SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.database import Database


def migrate_csv_to_sqlite(
    data_dir: Path = None,
    db_path: Path = None,
    backup_csv: bool = True,
    clear_existing: bool = False,
) -> None:
    """Migrate data from CSV files to SQLite database.
    
    Args:
        data_dir: Directory containing CSV files (defaults to `data/`)
        db_path: Path to SQLite database (defaults to `data/ekam.db`)
        backup_csv: If True, rename CSV files with .backup suffix
        clear_existing: If True, clear existing data in database before importing
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent / "data" / "ekam.db"

    data_dir = Path(data_dir)
    db_path = Path(db_path)

    products_csv = data_dir / "products.csv"
    users_csv = data_dir / "users.csv"
    interactions_csv = data_dir / "interactions.csv"

    # Check that CSV files exist
    missing = []
    if not products_csv.exists():
        missing.append(str(products_csv))
    if not users_csv.exists():
        missing.append(str(users_csv))
    if not interactions_csv.exists():
        missing.append(str(interactions_csv))

    if missing:
        raise FileNotFoundError(f"Missing CSV files: {', '.join(missing)}")

    # Initialize database
    db = Database(db_path)
    db.initialize()

    # Clear existing data if requested
    if clear_existing:
        print("Clearing existing database tables...")
        db.clear_table("interactions")
        db.clear_table("products")
        db.clear_table("users")

    # Load CSV files
    print(f"Loading data from {data_dir}...")
    products_df = pd.read_csv(products_csv)
    users_df = pd.read_csv(users_csv)
    interactions_df = pd.read_csv(interactions_csv)

    # Import into database
    print(f"Importing {len(products_df)} products...")
    db.import_from_dataframe("products", products_df)

    print(f"Importing {len(users_df)} users...")
    db.import_from_dataframe("users", users_df)

    print(f"Importing {len(interactions_df)} interactions...")
    db.import_from_dataframe("interactions", interactions_df)

    print(f"✅ Migration complete. Database: {db_path}")

    # Backup CSV files if requested
    if backup_csv:
        print("Backing up CSV files...")
        products_csv.rename(products_csv.with_suffix(".csv.backup"))
        users_csv.rename(users_csv.with_suffix(".csv.backup"))
        interactions_csv.rename(interactions_csv.with_suffix(".csv.backup"))
        print("CSV files backed up with .backup suffix")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate CSV data to SQLite database.")
    parser.add_argument("--data-dir", type=Path, help="Path to data directory containing CSV files.")
    parser.add_argument("--db-path", type=Path, help="Path to SQLite database file.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup CSV files.")
    parser.add_argument("--clear", action="store_true", help="Clear existing database tables before importing.")
    args = parser.parse_args()

    migrate_csv_to_sqlite(
        data_dir=args.data_dir,
        db_path=args.db_path,
        backup_csv=not args.no_backup,
        clear_existing=args.clear,
    )


if __name__ == "__main__":
    main()
