"""Part A: audit the raw CSV files, clean them and analyse the resulting data.

Run this after generate_data.py. All outputs go to data/clean and outputs.
The original files in data/raw are never overwritten.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

RAW_FOLDER = Path("data/raw")
CLEAN_FOLDER = Path("data/clean")
OUTPUT_FOLDER = Path("outputs")


def audit_table(name, data):
    """Show data types, missing counts and percentages for one CSV table."""
    missing_count = data.isna().sum()
    missing_percent = data.isna().mean() * 100
    report = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing_count": missing_count,
        "missing_pct": missing_percent.round(2),
    })
    print(f"\n{name}: shape={data.shape}\n{report.to_string()}")
    return report


def count_raw_problems(customers, orders):
    """Count problems first so we can explain what was cleaned later."""
    amount_as_number = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery_as_number = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    invalid_ages = (customers["age"] < 10) | (customers["age"] > 100)

    results = {
        "raw_orders": len(orders),
        "exact_duplicates": int(orders.duplicated().sum()),
        "pending": int((orders["status"] == "pending").sum()),
        "distinct_order_ids": int(orders["order_id"].nunique()),
        "invalid_amount": int(amount_as_number.isna().sum()),
        "negative_delivery": int((delivery_as_number < 0).sum()),
        "blank_delivery": int(orders["delivery_minutes"].isna().sum()),
        "blank_age": int(customers["age"].isna().sum()),
        "invalid_age": int(invalid_ages.sum()),
        "raw_cities": int(customers["city"].nunique()),
        "raw_genders": int(customers["gender"].dropna().nunique()),
        "blank_gender": int(customers["gender"].isna().sum()),
    }
    print("AUDIT:", results)
    return results


def clean_customers(customers):
    """Clean city, gender, age and signup date, in that order."""
    # The raw data has 24 city spellings representing only six cities.
    customers["city"] = customers["city"].str.strip().str.title()

    # Convert M/m and F/f into their full labels; don't guess missing gender.
    customers["gender"] = customers["gender"].str.strip().str.lower()
    customers["gender"] = customers["gender"].replace({"m": "male", "f": "female"})
    customers["gender"] = customers["gender"].fillna("unknown")

    # Invalid ages become missing, then take the median for that customer's city.
    invalid_age = (customers["age"] < 10) | (customers["age"] > 100)
    customers.loc[invalid_age, "age"] = np.nan
    city_median = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_median)

    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    return customers


def clean_orders(orders, products):
    """Keep one final order record, repair its amount and delivery duration."""
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Remove exact duplicate rows first.
    orders = orders.drop_duplicates()

    # Some orders have an earlier pending entry and a newer final entry.
    orders = orders.sort_values("order_ts", kind="stable")
    orders = orders.drop_duplicates(subset=["order_id"], keep="last")

    # 'N/A' and 'error' cannot be converted to money: mark them missing.
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

    # Add catalogue price/category so we can reconstruct amount and delivery median.
    orders = orders.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    calculated_amount = (
        orders["price_inr"] * orders["quantity"] *
        (1 - orders["discount_pct"] / 100)
    ).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    completed = orders["status"] == "completed"
    # Median must only use completed orders within the same product category.
    category_median = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
    orders.loc[completed, "delivery_minutes"] = (
        orders.loc[completed, "delivery_minutes"].fillna(category_median)
    )
    # A cancelled order was never delivered, so do not invent a duration.
    orders.loc[~completed, "delivery_minutes"] = np.nan

    orders = orders.drop(columns=["price_inr", "category"])
    return orders


def save_clean_data(customers, products, orders):
    """Write the three cleaned CSVs and their matching SQLite tables."""
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    datasets = {"customers": customers, "products": products, "orders": orders}
    for name, table in datasets.items():
        table.to_csv(CLEAN_FOLDER / f"{name}.csv", index=False)

    with sqlite3.connect("data/instamart.db") as connection:
        for name, table in datasets.items():
            table.to_sql(name, connection, if_exists="replace", index=False)


def run_sql_and_analysis(customers, products, orders):
    """Run the seven required SQL questions and cross-check using pandas."""
    with sqlite3.connect("data/instamart.db") as connection:
        sql_text = Path("sql/queries.sql").read_text(encoding="utf-8")
        queries = [query.strip() for query in sql_text.split(";") if query.strip()]
        for number, query in enumerate(queries, start=1):
            result = pd.read_sql_query(query, connection)
            result.to_csv(OUTPUT_FOLDER / f"sql_q{number}.csv", index=False)
            print(f"Q{number} rows={len(result)}")
        sql_revenue_by_city_category = pd.read_sql_query(queries[1], connection)

    completed = orders[orders["status"] == "completed"]
    with_city = completed.merge(customers[["customer_id", "city"]], on="customer_id")
    with_category = with_city.merge(products[["product_id", "category"]], on="product_id")

    pivot = pd.pivot_table(
        with_category,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
    )
    pivot.to_csv(OUTPUT_FOLDER / "city_category_pivot.csv")
    city_summary = with_city.groupby("city").agg(
        total=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    city_summary.to_csv(OUTPUT_FOLDER / "city_summary.csv")
    assert np.isclose(pivot.loc["All", "All"], sql_revenue_by_city_category["revenue"].sum())

    amounts = completed["order_amount"]
    q1, q3 = amounts.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    statistics = {
        "mean": float(amounts.mean()),
        "median": float(amounts.median()),
        "skew": float(amounts.skew()),
        "q1": float(q1),
        "q3": float(q3),
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "lower_outliers": int((amounts < lower_fence).sum()),
        "upper_outliers": int((amounts > upper_fence).sum()),
    }
    correlation = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    pd.DataFrame([statistics]).to_csv(OUTPUT_FOLDER / "order_stats.csv", index=False)
    correlation.to_csv(OUTPUT_FOLDER / "order_correlations.csv")
    print("STATS:", statistics, "\nCORR:\n", correlation)


def main():
    CLEAN_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(RAW_FOLDER / "customers.csv")
    products = pd.read_csv(RAW_FOLDER / "products.csv")
    orders = pd.read_csv(RAW_FOLDER / "orders.csv")
    for name, table in [("customers", customers), ("products", products), ("orders", orders)]:
        audit_table(name, table)

    audit_results = count_raw_problems(customers, orders)
    pd.DataFrame([audit_results]).to_csv(OUTPUT_FOLDER / "raw_audit.csv", index=False)

    customers = clean_customers(customers)
    orders = clean_orders(orders, products)
    save_clean_data(customers, products, orders)
    run_sql_and_analysis(customers, products, orders)
    print("Part A created clean files and SQLite database.")


if __name__ == "__main__":
    main()
