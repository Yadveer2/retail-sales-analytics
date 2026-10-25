"""
05_rfm_segmentation.py

RFM stands for Recency, Frequency, Monetary.
It is a simple way to group customers based on their buying behaviour:

  Recency   — how many days since their last order (lower = better)
  Frequency — how many orders they placed (higher = better)
  Monetary  — how much total money they spent (higher = better)

We use KMeans clustering to automatically group customers into 4 segments:
  VIP, Loyal, Regular, At-Risk
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- paths ---
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data" / "master.csv"
OUTPUT_DIR = ROOT / "docs" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Run 01_data_cleaning.ipynb first. File not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE, parse_dates=["order_purchase_timestamp"])


# ── 2. Build RFM table ────────────────────────────────────────────────────────

def build_rfm(df):
    """
    For each customer, calculate:
      - Recency   : days since their last purchase
      - Frequency : number of unique orders
      - Monetary  : total revenue generated
    """
    # use the day after the last order as our reference point
    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            Recency   = ("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
            Frequency = ("order_id", "nunique"),
            Monetary  = ("revenue", "sum"),
        )
        .reset_index()
    )
    return rfm


# ── 3. Prepare features for clustering ───────────────────────────────────────

def prepare_features(rfm):
    """
    KMeans works best when all features are on the same scale.
    We:
      1. Cap very high spenders at the 99th percentile (so outliers don't skew the clusters)
      2. Apply log transform to Frequency and Monetary (they are skewed)
      3. Standardise everything to have mean=0, std=1
    """
    # cap outliers
    cap = rfm["Monetary"].quantile(0.99)
    rfm["Monetary_capped"] = rfm["Monetary"].clip(upper=cap)

    features = rfm[["Recency", "Frequency", "Monetary_capped"]].copy()

    # log transform to reduce skew
    features["Frequency"]       = np.log1p(features["Frequency"])
    features["Monetary_capped"] = np.log1p(features["Monetary_capped"])

    # standardise so all features have equal weight
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# ── 4. Cluster and label segments ────────────────────────────────────────────

def assign_segments(rfm):
    """
    Run KMeans with 4 clusters, then label each cluster with a business name.
    The cluster with the highest average spend = VIP, lowest recency (longest ago) = At-Risk.
    """
    scaled = prepare_features(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # summarise each cluster so we can label them
    summary = (
        rfm.groupby("Cluster")
        .agg(
            Recency   = ("Recency",            "mean"),
            Frequency = ("Frequency",          "mean"),
            Monetary  = ("Monetary",           "mean"),
            Count     = ("customer_unique_id", "count"),
        )
        .round(2)
    )

    # rank clusters by monetary value (highest spend = VIP, lowest = Regular)
    by_money   = summary["Monetary"].sort_values(ascending=False).index.tolist()
    # the cluster with the worst recency (bought longest ago) becomes At-Risk
    worst_recency = summary["Recency"].idxmax()

    label_map = {}
    labels = ["VIP", "Loyal", "Regular", "At-Risk"]
    for rank, cluster_id in enumerate(by_money):
        label_map[cluster_id] = labels[rank]

    # override: the cluster that bought longest ago is always At-Risk
    label_map[worst_recency] = "At-Risk"

    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm, summary


# ── 5. Charts ─────────────────────────────────────────────────────────────────

COLORS = {
    "VIP":     "#2E86AB",
    "Loyal":   "#A23B72",
    "Regular": "#F18F01",
    "At-Risk": "#C73E1D",
}

def plot_segments(rfm):
    # Chart 1 — how many customers are in each segment?
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = rfm["Segment"].value_counts()
    counts.plot(kind="bar", ax=ax, color=[COLORS[s] for s in counts.index], edgecolor="black")
    ax.set_title("Customer Segment Distribution")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    # add count labels on top of each bar
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v:,}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_distribution.png", dpi=150)
    plt.close()

    # Chart 2 — how much revenue does each segment bring in?
    fig, ax = plt.subplots(figsize=(9, 5))
    rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    rev.plot(kind="bar", ax=ax, color=[COLORS[s] for s in rev.index], edgecolor="black")
    ax.set_title("Total Revenue by Segment")
    ax.set_ylabel("Revenue (BRL)")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_revenue.png", dpi=150)
    plt.close()

    # Chart 3 — scatter plot of Frequency vs Monetary, coloured by segment
    fig, ax = plt.subplots(figsize=(10, 6))
    for segment, color in COLORS.items():
        subset = rfm[rfm["Segment"] == segment]
        ax.scatter(subset["Frequency"], subset["Monetary"],
                   c=color, label=segment, alpha=0.4, s=15)
    ax.set_xlabel("Frequency (number of orders)")
    ax.set_ylabel("Monetary (total spend, BRL)")
    ax.set_title("Customer Segments — Frequency vs Spend")
    ax.set_xlim(0, 15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rfm_scatter.png", dpi=150)
    plt.close()


# ── 6. Print summary to console ───────────────────────────────────────────────

def print_insights(rfm):
    total_customers = len(rfm)
    total_revenue   = rfm["Monetary"].sum()

    print("\n--- Segment Summary ---")
    for segment in ["VIP", "Loyal", "Regular", "At-Risk"]:
        if segment not in rfm["Segment"].values:
            continue
        subset = rfm[rfm["Segment"] == segment]
        pct_cust = 100 * len(subset) / total_customers
        pct_rev  = 100 * subset["Monetary"].sum() / total_revenue
        print(f"\n{segment}")
        print(f"  Customers    : {len(subset):,}  ({pct_cust:.1f}%)")
        print(f"  Revenue share: {pct_rev:.1f}%")
        print(f"  Avg spend    : Rs {subset['Monetary'].mean():,.2f}")
        print(f"  Avg recency  : {subset['Recency'].mean():.0f} days since last order")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    rfm = build_rfm(df)
    print(f"Built RFM table for {len(rfm):,} customers")

    rfm, summary = assign_segments(rfm)

    # save results
    rfm.to_csv(OUTPUT_DIR / "rfm_segments.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")
    print("Saved rfm_segments.csv and cluster_summary.csv")

    plot_segments(rfm)
    print("Saved charts to", OUTPUT_DIR)

    print_insights(rfm)
    print("\nDone.")
