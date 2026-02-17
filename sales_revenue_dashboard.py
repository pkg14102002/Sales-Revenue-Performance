"""
=============================================================
 Sales & Revenue Performance Intelligence Dashboard
 Author  : Prince Kumar Gupta
 Role    : Data Analyst
 Tools   : Python, Pandas, NumPy, Matplotlib, SQLite
=============================================================
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = "sales_dashboard_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Generate 3 Years of Sales Data ────────────────────────
def generate_sales_data() -> pd.DataFrame:
    np.random.seed(99)

    regions  = ["North", "South", "East", "West", "Central"]
    products = {
        "Product A": {"base": 45000, "margin": 0.35},
        "Product B": {"base": 30000, "margin": 0.42},
        "Product C": {"base": 55000, "margin": 0.28},
        "Product D": {"base": 25000, "margin": 0.50},
        "Product E": {"base": 65000, "margin": 0.22},
    }
    channels   = ["Direct", "Online", "Partner", "Retail"]
    reps       = [f"Rep_{i}" for i in range(1, 21)]

    rows = []
    start = datetime(2022, 1, 1)
    end   = datetime(2024, 12, 31)
    current = start

    while current <= end:
        n_deals = np.random.randint(15, 45)
        for _ in range(n_deals):
            product  = np.random.choice(list(products.keys()))
            info     = products[product]
            # Add seasonality
            month_factor = 1.0 + 0.3 * np.sin((current.month - 3) * np.pi / 6)
            year_factor  = 1.0 + 0.12 * (current.year - 2022)  # 12% YoY growth
            revenue  = round(info["base"] * month_factor * year_factor * np.random.uniform(0.7, 1.4), 2)
            target   = round(revenue * np.random.uniform(0.85, 1.20), 2)
            units    = np.random.randint(1, 50)
            rows.append({
                "date"         : current.strftime("%Y-%m-%d"),
                "year"         : current.year,
                "month"        : current.month,
                "quarter"      : f"Q{(current.month-1)//3+1}",
                "region"       : np.random.choice(regions),
                "product"      : product,
                "channel"      : np.random.choice(channels, p=[0.35,0.30,0.20,0.15]),
                "sales_rep"    : np.random.choice(reps),
                "revenue"      : revenue,
                "target"       : target,
                "units_sold"   : units,
                "margin_pct"   : round(info["margin"] * np.random.uniform(0.8, 1.2), 3),
                "discount_pct" : round(np.random.uniform(0, 0.25), 3),
                "cost"         : round(revenue * (1 - info["margin"]), 2),
            })
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df["gross_profit"] = (df["revenue"] * df["margin_pct"]).round(2)
    df["achievement"]  = (df["revenue"] / df["target"] * 100).round(1)
    df["date"]         = pd.to_datetime(df["date"])

    log.info(f"✅ Generated {len(df):,} sales records across 3 years.")
    return df

# ── Step 2: Compute KPIs ──────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {}

    kpis["total_revenue"]    = df["revenue"].sum()
    kpis["total_profit"]     = df["gross_profit"].sum()
    kpis["overall_margin"]   = (kpis["total_profit"] / kpis["total_revenue"] * 100)
    kpis["total_units"]      = df["units_sold"].sum()
    kpis["avg_deal_size"]    = df["revenue"].mean()
    kpis["avg_achievement"]  = df["achievement"].mean()
    kpis["total_target"]     = df["target"].sum()

    # YoY comparison
    kpis["by_year"] = df.groupby("year").agg(
        Revenue     =("revenue",      "sum"),
        Profit      =("gross_profit", "sum"),
        Units       =("units_sold",   "sum"),
        Avg_Margin  =("margin_pct",   "mean"),
        Achievement =("achievement",  "mean"),
        Deals       =("revenue",      "count"),
    ).reset_index()
    kpis["by_year"]["Revenue_M"]    = (kpis["by_year"]["Revenue"] / 1e6).round(2)
    kpis["by_year"]["YoY_Growth"]   = kpis["by_year"]["Revenue"].pct_change() * 100

    # Monthly trend
    kpis["monthly"] = df.groupby(["year","month"]).agg(
        Revenue=("revenue","sum"),
        Target =("target", "sum"),
        Profit =("gross_profit","sum"),
    ).reset_index()
    kpis["monthly"]["period"] = pd.to_datetime(
        kpis["monthly"]["year"].astype(str) + "-" + kpis["monthly"]["month"].astype(str).str.zfill(2)
    )

    # By product
    kpis["by_product"] = df.groupby("product").agg(
        Revenue    =("revenue","sum"),
        Units      =("units_sold","sum"),
        Avg_Margin =("margin_pct","mean"),
        Deals      =("revenue","count"),
    ).reset_index().sort_values("Revenue", ascending=False)

    # By region
    kpis["by_region"] = df.groupby("region").agg(
        Revenue =("revenue","sum"),
        Profit  =("gross_profit","sum"),
        Units   =("units_sold","sum"),
        Deals   =("revenue","count"),
    ).reset_index().sort_values("Revenue", ascending=False)
    kpis["by_region"]["Margin%"] = (kpis["by_region"]["Profit"] / kpis["by_region"]["Revenue"] * 100).round(1)

    # By channel
    kpis["by_channel"] = df.groupby("channel")["revenue"].sum().reset_index()

    # Top 10 sales reps
    kpis["top_reps"] = (
        df.groupby("sales_rep").agg(Revenue=("revenue","sum"), Deals=("revenue","count"))
        .reset_index().sort_values("Revenue", ascending=False).head(10)
    )

    # Seasonal patterns
    kpis["by_month_avg"] = df.groupby("month")["revenue"].mean()
    kpis["by_quarter"]   = df.groupby(["year","quarter"]).agg(Revenue=("revenue","sum")).reset_index()

    log.info(f"✅ KPIs computed. Total Revenue: Rs.{kpis['total_revenue']/1e6:.1f}M")
    return kpis

# ── Step 3: Build Dashboard ───────────────────────────────────────
def build_dashboard(df: pd.DataFrame, kpis: dict):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#F0F4F8")
    gs  = GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    NAVY = "#0D2137"; BLUE = "#1565C0"; GREEN = "#2E7D32"
    RED  = "#C62828"; AMBER= "#E65100"; TEAL  = "#00695C"
    LBLUE= "#E3F2FD"; GRAY = "#546E7A"

    fig.suptitle(
        "Sales & Revenue Performance Intelligence Dashboard\n"
        "3-Year Analysis (2022–2024) | Prince Kumar Gupta | Data Analyst",
        fontsize=14, fontweight="bold", color=NAVY, y=0.99
    )

    fmt_cr = lambda x: f"Rs.{x/1e7:.1f}Cr"

    # ── Row 0: KPI Cards ────────────────────────────────────────
    kpi_data = [
        ("Total Revenue",  fmt_cr(kpis["total_revenue"]),  NAVY),
        ("Gross Profit",   fmt_cr(kpis["total_profit"]),   GREEN),
        ("Profit Margin",  f"{kpis['overall_margin']:.1f}%", TEAL),
    ]
    for i, (title, val, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color)
        ax.axis("off")
        ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=10,
                color="#B0BEC5", transform=ax.transAxes)
        ax.text(0.5, 0.35, val, ha="center", va="center", fontsize=18,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.set_title("")

    # ── Row 1: Revenue Trend (Monthly) ──────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    monthly = kpis["monthly"].sort_values("period")
    ax1.fill_between(monthly["period"], monthly["Revenue"]/1e5,
                     alpha=0.3, color=BLUE)
    ax1.plot(monthly["period"], monthly["Revenue"]/1e5,
             color=BLUE, linewidth=2, label="Actual Revenue")
    ax1.plot(monthly["period"], monthly["Target"]/1e5,
             color=RED, linewidth=1.5, linestyle="--", label="Target")
    ax1.set_facecolor(LBLUE)
    ax1.set_title("Monthly Revenue vs Target (3-Year Trend)", fontsize=11,
                  fontweight="bold", color=NAVY)
    ax1.set_ylabel("Revenue (Rs. Lakhs)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"Rs.{x:.0f}L"))
    ax1.legend(fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── Row 2: By Product (bar) ──────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    prod = kpis["by_product"]
    colors_p = [BLUE, TEAL, GREEN, AMBER, RED]
    bars = ax2.bar(prod["product"], prod["Revenue"]/1e5,
                   color=colors_p, edgecolor="white")
    for bar, val in zip(bars, prod["Revenue"]/1e5):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"Rs.{val:.0f}L", ha="center", fontsize=7.5, fontweight="bold")
    ax2.set_facecolor(LBLUE)
    ax2.set_title("Revenue by Product", fontsize=10, fontweight="bold", color=NAVY)
    ax2.set_ylabel("Revenue (Lakhs)", fontsize=8)
    ax2.tick_params(labelsize=8, axis="x", rotation=20)

    # ── Row 2: By Region ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    reg = kpis["by_region"]
    ax3.barh(reg["region"], reg["Revenue"]/1e5,
             color=[NAVY, BLUE, TEAL, GREEN, AMBER], edgecolor="white")
    for i, (rev, margin) in enumerate(zip(reg["Revenue"]/1e5, reg["Margin%"])):
        ax3.text(rev+0.3, i, f"Rs.{rev:.0f}L | {margin}% margin",
                 va="center", fontsize=7.5)
    ax3.set_facecolor(LBLUE)
    ax3.set_title("Revenue & Margin by Region", fontsize=10, fontweight="bold", color=NAVY)
    ax3.tick_params(labelsize=8)

    # ── Row 2: Channel Mix ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    ch = kpis["by_channel"]
    wedges, texts, autotexts = ax4.pie(
        ch["revenue"], labels=ch["channel"], autopct="%1.1f%%",
        colors=[BLUE, TEAL, GREEN, AMBER], startangle=90, pctdistance=0.75
    )
    for at in autotexts: at.set_fontsize(8)
    ax4.set_title("Revenue by Sales Channel", fontsize=10, fontweight="bold", color=NAVY)

    # ── Row 3: YoY Growth ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    yr  = kpis["by_year"]
    x   = range(len(yr))
    ax5_twin = ax5.twinx()
    ax5.bar(x, yr["Revenue_M"], color=BLUE, alpha=0.8, label="Revenue (M)")
    ax5_twin.plot(x, yr["YoY_Growth"].fillna(0), "o-", color=AMBER,
                  linewidth=2, markersize=8, label="YoY Growth %")
    ax5.set_xticks(x)
    ax5.set_xticklabels(yr["year"], fontsize=9)
    ax5.set_facecolor(LBLUE)
    ax5.set_title("Year-over-Year Revenue Growth", fontsize=10, fontweight="bold", color=NAVY)
    ax5.set_ylabel("Revenue (Rs. Millions)", fontsize=8)
    ax5_twin.set_ylabel("YoY Growth %", fontsize=8, color=AMBER)
    ax5.tick_params(labelsize=8)

    # ── Row 3: Seasonal Pattern ──────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal = kpis["by_month_avg"]
    bar_colors = [RED if v >= seasonal.quantile(0.75) else (GREEN if v <= seasonal.quantile(0.25) else BLUE)
                  for v in seasonal.values]
    ax6.bar(months, seasonal.values/1e4, color=bar_colors, edgecolor="white")
    ax6.set_facecolor(LBLUE)
    ax6.set_title("Seasonal Revenue Patterns\n(Red=Peak | Green=Low)", fontsize=10,
                  fontweight="bold", color=NAVY)
    ax6.set_ylabel("Avg Revenue (Rs. 10K)", fontsize=8)
    ax6.tick_params(labelsize=7, axis="x", rotation=30)

    # ── Row 3: Top Reps ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 2])
    top = kpis["top_reps"].head(8).sort_values("Revenue")
    ax7.barh(top["sales_rep"], top["Revenue"]/1e5,
             color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top))), edgecolor="white")
    ax7.set_facecolor(LBLUE)
    ax7.set_title("Top 8 Sales Representatives", fontsize=10, fontweight="bold", color=NAVY)
    ax7.set_xlabel("Revenue (Lakhs)", fontsize=8)
    ax7.tick_params(labelsize=8)

    path = os.path.join(OUTPUT_DIR, "sales_revenue_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"✅ Dashboard saved → {path}")
    return path

# ── Step 4: Save Summaries ────────────────────────────────────────
def save_summaries(kpis: dict):
    kpis["by_product"].to_csv(os.path.join(OUTPUT_DIR, "product_performance.csv"), index=False)
    kpis["by_region"].to_csv(os.path.join(OUTPUT_DIR,  "region_performance.csv"),  index=False)
    kpis["by_year"].to_csv(os.path.join(OUTPUT_DIR,    "yearly_growth.csv"),       index=False)
    kpis["top_reps"].to_csv(os.path.join(OUTPUT_DIR,   "top_sales_reps.csv"),      index=False)
    log.info(f"✅ Summary CSVs saved to {OUTPUT_DIR}/")

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Starting Sales Revenue Analysis")
    df   = generate_sales_data()
    kpis = compute_kpis(df)

    log.info(f"📊 Total Revenue  : Rs.{kpis['total_revenue']/1e6:.1f}M")
    log.info(f"📊 Gross Profit   : Rs.{kpis['total_profit']/1e6:.1f}M")
    log.info(f"📊 Profit Margin  : {kpis['overall_margin']:.1f}%")
    log.info(f"📊 Avg Achievement: {kpis['avg_achievement']:.1f}%")

    build_dashboard(df, kpis)
    save_summaries(kpis)
    log.info("🏁 Sales revenue analysis complete.")
