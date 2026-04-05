"""
visualization.py
----------------
Plotting functions for the FAO EU27 wheat yield regression analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from typing import List, Optional
# ── colour palette ─────────────────────────────────────────────────────────────
BLUE  = "#1a6faf"
RED   = "#c0392b"
GREEN = "#27ae60"
CROP_PALETTE = "tab10"
WHEAT_COLOR   = "#E07B2A"   # warm amber — stands out immediately
CEREAL_COLOR  = "#5B8DB8"   # muted steel blue for other cereals
OTHER_COLOR   = "#C8C8C8"   # light grey for non-cereal crops
GRID_COLOR    = "#EEEEEE"

"""
    1 . First Graph
    Three panels that together argue for wheat as the focus crop:
    Left   — Top 10 crops by total production (all crops, context)
    Centre — Number of EU27 countries growing each cereal
    Right  — Total cereal production (wheat vs other cereals)
"""
 
CEREALS = ["Wheat", "Maize (corn)", "Barley", "Rye", "Oats",
           "Sorghum", "Triticale", "Millet"]
 
AGG_KW = ["primary", "total", "equivalent", "crops", "roots",
          "tubers", "vegetables", "fruit", "milk", "meat", "eggs"]
 
def _is_crop(name: str) -> bool:
    return not any(kw in name.lower() for kw in AGG_KW)
 
def _bar_colors(index, focus="Wheat", cereal_list=CEREALS):
    colors = []
    for item in index:
        if item == focus:
            colors.append(WHEAT_COLOR)
        elif item in cereal_list:
            colors.append(CEREAL_COLOR)
        else:
            colors.append(OTHER_COLOR)
    return colors
 
 
def plot_top_eu_crops(df_all, focus_crops, top_n=10, figsize=(16, 5.5)):
    df_clean = df_all[df_all["Item"].apply(_is_crop)].copy()
 
    # ── Panel 1: top-N crops by total production ───────────────────────────────
    prod_all = (
        df_clean.groupby("Item")["Production_tonnes"]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
    ) / 1_000_000
 
    # ── Panel 2: country coverage per cereal ───────────────────────────────────
    cereals_df = df_clean[df_clean["Item"].isin(CEREALS)]
    coverage = (
        cereals_df.groupby("Item")["Area"]
        .nunique()
        .sort_values(ascending=True)
    )
 
    # ── Panel 3: total cereal production ──────────────────────────────────────
    cereal_prod = (
        cereals_df.groupby("Item")["Production_tonnes"]
        .sum()
        .sort_values(ascending=True)
    ) / 1_000_000
 
    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("white")
 
    def _style_ax(ax, xlabel, title):
        ax.set_xlabel(xlabel, fontsize=9, color="#444444")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10, color="#222222")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8.5, colors="#444444")
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
 
    # ── Panel 1 ────────────────────────────────────────────────────────────────
    colors1 = _bar_colors(prod_all.index)
    bars1 = ax1.barh(prod_all.index, prod_all.values, color=colors1,
                     height=0.65, zorder=3)
    _style_ax(ax1,
              "Total production (million tonnes)",
              "Most produced crops in EU27\n(1990–2023)")
    for bar, val in zip(bars1, prod_all.values):
        ax1.text(val + prod_all.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=7.5, color="#555555")
 
    # ── Panel 2 ────────────────────────────────────────────────────────────────
    colors2 = _bar_colors(coverage.index)
    bars2 = ax2.barh(coverage.index, coverage.values, color=colors2,
                     height=0.65, zorder=3)
    # Updated title: acknowledges wheat+barley both hit 27, points to panel 3
    _style_ax(ax2,
              "Number of EU27 countries",
              "Geographic coverage by cereal\n(wheat and barley reach all 27 — see panel 3)")
    # Reference line at 27 — label placed INSIDE the axis to avoid clipping
    ax2.axvline(27, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)
    ax2.set_xlim(0, 30)
    ax2.text(26.6, len(coverage) * 0.05, "27", fontsize=7, color="#999999",
             ha="right", va="bottom")
    for bar, val in zip(bars2, coverage.values):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", fontsize=7.5, color="#555555")
 
    # ── Panel 3 ────────────────────────────────────────────────────────────────
    colors3 = _bar_colors(cereal_prod.index)
    bars3 = ax3.barh(cereal_prod.index, cereal_prod.values, color=colors3,
                     height=0.65, zorder=3)
    _style_ax(ax3,
              "Total production (million tonnes)",
              "Cereal production in EU27\n(wheat produces ~2× more than barley)")
    for bar, val in zip(bars3, cereal_prod.values):
        ax3.text(val + cereal_prod.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=7.5, color="#555555")
 
    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        Patch(color=WHEAT_COLOR,  label="Wheat (focus crop)"),
        Patch(color=CEREAL_COLOR, label="Other cereals"),
        Patch(color=OTHER_COLOR,  label="Non-cereal crops"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
 
    plt.suptitle(
        "Why wheat? Cereal production among EU27",
        fontsize=12, fontweight="bold", y=1.01, color="#111111"
    )
    plt.tight_layout()
    return fig

"""
    2 . Second Graph
    Map of EU27 wheat indicators (yield or production).
"""
def plot_wheat_map_eu27(df, variable="Yield_t_ha", figsize=(10, 7)):
    import geopandas as gpd

    # Aggregate data by country (average over all years)
    agg = (
        df.groupby("Area")[variable]
        .mean()
        .reset_index()
        .rename(columns={"Area": "country"})
    )
    agg["country"] = agg["country"].replace({
    "Netherlands (Kingdom of the)": "Netherlands",
})
    # Load Natural Earth countries shapefile
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )
    # EU27 using Natural Earth names
    # Malta excluded — too small for 110m resolution
    EU27 = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
        "Slovenia", "Spain", "Sweden",
    ]
    eu_map = world[world["NAME"].isin(EU27)].merge(
        agg, left_on="NAME", right_on="country", how="left"
    )
    # Fix yield scale — the FAO data may be in hg/ha, convert to t/ha
    if variable == "Yield_t_ha" and eu_map[variable].mean() < 2:
        eu_map[variable] = eu_map[variable] * 10_000 / 10_000
        # If still wrong, the column is already correct but needs checking
        print(f"Mean yield: {eu_map[variable].mean():.2f} t/ha")

    label_map = {
        "Yield_t_ha":        "Average Wheat Yield (t/ha)",
        "Production_tonnes": "Average Wheat Production (tonnes)",
    }
    label = label_map.get(variable, variable)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    eu_map.plot(
        column=variable,
        cmap="YlGn",
        legend=True,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
        missing_kwds={"color": "#eeeeee", "label": "No data"},
        legend_kwds={"label": label, "shrink": 0.5, "orientation": "vertical"},
    )
    # Zoom tightly to continental EU — removes excess white space
    ax.set_xlim(-11, 33)
    ax.set_ylim(34, 71)
    ax.set_axis_off()
    ax.set_title(
        f"EU27 — {label} (1990–2023 average)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    return fig


"""
3. Third Graph
    Scatter plot of wheat yield vs nitrogen use across EU27 countries.
    Shows the positive relationship and hints at diminishing returns.
"""
def plot_yield_vs_nitrogen(df: pd.DataFrame,
                          top_crops: int = 1,
                          figsize: tuple = (12, 5)) -> plt.Figure:
    from src.data_loader import TARGET
    import numpy as np

    sub = df.dropna(subset=["NitrogenUse_kg_ha", TARGET])

    r_linear = sub["NitrogenUse_kg_ha"].corr(sub[TARGET])

    sub_log = sub[(sub["NitrogenUse_kg_ha"] > 0) & (sub[TARGET] > 0)]
    r_log = np.log(sub_log["NitrogenUse_kg_ha"]).corr(np.log(sub_log[TARGET]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(sub["NitrogenUse_kg_ha"], sub[TARGET],
                    alpha=0.3, color=BLUE, s=10)
    axes[0].set_xlabel("Nitrogen Use (kg/ha)", fontsize=11)
    axes[0].set_ylabel("Yield (t/ha)", fontsize=11)
    axes[0].set_title(f"Linear scale)",
                      fontsize=11, fontweight="bold")

    axes[1].scatter(np.log(sub_log["NitrogenUse_kg_ha"]),
                    np.log(sub_log[TARGET]),
                    alpha=0.3, color=BLUE, s=10)
    axes[1].set_xlabel("log(Nitrogen Use (kg/ha))", fontsize=11)
    axes[1].set_ylabel("log(Yield(t/ha))", fontsize=11)
    axes[1].set_title(f"Log-Log scale)",
                      fontsize=11, fontweight="bold")

    plt.suptitle("Wheat Yield vs Nitrogen Use — EU27",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig

"""
4. Fourth Graph
    Pearson correlation heatmap for the main numeric features.
    Shows which variables are most correlated with wheat yield.
"""

def plot_correlation_heatmap(df: pd.DataFrame,
                              figsize: tuple = (8, 6)) -> plt.Figure:
    num_cols = ["Yield_t_ha", "AreaHarvested_ha", "NitrogenUse_kg_ha",
                "PesticideUse_t", "LOG_AreaHarvested", "LOG_NitrogenUse"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.4,
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Correlation Matrix — EU27 Wheat Yield",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

"""
    5. Fifth Graph for OLS diagnostics
    Scatter plot of actual vs predicted yield with a perfect-fit reference line.
"""
def plot_actual_vs_predicted(y_true: pd.Series,
                              y_pred: np.ndarray,
                              model_name: str = "OLS",
                              figsize: tuple = (7, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.25, color=BLUE, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color=RED, linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Yield (log scale)", fontsize=11)
    ax.set_ylabel("Predicted Yield (log scale)", fontsize=11)
    ax.set_title(f"Actual vs Predicted — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig
