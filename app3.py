"""
app3.py  ─  TP-317 RNAseq  |  Gene–Tumour Correlation Explorer  (CPM-normalized)
==================================================================================
Scatter plots of tumour volume (X) vs gene expression (Y) across all 10
mouse tumour experiments, with Pearson regression, 95% CI band, and
unadjusted p-value per panel.

NORMALIZATION
─────────────
Raw read counts are converted to CPM before computing correlations:

    CPM_ij = (raw_count_ij / sample_total_j) × 1,000,000

Column sums are computed once at startup (cached). This removes
sequencing-depth variation without needing gene-length information.

Y-axis: CPM  (not log-transformed here so the correlation is in a
               linear, interpretable space matching tumour volume)

STATISTICS
──────────
Pearson r, unadjusted p (two-tailed t-distribution with n−2 df).
No multiple-testing correction — each panel is an independent experiment.

FOLDER STRUCTURE
────────────────
    app3.py
    gene_list.csv
    datafiles/        ← the 10 raw RNAseq CSV files
"""

import csv
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP-317 Gene–Tumour Correlation",
    page_icon="📈",
    layout="wide",
)

# ── File locations ────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "datafiles")

EXPERIMENTS = [
    ("K5 · B16F10",  "_01_K5_B16F10_bulkRNAseqCounts_msTP1.csv"),
    ("E10 · B16F10", "_02_E10_B16F10_bulkRNAseqCounts_msTP1.csv"),
    ("K1 · LLC",     "_03_K1_LLC_bulkRNAseqCounts_msTP1.csv"),
    ("K8 · LLC",     "_04_K8_LLC_bulkRNAseqCounts_msTP1.csv"),
    ("E13 · KPCY",   "_05_E13_KPCY_bulkRNAseqCounts_msTP1.csv"),
    ("E26 · KPCY",   "_06_E26_KPCY_bulkRNAseqCounts_msTP1.csv"),
    ("E8 · KPC",     "_07_E8_KPC_bulkRNAseqCounts_msTP1.csv"),
    ("E9 · Panc2",   "_08_E9_Panc2_bulkRNAseqCounts_msTP1.csv"),
    ("K22 · CT26",   "_09_K22_CT26_bulkRNAseqCounts_msTP1.csv"),
    ("K23 · CT26",   "_10_K23_CT26_bulkRNAseqCounts_msTP1.csv"),
]

# Treatment colours (same palette as app.py / app2.py)
GROUP_ORDER = [
    "CONTR", "TP_300", "TP317_300", "TP_3000", "TP317",
    "aPD1", "dual-IO", "TP+aPD1", "TP-dual-IO",
    "cis+aPD1", "triple", "BLTi", "TP+BLTi",
]
COLOURS = {
    "CONTR":      "#888888",
    "TP_300":     "#a8d5a2",
    "TP317_300":  "#a8d5a2",
    "TP_3000":    "#4caf50",
    "TP317":      "#2e7d32",
    "aPD1":       "#f8bbd0",
    "dual-IO":    "#f8bbd0",
    "TP+aPD1":    "#b71c1c",
    "TP-dual-IO": "#b71c1c",
    "cis+aPD1":   "#880e4f",
    "triple":     "#6a1b9a",
    "BLTi":       "#f9a825",
    "TP+BLTi":    "#e65100",
}
DEFAULT_COLOUR = "#555555"

# Pearson r colour thresholds for annotation colour
R_COLOURS = {
    (0.7, 1.01):  "#c62828",  # strong positive
    (0.4, 0.7):   "#e65100",  # moderate positive
    (-0.4, 0.4):  "#555555",  # weak
    (-0.7, -0.4): "#1565c0",  # moderate negative
    (-1.01, -0.7):"#0d47a1",  # strong negative
}


def _r_colour(r):
    for (lo, hi), col in R_COLOURS.items():
        if lo <= r < hi:
            return col
    return "#555555"


# ── CPM normalization helpers ─────────────────────────────────────────────────

def _find_sample_cols(header_rows):
    sample_cols = []
    n_cols = max(len(r) for r in header_rows)
    row1, row2 = header_rows[1], header_rows[2]
    for c in range(1, n_cols):
        trt = row1[c].strip() if c < len(row1) else ""
        vol = row2[c].strip() if c < len(row2) else ""
        if trt and trt != "nan" and vol and vol != "nan":
            try:
                float(vol)
                sample_cols.append(c)
            except ValueError:
                pass
    return sample_cols


@st.cache_data(show_spinner=False)
def get_experiment_meta(path):
    """
    Returns dict with sample_cols, col_sums, treatments, tumor_vols,
    gene_col_idx — computed once and cached.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header_rows = [next(reader) for _ in range(3)]

    sample_cols = _find_sample_cols(header_rows)
    treatments  = [header_rows[1][c].strip() for c in sample_cols]
    tumor_vols  = [float(header_rows[2][c]) for c in sample_cols]

    gene_col_idx = next(
        (i for i, v in enumerate(header_rows[2]) if v.strip() == "gene_name"),
        None,
    )

    col_sums = np.zeros(len(sample_cols), dtype=np.float64)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for _ in range(3):
            next(reader)
        for row in reader:
            for k, c in enumerate(sample_cols):
                if c < len(row):
                    try:
                        col_sums[k] += float(row[c])
                    except ValueError:
                        pass

    return {
        "sample_cols":  sample_cols,
        "col_sums":     col_sums,
        "treatments":   treatments,
        "tumor_vols":   tumor_vols,
        "gene_col_idx": gene_col_idx,
    }


@st.cache_data(show_spinner=False)
def load_gene_cpm_corr(path, gene_name):
    """
    Return (x_tumor_vol, y_cpm, treatments) as parallel arrays,
    or None if the gene is not found.
    """
    meta = get_experiment_meta(path)
    sample_cols  = meta["sample_cols"]
    col_sums     = meta["col_sums"]
    treatments   = meta["treatments"]
    tumor_vols   = meta["tumor_vols"]
    gene_col_idx = meta["gene_col_idx"]

    if gene_col_idx is None:
        return None

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for _ in range(3):
            next(reader)
        for row in reader:
            if gene_col_idx < len(row) and row[gene_col_idx].strip() == gene_name:
                raw = []
                for c in sample_cols:
                    try:
                        raw.append(float(row[c]) if c < len(row) else 0.0)
                    except ValueError:
                        raw.append(0.0)
                cpm = np.array(raw) / col_sums * 1_000_000
                return (
                    np.array(tumor_vols, dtype=float),
                    cpm,
                    treatments,
                )
    return None


# ── Gene list ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_gene_list():
    gene_file = os.path.join(_HERE, "gene_list.csv")
    if not os.path.isfile(gene_file):
        return []
    with open(gene_file) as f:
        return [line.strip() for line in f if line.strip()]


# ── Figure builder ────────────────────────────────────────────────────────────
def build_figure(gene_name, all_data):
    """
    all_data: list of (label, result_or_None)
    result   = (x_tumor_vol, y_cpm, treatments)
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, (label, result) in enumerate(all_data):
        ax = axes[idx]
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        if result is None:
            ax.text(0.5, 0.5, "Gene not found\nin this dataset",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")
            ax.axis("off")
            continue

        x_vol, y_cpm, treatments = result

        # Determine axis label
        is_weight = "E9" in label or "Panc2" in label
        x_label = "Tumour weight (g)" if is_weight else "Tumour volume (% increase)"

        # Scatter coloured by treatment
        for i, (xv, yv, trt) in enumerate(zip(x_vol, y_cpm, treatments)):
            col = COLOURS.get(trt, DEFAULT_COLOUR)
            ax.scatter(xv, yv, color=col, s=35, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4)

        # Pearson regression + 95% CI band
        n = len(x_vol)
        if n >= 4:
            r, p = stats.pearsonr(x_vol, y_cpm)

            # Regression line
            slope, intercept, _, _, _ = stats.linregress(x_vol, y_cpm)
            x_fit = np.linspace(x_vol.min(), x_vol.max(), 200)
            y_fit = slope * x_fit + intercept

            # 95% CI (pointwise, ±t * SE(ŷ))
            x_mean = x_vol.mean()
            SSx    = np.sum((x_vol - x_mean) ** 2)
            se_fit = np.sqrt(
                np.sum((y_cpm - (slope * x_vol + intercept)) ** 2) / (n - 2)
            ) * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / SSx)

            t_crit = stats.t.ppf(0.975, df=n - 2)
            ci_lo  = y_fit - t_crit * se_fit
            ci_hi  = y_fit + t_crit * se_fit

            ax.plot(x_fit, y_fit, color="#333333", linewidth=1.4, zorder=4)
            ax.fill_between(x_fit, ci_lo, ci_hi, color="#bbbbbb",
                            alpha=0.35, zorder=2)

            # Annotation: r and p
            p_str = f"p = {p:.3f}" if p >= 0.001 else f"p < 0.001"
            r_col = _r_colour(r)
            ax.text(0.97, 0.96, f"r = {r:.2f}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    ha="right", va="top", color=r_col)
            ax.text(0.97, 0.86, p_str,
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", color="#333333")

        ax.set_xlabel(x_label, fontsize=7.5)
        ax.set_ylabel(f"{gene_name} (CPM)", fontsize=7.5)
        ax.tick_params(labelsize=7)

    # Legend (treatments actually present)
    seen_trts = set()
    for _, result in all_data:
        if result is not None:
            seen_trts |= set(result[2])
    legend_patches = []
    for grp in GROUP_ORDER:
        if grp in seen_trts:
            from matplotlib.patches import Patch
            legend_patches.append(Patch(facecolor=COLOURS.get(grp, DEFAULT_COLOUR),
                                         label=grp))

    fig.suptitle(
        f"Gene–Tumour Correlation  ·  {gene_name}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    if legend_patches:
        fig.legend(
            handles=legend_patches, loc="lower center",
            ncol=min(len(legend_patches), 8), fontsize=7.5,
            frameon=False, bbox_to_anchor=(0.5, 0.005),
        )
    fig.text(
        0.5, 0.44,
        "Y-axis: CPM (Counts Per Million, sequencing-depth normalized)  |  "
        "Line: Pearson regression + 95% CI band  |  "
        "p-values unadjusted, two-tailed",
        ha="center", fontsize=7, color="gray",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Gene selection")

gene_list = load_gene_list()
if gene_list:
    gene_input = st.sidebar.selectbox(
        "Select gene (type to search)",
        options=[""] + gene_list,
        index=0,
    )
else:
    gene_input = st.sidebar.text_input(
        "Gene name (case-sensitive)",
        value="",
        placeholder="e.g. Tap1, Cd8a, Cxcl9",
    )

plot_button = st.sidebar.button("▶  Plot", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Normalization**")
st.sidebar.markdown(
    "Gene expression is shown as **CPM** (Counts Per Million), "
    "correcting for sequencing-depth differences between samples.  \n\n"
    "No multiple-testing correction applied — each panel is an "
    "independent biological experiment."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data files**")
for label, fname in EXPERIMENTS:
    path = os.path.join(DATA_DIR, fname)
    st.sidebar.markdown(
        f"{'✅' if os.path.isfile(path) else '❌'} {label}"
        + ("" if os.path.isfile(path) else " — **not found**")
    )

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📈 TP-317 · Gene–Tumour Correlation Explorer")
st.markdown(
    "Select a gene to display scatter plots of tumour volume vs gene "
    "expression across all 10 experiments.  \n"
    "**Y-axis**: CPM (depth-normalized)  |  "
    "**Line**: Pearson regression + 95% CI  |  "
    "**Colour**: treatment group"
)

if plot_button:
    if not gene_input:
        st.warning("Please select or type a gene name first.")
    else:
        found_in = []
        not_found_in = []

        with st.spinner(f"Building correlation plots for *{gene_input}* …"):
            all_data = []
            for label, fname in EXPERIMENTS:
                path = os.path.join(DATA_DIR, fname)
                if not os.path.isfile(path):
                    all_data.append((label, None))
                    not_found_in.append(label)
                    continue
                result = load_gene_cpm_corr(path, gene_input)
                all_data.append((label, result))
                if result is not None:
                    found_in.append(label)
                else:
                    not_found_in.append(label)

        if not found_in:
            st.error(
                f"Gene **{gene_input}** was not found in any dataset. "
                "Check spelling — gene names are case-sensitive."
            )
        else:
            fig = build_figure(gene_input, all_data)
            st.pyplot(fig, use_container_width=True)

            png_buf = io.BytesIO()
            svg_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=180,
                        bbox_inches="tight", facecolor="white")
            fig.savefig(svg_buf, format="svg",
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)

            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                st.download_button(
                    "⬇️ Download PNG", png_buf.getvalue(),
                    file_name=f"correlation_CPM_{gene_input}.png",
                    mime="image/png",
                )
            with c2:
                st.download_button(
                    "⬇️ Download SVG", svg_buf.getvalue(),
                    file_name=f"correlation_CPM_{gene_input}.svg",
                    mime="image/svg+xml",
                )

            if not_found_in:
                st.info(
                    f"ℹ️ **{gene_input}** not found in: "
                    + ", ".join(not_found_in)
                    + " (different sequencing platform)"
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: bulk RNAseq · 10 mouse tumour experiments  |  Drug: TP-317  |  "
    "X-axis: tumour volume % increase (or weight in g for E9·Panc2)  |  "
    "Y-axis: CPM (Counts Per Million, sequencing-depth normalized)  |  "
    "Statistics: Pearson r, unadjusted p"
)
