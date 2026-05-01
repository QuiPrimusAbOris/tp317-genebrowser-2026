"""
app3.py  ─  TP-317 RNAseq  |  Gene–Tumor Correlation Explorer
═══════════════════════════════════════════════════════════════
For a user-selected gene G the app shows, across all 10 mouse
tumour experiments, a scatter plot of

    X  =  tumour volume (% increase from baseline)
              ── or ──
           tumour weight at sacrifice (g)    [E9 · Panc2 only]

    Y  =  expression of gene G (raw read counts)

together with a Pearson regression line, 95 % confidence band,
and the unadjusted Pearson r / t-statistic p-value per panel.

Statistics note
───────────────
p-values are NOT corrected for multiple testing.  Each panel is
an independent biological experiment and a single-feature test:
    t = r * sqrt((n-2) / (1-r²))    p from t_{n-2}, two-tailed.

Folder layout expected in the GitHub repo
─────────────────────────────────────────
    app3.py
    requirements.txt
    datafiles/
        _01_K5_B16F10_bulkRNAseqCounts_msTP1.csv
        _02_E10_B16F10_bulkRNAseqCounts_msTP1.csv
        … (all 10 CSV files)
"""

import csv
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "datafiles")   # matches your GitHub folder name

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Experiment registry
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {"id": "01", "label": "K5 · B16F10",  "file": "_01_K5_B16F10_bulkRNAseqCounts_msTP1.csv",  "model": "Melanoma"},
    {"id": "02", "label": "E10 · B16F10", "file": "_02_E10_B16F10_bulkRNAseqCounts_msTP1.csv", "model": "Melanoma"},
    {"id": "03", "label": "K1 · LLC",     "file": "_03_K1_LLC_bulkRNAseqCounts_msTP1.csv",     "model": "Lung"},
    {"id": "04", "label": "K8 · LLC",     "file": "_04_K8_LLC_bulkRNAseqCounts_msTP1.csv",     "model": "Lung"},
    {"id": "05", "label": "E13 · KPCY",   "file": "_05_E13_KPCY_bulkRNAseqCounts_msTP1.csv",   "model": "Pancreatic"},
    {"id": "06", "label": "E26 · KPCY",   "file": "_06_E26_KPCY_bulkRNAseqCounts_msTP1.csv",   "model": "Pancreatic"},
    {"id": "07", "label": "E8 · KPC",     "file": "_07_E8_KPC_bulkRNAseqCounts_msTP1.csv",     "model": "Pancreatic"},
    {"id": "08", "label": "E9 · Panc2",   "file": "_08_E9_Panc2_bulkRNAseqCounts_msTP1.csv",   "model": "Pancreatic"},
    {"id": "09", "label": "K22 · CT26",   "file": "_09_K22_CT26_bulkRNAseqCounts_msTP1.csv",   "model": "Colorectal"},
    {"id": "10", "label": "K23 · CT26",   "file": "_10_K23_CT26_bulkRNAseqCounts_msTP1.csv",   "model": "Colorectal"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Colour palettes
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_PALETTE = {
    # Control
    "CONTR":        "#888888",   # grey
    # TP317 monotherapy – green shades (light → dark with dose)
    "TP317":        "#2ca25f",   # medium green
    "TP_300":       "#74c476",   # light green   (300 mg/kg)
    "TP_3000":      "#006d2c",   # dark green    (3000 mg/kg)
    "TP317_300":    "#41ab5d",   # green
    # IO monotherapy – pink
    "aPD1":         "#f48fb1",   # pink
    "dual-IO":      "#f06292",   # deeper pink
    # TP317 + IO combos – dark red
    "TP+aPD1":      "#c0392b",   # dark red
    "TP-dual-IO":   "#922b21",   # darker red
    # Platinum combo – dark pink / magenta
    "cis+aPD1":     "#ad1457",
    # Triple combo – purple
    "triple":       "#7b2d8b",
    # BLT inhibitor – yellow / orange
    "BLTi":         "#f9d71c",   # yellow
    "TP+BLTi":      "#e67e22",   # orange
}

MODEL_COLORS = {
    "Melanoma":   "#2171b5",
    "Lung":       "#31a354",
    "Pancreatic": "#756bb1",
    "Colorectal": "#fd8d3c",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data loading  (cached)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_path(filename: str) -> str | None:
    """Check datafiles/ subfolder first, then repo root."""
    for candidate in [
        os.path.join(_DATA_DIR, filename),
        os.path.join(_HERE, filename),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_all_experiments() -> tuple[dict, list]:
    all_data: dict = {}
    all_genes: set = set()

    for exp in EXPERIMENTS:
        fp = _resolve_path(exp["file"])
        if fp is None:
            all_data[exp["id"]] = None
            continue

        with open(fp, encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))

        row0, row1, row2 = rows[0], rows[1], rows[2]
        tumor_label_raw = row2[0].strip()

        # Column where gene annotations start
        gene_name_col = next(
            (i for i, v in enumerate(row2) if v.strip().lower() == "gene_name"),
            None,
        )
        # Number of contiguous numeric sample columns
        n = 0
        for i in range(1, len(row2)):
            try:
                float(row2[i].replace(",", ""))
                n += 1
            except (ValueError, AttributeError):
                break

        treatments = [row1[i].strip() for i in range(1, n + 1)]
        tumor_vals = [float(row2[i].replace(",", "")) for i in range(1, n + 1)]
        x_label = (
            "Tumor Weight at sac (g)"
            if "weight" in tumor_label_raw.lower()
            else "Tumor Vol. (% increase)"
        )

        gene_expr: dict = {}
        for row in rows[3:]:
            if gene_name_col is None or gene_name_col >= len(row):
                continue
            gname = row[gene_name_col].strip()
            if not gname or gname.lower() == "gene_name":
                continue
            try:
                vals = [
                    float(row[i]) if i < len(row) and row[i].strip() else np.nan
                    for i in range(1, n + 1)
                ]
                gene_expr[gname] = vals
            except ValueError:
                continue

        all_data[exp["id"]] = dict(
            treatments=treatments,
            tumor_vals=tumor_vals,
            x_label=x_label,
            gene_expr=gene_expr,
        )
        all_genes.update(gene_expr.keys())

    return all_data, sorted(all_genes, key=str.casefold)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  95 % confidence band for regression line
# ─────────────────────────────────────────────────────────────────────────────
def _conf_band(
    x: np.ndarray, y: np.ndarray, x_line: np.ndarray, alpha: float = 0.95
) -> tuple:
    """
    Returns (lo, hi, slope, intercept, r, p).

    p-value: unadjusted, from t = r*sqrt((n-2)/(1-r^2)), t_{n-2}, two-tailed.
    No multiple-testing correction applied.
    """
    n = len(x)
    slope, intercept, r, p, _ = stats.linregress(x, y)   # scipy Pearson
    y_hat  = slope * x + intercept
    s_err  = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    x_mean = np.mean(x)
    t_crit = stats.t.ppf((1 + alpha) / 2, df=n - 2)
    se_ln  = s_err * np.sqrt(
        1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    )
    lo = slope * x_line + intercept - t_crit * se_ln
    hi = slope * x_line + intercept + t_crit * se_ln
    return lo, hi, slope, intercept, r, p


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure builder
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(gene: str, all_data: dict) -> plt.Figure:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "axes.labelsize":     8.5,
    })

    fig, axes = plt.subplots(5, 2, figsize=(13, 22))
    fig.suptitle(gene, fontsize=30, fontweight="bold", y=0.9985, color="#1a1a2e")

    for idx, exp in enumerate(EXPERIMENTS):
        ax        = axes[idx // 2][idx % 2]
        dat       = all_data.get(exp["id"])
        model_col = MODEL_COLORS[exp["model"]]

        ax.set_title(
            f"{exp['label']}  —  {exp['model']}",
            fontsize=10, fontweight="bold", color=model_col, pad=5,
        )

        # ── Missing file ───────────────────────────────────────────────────
        if dat is None:
            ax.text(0.5, 0.5, "Data file not found",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)
            continue

        # ── Gene absent in this dataset ────────────────────────────────────
        gv = dat["gene_expr"].get(gene)
        if gv is None:
            ax.text(0.5, 0.5, "Gene not in this dataset",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)
            continue

        xraw = np.array(dat["tumor_vals"], dtype=float)
        yraw = np.array(gv,                dtype=float)
        trts = np.array(dat["treatments"])

        valid = np.isfinite(xraw) & np.isfinite(yraw)
        x, y, trts = xraw[valid], yraw[valid], trts[valid]

        if len(x) < 3:
            ax.text(0.5, 0.5, "Insufficient data (n < 3)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)
            continue

        # ── Regression + 95 % CI band ──────────────────────────────────────
        x_line = np.linspace(x.min(), x.max(), 300)
        lo, hi, slope, intercept, r_val, p_val = _conf_band(x, y, x_line)

        ax.fill_between(x_line, lo, hi, color="#cccccc", alpha=0.50, zorder=1)
        ax.plot(x_line, slope * x_line + intercept,
                color="#333333", linewidth=1.8, zorder=3)

        # ── Scatter: one series per treatment group ────────────────────────
        for tg in dict.fromkeys(trts):      # unique, order-preserving
            mask = trts == tg
            col  = TREATMENT_PALETTE.get(tg, "#aaaaaa")
            ax.scatter(
                x[mask], y[mask],
                color=col, s=80,
                edgecolors="white", linewidths=0.7,
                zorder=4, label=tg, alpha=0.92,
            )

        # ── Inset legend ───────────────────────────────────────────────────
        leg = ax.legend(
            fontsize=6.8, loc="best",
            framealpha=0.88, edgecolor="#cccccc",
            frameon=True, borderpad=0.5,
            labelspacing=0.35, handlelength=0.9,
            handletextpad=0.5, markerscale=1.0,
        )
        leg.get_frame().set_linewidth(0.6)

        # ── Stats annotation (bottom-right) ───────────────────────────────
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        r_str = f"r = {r_val:+.3f}"
        if   abs(r_val) >= 0.6:  r_col = "#c0392b"
        elif abs(r_val) >= 0.35: r_col = "#e67e22"
        else:                     r_col = "#555555"

        ax.text(
            0.97, 0.03, f"{r_str}\n{p_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=r_col,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#cccccc", lw=0.7, alpha=0.92),
        )

        # ── Axis labels & limits ───────────────────────────────────────────
        ax.set_xlabel(dat["x_label"], fontsize=8.5, labelpad=3)
        ax.set_ylabel(f"{gene} expression (reads)", fontsize=8.5, labelpad=3)
        ax.tick_params(axis="both", which="major", length=3, width=0.7)

        x_pad = 0.04 * (x.max() - x.min()) if x.max() != x.min() else 1
        y_pad = 0.06 * (y.max() - y.min()) if y.max() != y.min() else 1
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(max(0, y.min() - y_pad), y.max() + y_pad)

        # ── Coloured left spine = model indicator ──────────────────────────
        ax.spines["left"].set_color(model_col)
        ax.spines["left"].set_linewidth(2.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.994], h_pad=3.5, w_pad=3.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Streamlit page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP-317 · Correlation Explorer",
    page_icon="🧬",
    layout="wide",
)

# Load data
with st.spinner("Loading all 10 experiments…"):
    all_data, all_genes = load_all_experiments()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 TP-317 Study")
    st.caption("Gene–Tumor Correlation Explorer · app3")
    st.divider()

    default_idx = all_genes.index("Cd8a") if "Cd8a" in all_genes else 0
    gene_input = st.selectbox(
        "Select gene (G)",
        options=all_genes,
        index=default_idx,
        help="Type to filter — autocomplete across all 10 experiments.",
    )

    plot_button = st.button("▶ Plot", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Tumour model (spine colour)**")
    for model, col in MODEL_COLORS.items():
        st.markdown(
            f"<span style='color:{col}; font-size:18px;'>▌</span> {model}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Treatment group colours**")
    palette_display = [
        ("CONTR",       "Control"),
        ("TP317",       "TP-317"),
        ("TP_300",      "TP-317 300 mg/kg"),
        ("TP_3000",     "TP-317 3000 mg/kg"),
        ("TP317_300",   "TP-317 300 mg/kg"),
        ("aPD1",        "aPD-1"),
        ("dual-IO",     "Dual IO"),
        ("TP+aPD1",     "TP-317 + aPD-1"),
        ("TP-dual-IO",  "TP-317 + Dual IO"),
        ("cis+aPD1",    "Cis + aPD-1"),
        ("triple",      "Triple combo"),
        ("BLTi",        "BLTi"),
        ("TP+BLTi",     "TP-317 + BLTi"),
    ]
    for key, label in palette_display:
        c = TREATMENT_PALETTE[key]
        st.markdown(
            f"<span style='color:{c}; font-size:16px;'>●</span> {label}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Data files**")
    for exp in EXPERIMENTS:
        found = _resolve_path(exp["file"]) is not None
        st.caption(f"{'✅' if found else '❌'} {exp['label']}")

    st.divider()
    st.markdown("**Statistics**")
    st.caption(
        "Unadjusted Pearson r.  \n"
        "p from t-statistic (df = n−2), two-tailed.  \n"
        "No multiple-testing correction applied."
    )

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("TP-317 · Gene–Tumor Correlation Explorer")
st.markdown(
    "Select a gene in the sidebar, then press **▶ Plot**.  \n"
    "Each panel shows a scatter plot of **tumour volume** (or weight for E9·Panc2) "
    "vs **gene expression** (raw reads), with Pearson regression line and 95 % "
    "confidence band. Dots are coloured by treatment group."
)

if plot_button:
    if not gene_input:
        st.warning("Please select a gene first.")
    else:
        with st.spinner(f"Building plots for *{gene_input}*…"):
            fig = build_figure(gene_input, all_data)

        st.pyplot(fig, use_container_width=True)

        # ── Download buttons ──────────────────────────────────────────────
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
                file_name=f"correlation_{gene_input}.png",
                mime="image/png",
            )
        with c2:
            st.download_button(
                "⬇️ Download SVG", svg_buf.getvalue(),
                file_name=f"correlation_{gene_input}.svg",
                mime="image/svg+xml",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: bulk RNAseq · 10 mouse tumour experiments "
    "(B16F10 melanoma · LLC lung · KPCY/KPC/Panc2 pancreatic · CT26 colorectal)  |  "
    "Drug: TP-317  |  "
    "X axis: tumour volume % increase (or weight in g for E9·Panc2)  |  "
    "Y axis: raw read counts  |  "
    "Statistics: unadjusted Pearson r, t-test p-value, no correction for multiple testing"
)
