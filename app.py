"""
TP-317 Gene Expression Browser  —  v3  (CPM-normalized)
=========================================================
Displays log2(CPM + 1) expression across all 10 experiments for any gene,
with all-pairwise Mann-Whitney U significance brackets.

NORMALIZATION
─────────────
Raw read counts are converted to CPM (Counts Per Million) before plotting:

    CPM_ij = (raw_count_ij / sample_total_j) × 1,000,000

This removes sequencing-depth differences between samples without
requiring gene-length information. Column sums are computed once at
startup and cached; gene values are normalized on the fly.

Y-axis: log₂(CPM + 1) — log-transform compresses the dynamic range
        and improves readability.

STATISTICS
──────────
Mann-Whitney U, all pairwise comparisons, unadjusted p-values.
Comparisons are hypothesis-driven (prior expectation of TP-317
activity), so multiple-testing correction does not apply.

Thresholds:  * p < 0.05  |  ** p < 0.01  |  *** p < 0.001

FOLDER STRUCTURE (GitHub repo)
───────────────────────────────
    app.py
    gene_list.csv          ← gene names, one per line
    datafiles/             ← the 10 raw RNAseq CSV files
"""

import csv
import io
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import mannwhitneyu
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP-317 Gene Expression Browser",
    page_icon="🧬",
    layout="wide",
)

# ── File locations ────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "datafiles")

# All 10 experiments in display order (label, filename)
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

# Canonical treatment-group display order (unknown groups appended after)
GROUP_ORDER = [
    "CONTR",
    "TP_300", "TP317_300",
    "TP_3000", "TP317",
    "aPD1", "dual-IO",
    "TP+aPD1", "TP-dual-IO",
    "cis+aPD1",
    "triple",
    "BLTi",
    "TP+BLTi",
]

# Colour map for treatment groups
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

P_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


# ── CPM Normalization helpers ─────────────────────────────────────────────────

def _find_sample_cols(header_rows):
    """
    Given the first 3 rows of a CSV (as a list-of-lists), return the column
    indices that correspond to biological samples.
    A column qualifies if its TreatmentGroup (row 1) is a non-empty string
    AND its TumorVolume (row 2) can be parsed as a float.
    """
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
    Read an experiment CSV once and return:
      sample_cols   : list of int column indices for biological samples
      col_sums      : np.array of total raw counts per sample  (CPM denominator)
      treatments    : list of treatment-group strings, one per sample
      tumor_vols    : list of float tumor volumes, one per sample
      gene_col_idx  : int column index of the 'gene_name' column in data rows
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header_rows = [next(reader) for _ in range(3)]

    sample_cols = _find_sample_cols(header_rows)

    treatments = [header_rows[1][c].strip() for c in sample_cols]
    tumor_vols = [float(header_rows[2][c]) for c in sample_cols]

    # Find gene_name column index
    gene_col_idx = next(
        (i for i, v in enumerate(header_rows[2]) if v.strip() == "gene_name"),
        None,
    )

    # Accumulate column sums by reading data rows
    col_sums = np.zeros(len(sample_cols), dtype=np.float64)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for _ in range(3):
            next(reader)          # skip 3 header rows
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
def load_gene_cpm(path, gene_name):
    """
    Return {treatment_group: [cpm_values]} for the requested gene,
    with CPM normalization applied using pre-computed column sums.
    Returns None if the gene is not found in this file.
    """
    meta = get_experiment_meta(path)
    sample_cols  = meta["sample_cols"]
    col_sums     = meta["col_sums"]
    treatments   = meta["treatments"]
    gene_col_idx = meta["gene_col_idx"]

    if gene_col_idx is None:
        return None

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for _ in range(3):
            next(reader)
        for row in reader:
            if gene_col_idx < len(row) and row[gene_col_idx].strip() == gene_name:
                raw_counts = []
                for c in sample_cols:
                    try:
                        raw_counts.append(float(row[c]) if c < len(row) else 0.0)
                    except ValueError:
                        raw_counts.append(0.0)

                # CPM normalization
                cpm = np.array(raw_counts) / col_sums * 1_000_000

                # Group by treatment
                groups = {}
                for i, trt in enumerate(treatments):
                    groups.setdefault(trt, []).append(cpm[i])
                return groups

    return None


# ── Gene list (for autocomplete) ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_gene_list():
    gene_file = os.path.join(_HERE, "gene_list.csv")
    if not os.path.isfile(gene_file):
        return []
    with open(gene_file) as f:
        return [line.strip() for line in f if line.strip()]


# ── Significance brackets ─────────────────────────────────────────────────────
def _bracket_label(p):
    for thresh, label in P_THRESHOLDS:
        if p < thresh:
            return label
    return None


def draw_brackets(ax, group_positions, group_values, y_max):
    """Draw all significant pairwise Mann-Whitney U brackets."""
    groups = list(group_positions.keys())
    n = len(groups)
    bracket_step = (y_max - ax.get_ylim()[0]) * 0.07
    current_top = y_max

    brackets = []
    for g1, g2 in combinations(groups, 2):
        v1, v2 = group_values[g1], group_values[g2]
        if len(v1) < 2 or len(v2) < 2:
            continue
        try:
            _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        except Exception:
            continue
        label = _bracket_label(p)
        if label:
            brackets.append((group_positions[g1], group_positions[g2], label))

    for x1, x2, label in brackets:
        y = current_top + bracket_step * 0.3
        ax.plot([x1, x1, x2, x2], [y, y + bracket_step * 0.3,
                                    y + bracket_step * 0.3, y],
                lw=1.0, color="black")
        ax.text((x1 + x2) / 2, y + bracket_step * 0.35, label,
                ha="center", va="bottom", fontsize=7)
        current_top = y + bracket_step * 0.6

    ax.set_ylim(top=current_top + bracket_step)


# ── Main figure ───────────────────────────────────────────────────────────────
def build_figure(gene_name, all_data):
    """
    all_data : list of (label, data_or_None) where data = {group: [cpm_vals]}
    """
    n_exp = len(all_data)
    fig = plt.figure(figsize=(22, 9))
    outer = gridspec.GridSpec(
        2, 5, figure=fig, hspace=0.55, wspace=0.45,
        left=0.04, right=0.97, top=0.88, bottom=0.12,
    )

    legend_patches = {}

    for idx, (label, data) in enumerate(all_data):
        row, col = divmod(idx, 5)
        ax = fig.add_subplot(outer[row, col])

        if data is None:
            ax.text(0.5, 0.5, "Gene not found\nin this dataset",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")
            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.axis("off")
            continue

        # Order groups canonically
        known   = [g for g in GROUP_ORDER if g in data]
        unknown = [g for g in data if g not in GROUP_ORDER]
        ordered_groups = known + unknown

        positions     = {}
        group_values  = {}
        log2_values   = {}

        for pos_idx, grp in enumerate(ordered_groups, start=1):
            raw_cpm = data[grp]
            log2    = list(np.log2(np.array(raw_cpm) + 1))
            positions[grp]    = pos_idx
            group_values[grp] = raw_cpm   # raw CPM for M-W test
            log2_values[grp]  = log2

        n_groups = len(ordered_groups)
        color_list = [COLOURS.get(g, DEFAULT_COLOUR) for g in ordered_groups]

        # Box plot
        bp = ax.boxplot(
            [log2_values[g] for g in ordered_groups],
            positions=list(positions.values()),
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor="white", color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        # Strip plot + coloured box faces
        for i, (grp, box) in enumerate(zip(ordered_groups, bp["boxes"])):
            col_hex = COLOURS.get(grp, DEFAULT_COLOUR)
            box.set_facecolor(col_hex + "40")   # 25% alpha fill
            x_jitter = positions[grp] + np.random.uniform(-0.15, 0.15,
                                                           len(log2_values[grp]))
            ax.scatter(x_jitter, log2_values[grp],
                       color=col_hex, s=28, zorder=3, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
            legend_patches[grp] = Patch(facecolor=col_hex, label=grp)

        # Axes formatting
        ax.set_xlim(0.3, n_groups + 0.7)
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(ordered_groups, rotation=35, ha="right",
                           fontsize=6.5)
        ax.set_ylabel("log₂(CPM + 1)", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Significance brackets (Mann-Whitney on raw CPM values)
        y_max = max(max(v) for v in log2_values.values())
        draw_brackets(ax, positions, group_values, y_max)

    # Figure title + legend
    fig.suptitle(
        f"Gene expression  ·  {gene_name}",
        fontsize=14, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.01,
        "Y-axis: log₂(CPM + 1)  |  CPM = Counts Per Million "
        "(sequencing-depth normalized)  |  "
        "Brackets: Mann-Whitney U, unadjusted p  |  * p<0.05  ** p<0.01  *** p<0.001",
        ha="center", fontsize=7, color="gray",
    )

    # Legend (ordered by GROUP_ORDER)
    ordered_legend = [legend_patches[g]
                      for g in GROUP_ORDER if g in legend_patches]
    other_legend   = [legend_patches[g]
                      for g in legend_patches if g not in GROUP_ORDER]
    all_legend = ordered_legend + other_legend
    if all_legend:
        fig.legend(
            handles=all_legend, loc="lower center",
            ncol=min(len(all_legend), 8), fontsize=7,
            frameon=False, bbox_to_anchor=(0.5, 0.03),
        )

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
    "Counts are converted to **CPM** (Counts Per Million) before "
    "plotting. This corrects for sequencing-depth differences between "
    "samples.\n\n"
    "Y-axis: log₂(CPM + 1)\n\n"
    "Statistics: Mann-Whitney U on raw CPM values, unadjusted p."
)

# ── File diagnostics (sidebar) ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Data files**")
for label, fname in EXPERIMENTS:
    path = os.path.join(DATA_DIR, fname)
    if os.path.isfile(path):
        st.sidebar.markdown(f"✅ {label}")
    else:
        st.sidebar.markdown(f"❌ {label} — **not found**")

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🧬 TP-317 Gene Expression Browser")
st.markdown(
    "Enter a gene name to visualise its expression across all 10 experiments.  \n"
    "**Y-axis**: log₂(CPM + 1)  ·  depth-normalized  "
    "|  **Statistics**: Mann-Whitney U, all pairwise, unadjusted p"
)

if plot_button:
    if not gene_input:
        st.warning("Please select or type a gene name first.")
    else:
        found_in = []
        not_found_in = []

        with st.spinner(f"Building plots for *{gene_input}* …"):
            all_data = []
            for label, fname in EXPERIMENTS:
                path = os.path.join(DATA_DIR, fname)
                if not os.path.isfile(path):
                    all_data.append((label, None))
                    not_found_in.append(label)
                    continue
                data = load_gene_cpm(path, gene_input)
                all_data.append((label, data))
                if data is not None:
                    found_in.append(label)
                else:
                    not_found_in.append(label)

        if not found_in:
            st.error(
                f"Gene **{gene_input}** was not found in any of the 10 datasets. "
                "Please check the spelling — gene names are case-sensitive."
            )
        else:
            fig = build_figure(gene_input, all_data)

            st.pyplot(fig, use_container_width=True)

            # Download
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
                    file_name=f"expr_CPM_{gene_input}.png",
                    mime="image/png",
                )
            with c2:
                st.download_button(
                    "⬇️ Download SVG", svg_buf.getvalue(),
                    file_name=f"expr_CPM_{gene_input}.svg",
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
    "Data: bulk RNAseq · 10 mouse tumour experiments "
    "(B16F10 melanoma · LLC lung · KPCY/KPC/Panc2 pancreatic · CT26 colorectal)  |  "
    "Drug: TP-317  |  "
    "Normalization: CPM (Counts Per Million) — corrects for sequencing depth  |  "
    "Statistics: Mann-Whitney U, unadjusted p-values"
)
