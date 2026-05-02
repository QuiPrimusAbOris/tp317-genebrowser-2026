"""
TP-317 Gene Expression Browser  —  v2
=======================================
Identical to app.py but uses pre-computed DESeq2 p-values for significance
brackets instead of Mann-Whitney U.

Statistics note: raw p-values (NOT padj) are used for bracket significance
thresholds, because comparisons are hypothesis-driven (prior expectation of
TP-317 activity), so multiple-testing correction does not apply.

Folder structure expected in the GitHub repo:
    app2.py
    gene_list.csv
    datafiles/          ← the 10 raw RNAseq CSV files
    deseq2_results/     ← the 10 per-experiment DESeq2 CSVs
        DESeq2_master_all_experiments.csv
        DESeq2_K5_B16F10.csv  … etc.
"""

import csv
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP-317 Gene Expression Browser v2",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 TP-317 Gene Expression Browser")
st.markdown(
    "Enter a gene name to visualise its expression across all 10 experiments.  \n"
    "**Y-axis**: log₂(raw count + 1)  |  "
    "**Statistics**: DESeq2 Wald test p-value (hypothesis-driven, uncorrected), "
    "brackets shown where p < 0.05"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_dir(name):
    candidate = os.path.join(_HERE, name)
    return candidate if os.path.isdir(candidate) else _HERE

DATA_DIR   = _find_dir("datafiles")
DESEQ_DIR  = _find_dir("deseq2_results")
MASTER_CSV = None   # not used — app loads per-experiment files directly

# ── Experiment definitions ────────────────────────────────────────────────────
# subplot_label  →  (csv_filename,  deseq2_key)
FILES = [
    ("_01_K5_B16F10",  "_01_K5_B16F10_bulkRNAseqCounts_msTP1.csv",  "K5_B16F10"),
    ("_02_E10_B16F10", "_02_E10_B16F10_bulkRNAseqCounts_msTP1.csv", "E10_B16F10"),
    ("_03_K1_LLC",     "_03_K1_LLC_bulkRNAseqCounts_msTP1.csv",     "K1_LLC"),
    ("_04_K8_LLC",     "_04_K8_LLC_bulkRNAseqCounts_msTP1.csv",     "K8_LLC"),
    ("_05_E13_KPCY",   "_05_E13_KPCY_bulkRNAseqCounts_msTP1.csv",   "E13_KPCY"),
    ("_06_E26_KPCY",   "_06_E26_KPCY_bulkRNAseqCounts_msTP1.csv",   "E26_KPCY"),
    ("_07_E8_KPC",     "_07_E8_KPC_bulkRNAseqCounts_msTP1.csv",     "E8_KPC"),
    ("_08_E9_Panc2",   "_08_E9_Panc2_bulkRNAseqCounts_msTP1.csv",   "E9_Panc2"),
    ("_09_K22_CT26",   "_09_K22_CT26_bulkRNAseqCounts_msTP1.csv",   "K22_CT26"),
    ("_10_K23_CT26",   "_10_K23_CT26_bulkRNAseqCounts_msTP1.csv",   "K23_CT26"),
]

GROUP_ORDER = [
    'CONTR', 'TP_300', 'TP_3000', 'TP317', 'TP317_300',
    'dual-IO', 'TP-dual-IO', 'aPD1', 'TP+aPD1',
    'cis+aPD1', 'triple', 'BLTi', 'TP+BLTi',
]
VALID_GROUPS = set(GROUP_ORDER)

GROUP_COLORS = {
    'CONTR':       '#AAAAAA',
    'TP_300':      '#90EE90',
    'TP317_300':   '#90EE90',
    'TP317':       '#3CB371',
    'TP_3000':     '#1A6B2A',
    'dual-IO':     '#FFB6C1',
    'TP-dual-IO':  '#FF69B4',
    'aPD1':        '#FFB6C1',
    'TP+aPD1':     '#CC2244',
    'cis+aPD1':    '#FF6688',
    'triple':      '#880022',
    'BLTi':        '#FFD700',
    'TP+BLTi':     '#FF8C00',
}

P_THRESH = 0.05   # raw (uncorrected) DESeq2 p-value threshold

# ── Load DESeq2 per-experiment tables ────────────────────────────────────────
@st.cache_data(show_spinner="Loading DESeq2 statistics…")
def load_deseq2():
    """
    Returns dict: deseq2_key -> DataFrame indexed by gene_name.
    Loads 10 individual per-experiment CSVs (each <12 MB).
    The master file is NOT needed and should NOT be uploaded to GitHub.
    """
    tables = {}
    for _, _, key in FILES:
        path = os.path.join(DESEQ_DIR, f"DESeq2_{key}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)  # index = ensembl_id
        df = df.reset_index().rename(columns={'ensembl_id': 'ensembl_id',
                                               'index': 'ensembl_id'})
        if 'ensembl_id' not in df.columns:
            df.insert(0, 'ensembl_id', df.index)
        df = df.set_index('gene_name', drop=False)
        df.index.name = 'gene_name'
        tables[key] = df
    return tables

# ── Load gene list for autocomplete ──────────────────────────────────────────
@st.cache_data
def load_gene_list():
    for candidate in [
        os.path.join(_HERE, "gene_list.csv"),
        os.path.join(_HERE, "datafiles", "gene_list.csv"),
        "gene_list.csv",
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return [line.strip() for line in f if line.strip()]
    return []

# ── Parse raw counts for one gene from one CSV ───────────────────────────────
@st.cache_data(show_spinner=False)
def parse_gene(path, gene):
    try:
        with open(path, 'r') as fh:
            reader = csv.reader(fh)
            next(reader)
            row2 = next(reader)
            row3 = next(reader)
            gene_col = next(
                (i for i, v in enumerate(row3) if v.strip() == 'gene_name'), None
            )
            if gene_col is None:
                return None, None
            for row in reader:
                if len(row) > gene_col and row[gene_col].strip() == gene:
                    groups = {}
                    for i in range(1, gene_col):
                        grp = row2[i].strip()
                        if grp not in VALID_GROUPS:
                            continue
                        try:
                            val = np.log2(float(row[i]) + 1)
                        except (ValueError, IndexError):
                            continue
                        groups.setdefault(grp, []).append(val)
                    ordered = [g for g in GROUP_ORDER if g in groups]
                    return ordered, groups
    except Exception:
        pass
    return None, None

# ── p-value label ─────────────────────────────────────────────────────────────
def pstr(p):
    if p < 0.001:  return 'p<0.001'
    elif p < 0.01: return 'p<0.01'
    else:          return f'p={p:.3f}'

# ── Get DESeq2 p-values for one gene in one experiment ───────────────────────
def get_deseq2_pvalues(deseq2_tables, gene, deseq2_key):
    """
    Returns dict: treatment_group → raw DESeq2 pvalue (vs CONTR).
    Looks up the per-experiment table for this experiment key.
    """
    if not deseq2_tables or deseq2_key not in deseq2_tables:
        return {}
    df = deseq2_tables[deseq2_key]
    if gene not in df.index:
        return {}
    row = df.loc[gene]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    pvals = {}
    for col in df.columns:
        if col.endswith('_vs_CONTR_pvalue'):
            treat = col.replace('_vs_CONTR_pvalue', '')
            val   = row[col]
            if pd.notna(val):
                pvals[treat] = float(val)
    return pvals

# ── Draw one subplot ──────────────────────────────────────────────────────────
def draw_subplot(ax, label, ordered_groups, groups, deseq2_pvals):
    if not ordered_groups:
        ax.text(0.5, 0.5, 'Gene not found\nin this dataset',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='grey')
        ax.set_title(label, fontsize=9.5, fontweight='bold', pad=5)
        ax.axis('off')
        return

    n_groups     = len(ordered_groups)
    positions    = np.arange(n_groups, dtype=float)
    pos_map      = {g: positions[i] for i, g in enumerate(ordered_groups)}
    group_values = [groups[g] for g in ordered_groups]
    face_colors  = [GROUP_COLORS.get(g, '#CCCCCC') for g in ordered_groups]

    # ── Box plots ─────────────────────────────────────────────────────────────
    bp = ax.boxplot(
        group_values, positions=positions, widths=0.45,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.3),
        capprops=dict(color='black', linewidth=1.5),
        boxprops=dict(color='black', linewidth=1.2),
    )
    for patch, fc in zip(bp['boxes'], face_colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.88)

    # ── Scatter dots ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    for pos, vals in zip(positions, group_values):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(pos + jitter, vals, s=60, color='black',
                   zorder=5, alpha=0.85, linewidths=0)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_groups, rotation=38, ha='right', fontsize=8)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylabel('log₂(count + 1)', fontsize=8)
    ax.set_title(label, fontsize=9.5, fontweight='bold', pad=5)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='grey')
    ax.set_axisbelow(True)

    # n= labels
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for pos, grp in zip(positions, ordered_groups):
        ax.text(pos, ylim[0] - span * 0.06, f'n={len(groups[grp])}',
                ha='center', va='top', fontsize=6.5)

    # ── Significance brackets from DESeq2 p-values ────────────────────────────
    # Only treatment vs CONTR comparisons are available from DESeq2
    if 'CONTR' not in ordered_groups or not deseq2_pvals:
        return

    pos_contr = pos_map['CONTR']
    sig_pairs = []
    for treat, p in deseq2_pvals.items():
        if treat in pos_map and p < P_THRESH:
            sig_pairs.append((pos_contr, pos_map[treat], p))

    if not sig_pairs:
        return

    # Sort by span width (narrower brackets lower)
    sig_pairs.sort(key=lambda x: abs(x[1] - x[0]))

    # Greedy non-overlapping level assignment
    levels = []
    level_intervals = []
    for pa, pb, p in sig_pairs:
        xlo, xhi = min(pa, pb), max(pa, pb)
        placed = False
        for lv, occupied in enumerate(level_intervals):
            if all(xhi <= iv[0] or xlo >= iv[1] for iv in occupied):
                levels.append(lv)
                occupied.append((xlo, xhi))
                placed = True
                break
        if not placed:
            levels.append(len(level_intervals))
            level_intervals.append([(xlo, xhi)])

    ylo, yhi  = ax.get_ylim()
    yspan     = yhi - ylo
    y_start   = yhi + yspan * 0.03
    y_step    = yspan * 0.11
    tip_h     = yspan * 0.015
    max_level = max(levels) if levels else 0

    ax.set_ylim(top=y_start + (max_level + 1) * y_step + yspan * 0.06)

    for (pa, pb, p), lv in zip(sig_pairs, levels):
        base_y = y_start + lv * y_step
        ax.plot([pa, pb], [base_y, base_y], color='black', lw=1.1, clip_on=False)
        ax.plot([pa, pa], [base_y - tip_h, base_y], color='black', lw=1.1, clip_on=False)
        ax.plot([pb, pb], [base_y - tip_h, base_y], color='black', lw=1.1, clip_on=False)
        ax.text((pa + pb) / 2, base_y + yspan * 0.008, pstr(p),
                ha='center', va='bottom', fontsize=6.5, clip_on=False)

# ── Build full figure ─────────────────────────────────────────────────────────
def build_figure(gene, parsed, deseq2_tables):
    row1_data = parsed[:5]
    row2_data = parsed[5:]
    ratios_r1 = [max(len(p[1]), 1) for p in row1_data]
    ratios_r2 = [max(len(p[1]), 1) for p in row2_data]

    fig = plt.figure(figsize=(22, 14), facecolor='white')
    gs1 = gridspec.GridSpec(1, 5, figure=fig, width_ratios=ratios_r1,
                             left=0.05, right=0.98, top=0.91, bottom=0.53,
                             wspace=0.45)
    gs2 = gridspec.GridSpec(1, 5, figure=fig, width_ratios=ratios_r2,
                             left=0.05, right=0.98, top=0.46, bottom=0.09,
                             wspace=0.45)

    for col, (label, ordered, groups, deseq2_key) in enumerate(row1_data):
        ax = fig.add_subplot(gs1[0, col])
        pvals = get_deseq2_pvalues(deseq2_tables, gene, deseq2_key)
        draw_subplot(ax, label, ordered, groups, pvals)

    for col, (label, ordered, groups, deseq2_key) in enumerate(row2_data):
        ax = fig.add_subplot(gs2[0, col])
        pvals = get_deseq2_pvalues(deseq2_tables, gene, deseq2_key)
        draw_subplot(ax, label, ordered, groups, pvals)

    legend_items = [
        Patch(facecolor=GROUP_COLORS['CONTR'],      edgecolor='black', label='CONTR (vehicle)'),
        Patch(facecolor=GROUP_COLORS['TP_300'],      edgecolor='black', label='TP317 low dose (300)'),
        Patch(facecolor=GROUP_COLORS['TP317'],       edgecolor='black', label='TP317 std dose'),
        Patch(facecolor=GROUP_COLORS['TP_3000'],     edgecolor='black', label='TP317 high dose (3000)'),
        Patch(facecolor=GROUP_COLORS['dual-IO'],     edgecolor='black', label='dual-IO'),
        Patch(facecolor=GROUP_COLORS['TP-dual-IO'],  edgecolor='black', label='TP + dual-IO'),
        Patch(facecolor=GROUP_COLORS['aPD1'],        edgecolor='black', label='aPD1'),
        Patch(facecolor=GROUP_COLORS['TP+aPD1'],     edgecolor='black', label='TP + aPD1'),
        Patch(facecolor=GROUP_COLORS['cis+aPD1'],    edgecolor='black', label='cis + aPD1'),
        Patch(facecolor=GROUP_COLORS['triple'],      edgecolor='black', label='triple combo'),
        Patch(facecolor=GROUP_COLORS['BLTi'],        edgecolor='black', label='BLTi'),
        Patch(facecolor=GROUP_COLORS['TP+BLTi'],     edgecolor='black', label='TP + BLTi'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=6,
               fontsize=8.5, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Gene expression — {gene}   [log₂(count+1)]   |   "
        f"DESeq2 Wald test p-value (uncorrected), p<{P_THRESH} shown",
        fontsize=12, fontweight='bold', y=0.98,
    )
    return fig

# ── Sidebar diagnostics ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data file status")
    st.caption(f"Data dir: `{DATA_DIR}`")
    for label, fname, _ in FILES:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label} — NOT FOUND")

    st.markdown("---")
    st.markdown("### 📊 DESeq2 results")
    st.caption(f"Dir: `{DESEQ_DIR}`")
    for _, _, key in FILES:
        p = os.path.join(DESEQ_DIR, f"DESeq2_{key}.csv")
        if os.path.exists(p):
            st.success(f"✅ DESeq2_{key}")
        else:
            st.error(f"❌ DESeq2_{key}.csv — NOT FOUND")

# ── Main UI ───────────────────────────────────────────────────────────────────
gene_list  = load_gene_list()
deseq2_tables = load_deseq2()

col1, col2 = st.columns([2, 1])
with col1:
    gene_input = st.selectbox(
        "Select or type a gene name:",
        options=[""] + gene_list,
        index=0,
        help="Type to filter. Gene names are case-sensitive (e.g. Tap1, Cxcl9)."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    plot_button = st.button("🔍 Plot gene", use_container_width=True)

if plot_button and gene_input:
    with st.spinner(f"Plotting {gene_input} across all 10 experiments…"):
        parsed = []
        missing = []
        for label, fname, deseq2_key in FILES:
            path = os.path.join(DATA_DIR, fname)
            ordered, groups = parse_gene(path, gene_input)
            if ordered is None:
                ordered, groups = [], {}
                missing.append(label)
            parsed.append((label, ordered, groups, deseq2_key))

        if all(len(p[2]) == 0 for p in parsed):
            st.error(
                f"Gene **{gene_input}** was not found in any of the 10 datasets. "
                "Please check spelling — gene names are case-sensitive."
            )
        else:
            if missing:
                st.info(f"ℹ️ Gene not found in: {', '.join(missing)}")

            # Show DESeq2 summary table for this gene
            if deseq2_tables:
                summary_rows = []
                for _, _, key in FILES:
                    if key not in deseq2_tables: continue
                    df_key = deseq2_tables[key]
                    if gene_input not in df_key.index: continue
                    row = df_key.loc[gene_input]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    for col in df_key.columns:
                        if not col.endswith('_vs_CONTR_pvalue'): continue
                        p = row[col]
                        if pd.isna(p): continue
                        treat    = col.replace('_vs_CONTR_pvalue','')
                        fc_col   = col.replace('_pvalue','_log2FC')
                        padj_col = col.replace('_pvalue','_padj')
                        summary_rows.append({
                            'Experiment': key,
                            'Comparison': f"{treat} vs CONTR",
                            'log2FC': round(float(row.get(fc_col, float('nan'))),3),
                            'p-value': float(p),
                            'padj': float(row.get(padj_col, float('nan'))),
                            'Significant*': '✅' if float(p) < P_THRESH else '',
                        })
                if summary_rows:
                    st.markdown(f"**DESeq2 results for {gene_input}** "
                                f"(* p < {P_THRESH}, uncorrected)")
                    st.dataframe(
                        pd.DataFrame(summary_rows).style.format(
                            {'p-value': '{:.4f}', 'padj': '{:.4f}', 'log2FC': '{:+.3f}'}
                        ),
                        use_container_width=True, hide_index=True
                    )

            fig = build_figure(gene_input, parsed, deseq2_tables)
            st.pyplot(fig, use_container_width=True)

            # Download buttons
            png_buf = io.BytesIO()
            svg_buf = io.BytesIO()
            fig.savefig(png_buf, format='png', dpi=180,
                        bbox_inches='tight', facecolor='white')
            fig.savefig(svg_buf, format='svg',
                        bbox_inches='tight', facecolor='white')
            plt.close(fig)

            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                st.download_button(
                    "⬇️ Download PNG", png_buf.getvalue(),
                    file_name=f"gene_expr_{gene_input}.png", mime="image/png"
                )
            with c2:
                st.download_button(
                    "⬇️ Download SVG", svg_buf.getvalue(),
                    file_name=f"gene_expr_{gene_input}.svg", mime="image/svg+xml"
                )

elif plot_button and not gene_input:
    st.warning("Please select or type a gene name first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: bulk RNAseq, 10 mouse tumor experiments (B16F10, LLC, KPCY/KPC/Panc2, CT26)  |  "
    "Drug: TP-317  |  Statistics: DESeq2 (PyDESeq2), Wald test, raw p-value  |  "
    "Platform A: ~35K genes (E8, E9, E10)  |  Platform B: ~55K genes (all others)"
)
