#!/usr/bin/env python3
"""
compute_structural_features.py

ViennaRNA wrapper: compute MFE, ensemble metrics and accessibility (RNAplfold)
for each fragment in fragments.tsv, and generate visualizations.

Inputs:
  - fragments.tsv : must include columns 'fragment_id' and 'sequence'
Outputs:
  - fragments.features.tsv : input rows + new structural feature columns
  - PNG plots in output directory

Dependencies:
  - Python libs: pandas, numpy, matplotlib, seaborn, tqdm, biopython (optional)
  - ViennaRNA command-line tools: RNAfold, RNAplfold
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Utility helpers -------------------------------------------------------

def check_tool(name):
    """Return True if executable exists in PATH."""
    return shutil.which(name) is not None

def run_cmd(cmd, input_text=None, cwd=None):
    """Run shell command list and return stdout, stderr, returncode."""
    proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, cwd=cwd)
    return proc.stdout, proc.stderr, proc.returncode

# --- ViennaRNA wrappers ----------------------------------------------------

def rnafold_with_ensemble(seq):
    """
    Run RNAfold -p --noPS on sequence (single sequence).
    Returns: dict with mfe (float), structure (dot-bracket string),
             free_energy_ensemble (float or None),
             ensemble_diversity (float or None)
    """
    # Prepare input as FASTA-less (plain seq) to RNAfold
    cmd = ["RNAfold", "-p", "--noPS"]
    stdout, stderr, rc = run_cmd(cmd, input_text=seq + "\n")
    if rc != 0:
        raise RuntimeError(f"RNAfold failed: {stderr.strip()}")

    # stdout format (example):
    # SEQUENCE
    # ((((...)))) ( -12.30)
    # free energy of ensemble = -11.22 kcal/mol
    # frequency of mfe structure in ensemble = 0.08
    # ensemble diversity = 6.12

    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    res = {"mfe": None, "structure": None, "free_energy_ensemble": None, "ensemble_diversity": None}

    if len(lines) >= 2:
        # parse structure + mfe from second non-empty line
        # e.g. "((((....)))) (-12.30)"
        struct_line = lines[1]
        m = re.search(r"^([().,\[\]\-A-Za-z_]+)\s+\(?\s*([\-0-9\.]+)\s*\)?", struct_line)
        if m:
            res["structure"] = m.group(1)
            try:
                res["mfe"] = float(m.group(2))
            except:
                res["mfe"] = None

    # parse ensemble free energy and diversity
    for line in lines[2:]:
        if "free energy of ensemble" in line:
            m = re.search(r"free energy of ensemble\s*=\s*([\-0-9\.]+)", line)
            if m:
                res["free_energy_ensemble"] = float(m.group(1))
        if "ensemble diversity" in line:
            m = re.search(r"ensemble diversity\s*=\s*([0-9\.]+)", line)
            if m:
                res["ensemble_diversity"] = float(m.group(1))

    return res

def rnaplfold_unpaired_probs(seq, max_u=8):
    """
    Run RNAplfold on seq in a temporary directory and parse the produced lunp file.
    Returns:
      - lunp_matrix: 2D numpy array shape (L, max_u) where lunp_matrix[i, u-1]
                     is probability that length-u segment starting at pos i (1-based) is unpaired.
      - available_u: maximum u found in lunp file (int)
    On failure returns (None, 0).
    """
    # Create a temporary working directory (RNAplfold writes several files)
    with tempfile.TemporaryDirectory() as td:
        # write sequence as single-line FASTA (RNAplfold expects plain seq on stdin too)
        prefix = "plfold_temp"
        fasta_path = os.path.join(td, f"{prefix}.fa")
        with open(fasta_path, "w") as fh:
            fh.write(f">{prefix}\n")
            # RNAplfold expects U for uracil; convert T->U
            fh.write(seq.replace("T", "U").replace("t", "u") + "\n")

        # RNAplfold parameters: set W = L, L = L so full-length context is used
        L = len(seq)
        W = max(L, 1)
        # call RNAplfold, input file via stdin redirection by reading file content in Python and piping
        cmd = ["RNAplfold", "-W", str(W), "-L", str(L), "-u", str(max_u)]
        with open(fasta_path, "r") as fh:
            stdout, stderr, rc = run_cmd(cmd, input_text=fh.read(), cwd=td)

        if rc != 0:
            # RNAplfold failed
            return None, 0

        # find lunp-like file in td. Different ViennaRNA versions produce names like:
        # <prefix>_lunp or <prefix>.lunp or plfold_lunp etc. We'll search for any file containing 'lunp' (case-insensitive).
        candidates = []
        for p in os.listdir(td):
            if 'lunp' in p.lower() or p.lower().endswith('.lunp'):
                candidates.append(os.path.join(td, p))

        if not candidates:
            # Sometimes RNAplfold produces files like <prefix>_uX or similar; try any file ending with "_u*"
            candidates = [os.path.join(td, p) for p in os.listdir(td) if p.startswith(prefix)]
            if not candidates:
                return None, 0

        # Pick the most recently modified candidate file
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        lunp_file = candidates[0]

        # parse lunp file: expected tabular format, possibly with comment/headers
        # Format guess: first column position, then columns for u=1..U
        try:
            with open(lunp_file, "r") as fh:
                lines = [l.strip() for l in fh if l.strip() and not l.startswith('#')]

            # Skip any header lines that don't start with a digit
            data_lines = [l for l in lines if re.match(r'^\s*\d+', l)]
            if not data_lines:
                return None, 0

            parsed = []
            for line in data_lines:
                parts = re.split(r'\s+', line)
                # first part is position; remaining are floats (u=1..)
                numeric = []
                for x in parts[1:]:
                    try:
                        numeric.append(float(x))
                    except:
                        numeric.append(np.nan)
                parsed.append(numeric)

            # If parsed is empty return None
            if len(parsed) == 0:
                return None, 0

            matrix = np.array(parsed)  # shape (L', Ufound)
            available_u = matrix.shape[1]
            # If the file is shorter than sequence length, pad with NaNs
            if matrix.shape[0] < L:
                pad_rows = L - matrix.shape[0]
                matrix = np.vstack([matrix, np.full((pad_rows, matrix.shape[1]), np.nan)])

            return matrix, available_u

        except Exception as e:
            return None, 0

# --- Feature computation ---------------------------------------------------

def compute_features_for_fragment(fragment_id, seq, seed_start=5, seed_end=10, max_u=8):
    """
    Compute structural features for a single fragment sequence.
    Returns a dict of features.
    seed_start/seed_end are 1-based positions (relative to fragment) to compute a seed accessibility region.
    """
    seq = seq.strip().upper().replace('T', 'U')  # ViennaRNA expects U
    L = len(seq)

    features = {
        "fragment_id": fragment_id,
        "length": L,
        "mfe_kcal": None,
        "structure": None,
        "free_energy_ensemble": None,
        "ensemble_diversity": None,
        "mean_unpaired_u1": None,
        "mean_unpaired_uk": None,  # where k = seed length
        "seed_accessibility_mean": None,
        "seed_accessibility_median": None
    }

    # 1) RNAfold with ensemble
    try:
        res = rnafold_with_ensemble(seq)
        features["mfe_kcal"] = res.get("mfe", None)
        features["structure"] = res.get("structure", None)
        features["free_energy_ensemble"] = res.get("free_energy_ensemble", None)
        features["ensemble_diversity"] = res.get("ensemble_diversity", None)
    except Exception as e:
        # leave values as None, but continue
        pass

    # 2) RNAplfold unpaired probabilities (u=1..max_u)
    try:
        lunp_matrix, available_u = rnaplfold_unpaired_probs(seq, max_u=max_u)
        if lunp_matrix is not None:
            # mean unpaired prob for u=1 is column 0
            features["mean_unpaired_u1"] = float(np.nanmean(lunp_matrix[:, 0]))
            # compute seed k (length)
            seed_k = max(1, seed_end - seed_start + 1)
            if seed_k <= available_u:
                # mean unpaired probability for segments of length seed_k across the entire fragment
                features["mean_unpaired_uk"] = float(np.nanmean(lunp_matrix[:, seed_k - 1]))
                # compute seed region positions (convert from 1-based to 0-based indices)
                s0 = max(0, seed_start - 1)
                e0 = min(L - 1, seed_end - 1)
                if s0 <= e0:
                    # We consider start positions within [s0..e0] and take lunp for seed_k
                    seed_slice = lunp_matrix[s0:e0+1, seed_k - 1]
                    features["seed_accessibility_mean"] = float(np.nanmean(seed_slice))
                    features["seed_accessibility_median"] = float(np.nanmedian(seed_slice))
            else:
                # no column for seed_k available
                features["mean_unpaired_uk"] = None
    except Exception as e:
        pass

    return features

# --- Visualization helpers -------------------------------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_histogram(series, title, xlabel, outpath, bins=40):
    plt.figure(figsize=(6,4))
    sns.histplot(series.dropna(), bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_scatter(x, y, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=x, y=y, edgecolor=None, s=40, alpha=0.7)
    sns.regplot(x=x, y=y, scatter=False, lowess=True, color='gray')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_box(series, title, ylabel, outpath):
    plt.figure(figsize=(5,4))
    sns.boxplot(data=series.dropna(), orient='v')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_heatmap_perposition(matrix, x_labels, title, outpath, vmax=None):
    """
    matrix: 2D numpy array (fragments x positions) maybe with np.nan
    x_labels: labels for x-axis (positions)
    """
    plt.figure(figsize=(10, max(4, matrix.shape[0]*0.15)))
    ax = sns.heatmap(matrix, cmap="viridis", cbar_kws={'label': 'unpaired probability'}, vmax=vmax)
    ax.set_xlabel("position (relative to fragment start)")
    ax.set_ylabel("fragment (rows)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute RNA structural features using ViennaRNA (RNAfold, RNAplfold).")
    parser.add_argument("-i", "--input", required=True, help="Input fragments TSV (from dsrna_sliding_window.py)")
    parser.add_argument("-o", "--output", required=True, help="Output TSV with structural features (e.g., fragments.features.tsv)")
    parser.add_argument("-p", "--plots", default="structural_plots", help="Output directory for plots")
    parser.add_argument("--seed-start", type=int, default=5, help="Seed start position (1-based, relative to fragment)")
    parser.add_argument("--seed-end", type=int, default=10, help="Seed end position (1-based, relative to fragment)")
    parser.add_argument("--max-u", type=int, default=8, help="max u (length) for RNAplfold unpaired probs")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary RNAplfold directories (for debugging)")
    parser.add_argument("--threads", type=int, default=1, help="(not used) placeholder for future parallel runs")
    args = parser.parse_args()

    # Check ViennaRNA tools
    if not check_tool("RNAfold"):
        print("ERROR: RNAfold not found in PATH. Please install ViennaRNA and ensure RNAfold is on PATH.", file=sys.stderr)
        sys.exit(1)
    if not check_tool("RNAplfold"):
        print("ERROR: RNAplfold not found in PATH. Please install ViennaRNA and ensure RNAplfold is on PATH.", file=sys.stderr)
        sys.exit(1)

    # Read fragments table
    df = pd.read_csv(args.input, sep='\t', dtype=str)
    # ensure necessary columns
    if 'fragment_id' not in df.columns or 'sequence' not in df.columns:
        print("ERROR: Input TSV must contain columns 'fragment_id' and 'sequence'.", file=sys.stderr)
        sys.exit(1)

    # Prepare output dir for plots
    ensure_dir(args.plots)

    # Prepare containers for per-fragment per-position u=1 arrays (for heatmap)
    frag_ids = []
    perpos_unpaired_u1 = []  # list of numpy arrays (len = fragment length)
    results = []

    print(f"Processing {len(df)} fragments with ViennaRNA (this may take some time)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fid = str(row['fragment_id'])
        seq_orig = str(row['sequence']).strip().upper()
        # convert T->U for RNA tools
        seq = seq_orig.replace('T', 'U').replace('t', 'u')

        feats = compute_features_for_fragment(fid, seq, seed_start=args.seed_start, seed_end=args.seed_end, max_u=args.max_u)
        results.append(feats)
        frag_ids.append(fid)

        # try to re-run RNAplfold one more time to get u=1 vector for heatmap (we already computed within function;
        # but because that function used temporary directories we can't retrieve lunp matrix here without modifying function.
        # Instead, rerun a lightweight RNAplfold invocation that dumps u=1 column into a temporary file we then parse.)
        try:
            lunp_matrix, available_u = rnaplfold_unpaired_probs(seq, max_u=1)
            if lunp_matrix is not None:
                # lunp_matrix shape (L, 1)
                vec = lunp_matrix[:, 0]
                # Ensure length matches fragment length
                if len(vec) < len(seq):
                    vec = np.concatenate([vec, np.full(len(seq) - len(vec), np.nan)])
                perpos_unpaired_u1.append(vec)
            else:
                perpos_unpaired_u1.append(np.full(len(seq), np.nan))
        except Exception:
            perpos_unpaired_u1.append(np.full(len(seq), np.nan))

    # Build features DataFrame
    feats_df = pd.DataFrame(results)
    # Merge with original df (left join on fragment_id)
    merged = df.merge(feats_df, on='fragment_id', how='left')

    # Save features TSV
    merged.to_csv(args.output, sep='\t', index=False)
    print(f"Saved features to: {args.output}")

    # --- Visualizations ----------------------------------------------------

    # Convert some fields to numeric
    merged['mfe_kcal'] = pd.to_numeric(merged['mfe_kcal'], errors='coerce')
    # If GC column exists use it, else compute
    if 'gc_percentage' in merged.columns:
        merged['gc_pct'] = pd.to_numeric(merged['gc_percentage'], errors='coerce')
    else:
        # compute GC% simple
        def compute_gc_pct(s):
            s = str(s).upper()
            s = re.sub(r'[^ACGTU]', '', s)
            if len(s) == 0:
                return np.nan
            g = s.count('G') + s.count('C')
            return 100.0 * g / len(s)
        merged['gc_pct'] = merged['sequence'].apply(compute_gc_pct)

    # Histograms
    plot_histogram(merged['mfe_kcal'], "MFE distribution (kcal/mol)", "MFE (kcal/mol)", os.path.join(args.plots, "hist_mfe.png"))
    plot_histogram(merged['gc_pct'], "GC% distribution", "GC %", os.path.join(args.plots, "hist_gc.png"))

    # Scatter MFE vs GC
    plot_scatter(merged['gc_pct'], merged['mfe_kcal'], "GC %", "MFE (kcal/mol)", "MFE vs GC%", os.path.join(args.plots, "mfe_vs_gc.png"))

    # Accessibility boxplot (seed)
    if 'seed_accessibility_mean' in merged.columns:
        merged['seed_accessibility_mean'] = pd.to_numeric(merged['seed_accessibility_mean'], errors='coerce')
        plot_box(merged['seed_accessibility_mean'], "Seed (positions {}-{}) accessibility (mean)".format(args.seed_start, args.seed_end),
                 "mean unpaired probability", os.path.join(args.plots, "seed_accessibility_box.png"))

    # Heatmap of per-position unpaired probability (u=1) across fragments
    # Build ragged matrix -> pad to max length with NaN
    max_len = max([len(arr) for arr in perpos_unpaired_u1]) if len(perpos_unpaired_u1) > 0 else 0
    heatmat = np.full((len(perpos_unpaired_u1), max_len), np.nan)
    for i, arr in enumerate(perpos_unpaired_u1):
        heatmat[i, :len(arr)] = arr

    if heatmat.size > 0:
        # sort fragments by mean unpaired to make heatmap interpretable
        row_order = np.argsort(np.nanmean(heatmat, axis=1))[::-1]
        heatmat_sorted = heatmat[row_order, :]
        # x labels: positions 1..max_len (but show ticks every 10 or suitable)
        x_labels = list(range(1, max_len+1))
        plot_heatmap_perposition(heatmat_sorted, x_labels, "Per-position unpaired probability (u=1) across fragments",
                                 os.path.join(args.plots, "heatmap_unpaired_u1.png"), vmax=1.0)

    # Save a small summary CSV with key fields for quick review
    summary_cols = ['fragment_id', 'length', 'gc_pct', 'mfe_kcal', 'ensemble_diversity', 'mean_unpaired_u1', 'mean_unpaired_uk', 'seed_accessibility_mean', 'design_issues_str'] if 'design_issues_str' in merged.columns else ['fragment_id', 'length', 'gc_pct', 'mfe_kcal', 'ensemble_diversity', 'mean_unpaired_u1', 'mean_unpaired_uk', 'seed_accessibility_mean']
    summary = merged[[c for c in summary_cols if c in merged.columns]].copy()
    summary_path = os.path.splitext(args.output)[0] + ".summary.tsv"
    summary.to_csv(summary_path, sep='\t', index=False)
    print(f"Saved summary to: {summary_path}")

    print("All plots saved to directory:", args.plots)
    print("Done.")

if __name__ == "__main__":
    main()
