

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/) (RNAfold + RNAplfold must be in PATH)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/dsRNA-Pipeline.git
cd dsRNA-Pipeline
```


### Step 2 — Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate dsrna-pipeline
```


### Step 3 — Install ViennaRNA (if not already installed)

```bash
conda install -c bioconda viennarna
```


---

## 🚀 How to Run the Pipeline

The pipeline runs in **three sequential steps**.
Input: a target gene CDS in FASTA format.

---

### Step 1 — Generate dsRNA Candidate Fragments

```bash
python dsrna_sliding_window.py \
    -i your_gene.fasta \
    -o fragments.tsv \
    -l 200 250 300 400 500 \
    -s 25 50
```

| Argument | Description |
| :-- | :-- |
| `-i` | Input CDS sequence (FASTA format) |
| `-o` | Output file for fragments (TSV) |
| `-l` | Window lengths in bp (default: 200 250 300 400 500) |
| `-s` | Step sizes for overlapping windows (default: 25 50) |
| `-m` | *(Optional)* Masked regions file (tab-delimited: start end) |
| `-t` | *(Optional)* Masked region overlap tolerance (default: 0.1) |


---

### Step 2 — Compute Structural Features (requires ViennaRNA)

```bash
python compute_structural_features.py \
    -i fragments.tsv \
    -o fragments.features.tsv \
    -p structural_plots/
```

| Argument | Description |
| :-- | :-- |
| `-i` | Input fragments TSV (from Step 1) |
| `-o` | Output TSV with structural features |
| `-p` | Output directory for structural plots |
| `--seed-start` | Seed region start position (default: 5) |
| `--seed-end` | Seed region end position (default: 10) |
| `--max-u` | Max unpaired length for RNAplfold (default: 8) |


---

### Step 3 — Rank, Score \& Visualize Candidates

```bash
python analyze_dsrna_candidates.py \
    -i fragments.features.tsv \
    -o dsRNA_analysis/ \
    -n 20
```

| Argument | Description |
| :-- | :-- |
| `-i` | Input features TSV (from Step 2) |
| `-o` | Output directory for results and figures |
| `-n` | Number of top candidates to report (default: 20) |


---

## 📁 Output Files

After running all three steps, the output directory will contain:

```
dsRNA_analysis/
├── ranked_dsRNA_candidates.csv   ← All fragments, fully ranked
├── top50_candidates.csv          ← Top 50 shortlist
├── dsRNA_Analysis_Report.txt     ← Comprehensive text report
├── Figure1_Overview_Panel.pdf    ← Quality metrics overview
├── Figure2_Radar_Plots.pdf       ← Score components per top candidate
├── Figure3_Genomic_Coverage.pdf  ← Coverage heatmap along CDS
├── Figure4_PCA_Analysis.pdf      ← PCA of fragment features
├── Figure5_Correlation_Heatmap.pdf ← Feature correlation matrix
└── Figure6_Top_Candidates.pdf    ← Multi-dimensional efficacy profile
```


---

## ⚡ Quick Run (All Steps in One)

```bash
# Step 1
python dsrna_sliding_window.py -i gene.fasta -o fragments.tsv

# Step 2
python compute_structural_features.py -i fragments.tsv -o fragments.features.tsv

# Step 3
python analyze_dsrna_candidates.py -i fragments.features.tsv -o results/
```


---

The env.yml file contains all required dependencies.
Simply run:
Step 1
bash
conda env create -f env.yml
This will automatically install all Python packages
(pandas, numpy, biopython, matplotlib, seaborn,
scikit-learn, scipy, tqdm, etc.)

Step 3 — Activate the Environment
bash
conda activate dsRNA-pipeline
⚠️ Always activate this environment before running any script.

Step 4 — Install ViennaRNA (if not already in env.yml)
bash
conda install -c bioconda viennarna
Step 5 — Verify Installation
bash
python -c "import pandas, numpy, Bio, sklearn; print('All packages OK')"
RNAfold --version
RNAplfold --version
If all commands return without errors, you are ready to run the pipeline.

🔄 Update Environment (if env.yml changes in future)
bash
conda env update -f env.yml --prune
❌ Remove Environment (if needed)
bash
conda deactivate
conda env remove -n dsRNA-pipeline

## 📌 Citation

If you use this pipeline in your research, please cite:
> Nayak A., Bose C., Nanda S. (2026). dsRNA-Pipeline: A modular in silico
> dsRNA design and ranking toolkit for RNAi-based pest management.
> GitHub: https://github.com/YOUR_USERNAME/dsRNA-Pipeline

```

***

