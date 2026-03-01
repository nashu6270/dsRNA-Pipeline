# dsRNA-Pipeline
Modular Python pipeline for in silico dsRNA design, structural annotation, and multi-criteria ranking for RNAi-based pest management.

dsRNA-Pipeline is a fully scriptable, command-line Python toolkit for the rational design, structural annotation, and multi-criteria ranking of double-stranded RNA (dsRNA) candidates from target insect gene coding sequences. It is intended to support the in silico phase of RNAi-based bioinsecticide development by reducing the experimental search space to a shortlist of high-confidence dsRNA constructs before any synthesis is attempted.

What it does
The pipeline consists of three sequential, modular scripts:

dsrna_sliding_window.py — Tiles a user-supplied CDS with overlapping and non-overlapping sliding windows across multiple configurable lengths (default: 200–500 bp) and step sizes. Each fragment is annotated with GC content, dinucleotide/trinucleotide composition, homopolymer burden (≥6 and ≥8 bp runs), 21-mer k-mer duplication rate, and masked-region overlap flags.

compute_structural_features.py — Wraps ViennaRNA tools (RNAfold + RNAplfold) to compute per-fragment thermodynamic and structural descriptors: minimum free energy (MFE), ensemble free energy, ensemble diversity, position-wise unpaired probabilities, and seed-region accessibility scores.

analyze_dsrna_candidates.py — Integrates sequence and structural features into a transparent, weight-configurable composite quality score. Outputs a fully ranked candidate table, a top-50 shortlist, a comprehensive text report, and six publication-quality figures (overview panel, radar plots, genomic coverage heatmap, PCA, correlation heatmap, and top-candidate profiles).

Key features
Fully local and CLI-driven — no web server dependency

Transparent, auditable multi-criteria scoring with per-component score traceability

Masked-region aware fragment generation

Publication-ready figure generation (300 DPI, PNG + PDF)

Designed for integration into HPC workflows and larger bioinformatics pipelines
# Licensed under the PolyForm Noncommercial License 1.0.0
# © 2026 A. Nayak, C. Bose, S. Nanda. Commercial use requires written authorization.


Dependencies
Python ≥ 3.8 · Biopython · pandas · numpy · matplotlib · seaborn · scikit-learn · scipy · tqdm · ViennaRNA (RNAfold, RNAplfold)
