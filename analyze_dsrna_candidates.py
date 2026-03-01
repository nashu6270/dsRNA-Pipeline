#!/usr/bin/env python3
"""
Comprehensive dsRNA Fragment Analysis and Selection Pipeline
Generates publication-quality figures and identifies optimal candidates based on
multiple criteria relevant to RNAi efficacy and practical synthesis.

Author: Generated for dsRNA design analysis
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class dsRNAAnalyzer:
    """
    Comprehensive analysis of dsRNA fragments for optimal candidate selection.
    """

    def __init__(self, fragments_file, output_dir="dsRNA_analysis"):
        """
        Initialize analyzer.

        Args:
            fragments_file: Path to fragments TSV file
            output_dir: Output directory for results
        """
        self.fragments_file = fragments_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        print(f"Loading fragments from {fragments_file}...")
        self.df = pd.read_csv(fragments_file, sep='\t')
        print(f"Loaded {len(self.df)} fragments\n")

        # Parse JSON fields
        self._parse_json_fields()

        # Calculate derived features
        self._calculate_derived_features()

        # Initialize scoring
        self.scores = {}

    def _parse_json_fields(self):
        """Parse JSON-encoded fields."""
        # Parse dinucleotide composition
        if 'dinuc_composition_json' in self.df.columns:
            dinuc_data = self.df['dinuc_composition_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) else {}
            )
            # Extract specific dinucleotides of interest
            for dinuc in ['CG', 'GC', 'AT', 'TA', 'AA', 'TT']:
                self.df[f'dinuc_{dinuc}'] = dinuc_data.apply(
                    lambda x: x.get(dinuc, 0)
                )

    def _calculate_derived_features(self):
        """Calculate additional features for analysis."""

        # GC deviation from optimal (40-60% is generally good)
        self.df['gc_deviation_from_50'] = abs(self.df['gc_percentage'] - 50)
        self.df['gc_in_optimal_range'] = (
            (self.df['gc_percentage'] >= 40) & 
            (self.df['gc_percentage'] <= 60)
        ).astype(int)

        # Homopolymer burden
        self.df['homopolymer_burden'] = (
            self.df['homopolymer_runs_6plus'] * 1.0 + 
            self.df['homopolymer_runs_8plus'] * 3.0
        )

        # Sequence complexity score (inverse of k-mer duplication)
        self.df['complexity_score'] = 1 - self.df['kmer_duplication_rate']

        # Length category
        self.df['length_category'] = pd.cut(
            self.df['window_length'],
            bins=[0, 250, 350, 600],
            labels=['Short (200-250)', 'Medium (300-350)', 'Long (400-500)']
        )

        # Position in CDS (normalized)
        cds_length = self.df['end_pos_1based'].max()
        self.df['relative_position'] = self.df['start_pos_1based'] / cds_length
        self.df['position_category'] = pd.cut(
            self.df['relative_position'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['5\' region', 'Middle region', '3\' region']
        )

        # Binary design quality
        self.df['has_design_issues'] = (
            self.df['design_issues_str'] != 'none'
        ).astype(int)

    def calculate_composite_score(self, weights=None):
        """
        Calculate composite quality score for each fragment.

        Scoring criteria based on RNAi literature:
        1. GC content (40-60% optimal)
        2. Low homopolymer content
        3. High sequence complexity
        4. Appropriate length (300-400 bp preferred)
        5. Minimal masked region overlap

        Args:
            weights: Dictionary of feature weights (optional)
        """

        if weights is None:
            weights = {
                'gc_content': 0.25,
                'complexity': 0.25,
                'homopolymer': 0.20,
                'length': 0.15,
                'masked': 0.10,
                'design_clean': 0.05
            }

        # Normalize scores to 0-1 range

        # 1. GC content score (Gaussian centered at 50%)
        gc_score = np.exp(-((self.df['gc_percentage'] - 50) ** 2) / (2 * 10 ** 2))

        # 2. Complexity score (already 0-1)
        complexity_score = self.df['complexity_score']

        # 3. Homopolymer score (inverse, normalized)
        max_homopoly = self.df['homopolymer_burden'].max()
        if max_homopoly > 0:
            homopolymer_score = 1 - (self.df['homopolymer_burden'] / max_homopoly)
        else:
            homopolymer_score = 1.0

        # 4. Length score (prefer 300-400 bp, Gaussian)
        optimal_length = 350
        length_score = np.exp(-((self.df['window_length'] - optimal_length) ** 2) / (2 * 75 ** 2))

        # 5. Masked region score (inverse of overlap)
        masked_score = 1 - self.df['masked_overlap_fraction']

        # 6. Design clean score
        design_score = 1 - self.df['has_design_issues']

        # Composite score
        self.df['composite_score'] = (
            weights['gc_content'] * gc_score +
            weights['complexity'] * complexity_score +
            weights['homopolymer'] * homopolymer_score +
            weights['length'] * length_score +
            weights['masked'] * masked_score +
            weights['design_clean'] * design_score
        )

        # Normalize to 0-100
        self.df['composite_score_normalized'] = (
            (self.df['composite_score'] - self.df['composite_score'].min()) /
            (self.df['composite_score'].max() - self.df['composite_score'].min())
        ) * 100

        # Store individual component scores
        self.df['score_gc'] = gc_score * 100
        self.df['score_complexity'] = complexity_score * 100
        self.df['score_homopolymer'] = homopolymer_score * 100
        self.df['score_length'] = length_score * 100
        self.df['score_masked'] = masked_score * 100
        self.df['score_design'] = design_score * 100

        print("Composite scoring completed.")
        print(f"Score range: {self.df['composite_score_normalized'].min():.2f} - "
              f"{self.df['composite_score_normalized'].max():.2f}")

    def identify_top_candidates(self, n=20, min_score=70):
        """
        Identify top dsRNA candidates.

        Args:
            n: Number of top candidates to return
            min_score: Minimum composite score threshold
        """

        # Filter by minimum score
        candidates = self.df[self.df['composite_score_normalized'] >= min_score].copy()

        # Sort by composite score
        candidates = candidates.sort_values('composite_score_normalized', ascending=False)

        # Select top N
        top_candidates = candidates.head(n)

        print(f"\nIdentified {len(top_candidates)} top candidates (score >= {min_score}):")
        print(top_candidates[[
            'fragment_id', 'window_length', 'gc_percentage',
            'complexity_score', 'homopolymer_runs_6plus',
            'composite_score_normalized', 'design_issues_str'
        ]].to_string(index=False))

        return top_candidates

    def plot_overview_panel(self):
        """
        Generate comprehensive overview figure (Figure 1).
        """

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # A. GC content distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df['gc_percentage'], bins=40, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax1.axvline(50, color='red', linestyle='--', linewidth=2, label='Optimal (50%)')
        ax1.axvspan(40, 60, alpha=0.2, color='green', label='Preferred range')
        ax1.set_xlabel('GC Content (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('A. GC Content Distribution', fontweight='bold', loc='left')
        ax1.legend(frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # B. Fragment length distribution by overlap type
        ax2 = fig.add_subplot(gs[0, 1])
        overlap_data = [
            self.df[self.df['overlap_type'] == 'overlapping']['window_length'],
            self.df[self.df['overlap_type'] == 'non-overlapping']['window_length']
        ]
        ax2.hist(overlap_data, bins=5, label=['Overlapping', 'Non-overlapping'],
                color=['#FF6B6B', '#4ECDC4'], edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Fragment Length (bp)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('B. Fragment Length by Design Type', fontweight='bold', loc='left')
        ax2.legend(frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # C. Sequence complexity (k-mer duplication)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.df['complexity_score'], bins=40, color='#95E1D3',
                edgecolor='black', alpha=0.7)
        ax3.axvline(self.df['complexity_score'].median(), color='red', 
                   linestyle='--', linewidth=2, label=f"Median: {self.df['complexity_score'].median():.3f}")
        ax3.set_xlabel('Complexity Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('C. Sequence Complexity', fontweight='bold', loc='left')
        ax3.legend(frameon=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # D. Homopolymer burden
        ax4 = fig.add_subplot(gs[1, 0])
        homopoly_bins = np.arange(0, self.df['homopolymer_runs_6plus'].max() + 2, 1)
        ax4.hist(self.df['homopolymer_runs_6plus'], bins=homopoly_bins,
                color='#F38181', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Homopolymer Runs (≥6 bp)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('D. Homopolymer Burden', fontweight='bold', loc='left')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # E. Design issues breakdown
        ax5 = fig.add_subplot(gs[1, 1])
        issues = self.df['design_issues_str'].value_counts()
        colors_issues = sns.color_palette("Set2", len(issues))
        ax5.barh(range(len(issues)), issues.values, color=colors_issues, edgecolor='black')
        ax5.set_yticks(range(len(issues)))
        ax5.set_yticklabels([label.replace('_', ' ').title() for label in issues.index])
        ax5.set_xlabel('Number of Fragments', fontweight='bold')
        ax5.set_title('E. Design Quality Issues', fontweight='bold', loc='left')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        # F. Position in CDS
        ax6 = fig.add_subplot(gs[1, 2])
        position_counts = self.df['position_category'].value_counts()
        colors_pos = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        wedges, texts, autotexts = ax6.pie(
            position_counts.values,
            labels=position_counts.index,
            autopct='%1.1f%%',
            colors=colors_pos,
            startangle=90,
            textprops={'fontweight': 'bold'}
        )
        ax6.set_title('F. Distribution Along CDS', fontweight='bold', loc='left')

        # G. GC content vs Complexity scatter
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(
            self.df['gc_percentage'],
            self.df['complexity_score'],
            c=self.df['composite_score_normalized'],
            cmap='RdYlGn',
            s=30,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        ax7.axvline(50, color='gray', linestyle='--', alpha=0.5)
        ax7.axhline(self.df['complexity_score'].median(), color='gray', 
                   linestyle='--', alpha=0.5)
        ax7.set_xlabel('GC Content (%)', fontweight='bold')
        ax7.set_ylabel('Complexity Score', fontweight='bold')
        ax7.set_title('G. GC Content vs Complexity', fontweight='bold', loc='left')
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Composite Score', fontweight='bold')
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)

        # H. Length vs Score boxplot
        ax8 = fig.add_subplot(gs[2, 1])
        length_cats = self.df['length_category'].dropna().unique()
        data_for_box = [
            self.df[self.df['length_category'] == cat]['composite_score_normalized'].values
            for cat in length_cats
        ]
        bp = ax8.boxplot(data_for_box, labels=length_cats, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
        ax8.set_ylabel('Composite Score', fontweight='bold')
        ax8.set_xlabel('Fragment Length Category', fontweight='bold')
        ax8.set_title('H. Score by Length Category', fontweight='bold', loc='left')
        ax8.tick_params(axis='x', rotation=15)
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)

        # I. Composite score distribution
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.hist(self.df['composite_score_normalized'], bins=40, 
                color='#6C5CE7', edgecolor='black', alpha=0.7)
        ax9.axvline(self.df['composite_score_normalized'].median(), 
                   color='red', linestyle='--', linewidth=2,
                   label=f"Median: {self.df['composite_score_normalized'].median():.1f}")
        ax9.axvline(70, color='green', linestyle='--', linewidth=2,
                   label='Threshold (70)')
        ax9.set_xlabel('Composite Score', fontweight='bold')
        ax9.set_ylabel('Frequency', fontweight='bold')
        ax9.set_title('I. Composite Quality Score', fontweight='bold', loc='left')
        ax9.legend(frameon=False)
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)

        plt.suptitle('Comprehensive dsRNA Fragment Analysis Overview', 
                    fontsize=16, fontweight='bold', y=0.995)

        output_file = self.output_dir / 'Figure1_Overview_Panel.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()

    def plot_score_components_radar(self, top_n=10):
        """
        Generate radar plot comparing score components of top candidates (Figure 2).
        """

        # Get top candidates
        top_candidates = self.df.nlargest(top_n, 'composite_score_normalized')

        # Score components
        components = ['score_gc', 'score_complexity', 'score_homopolymer',
                     'score_length', 'score_masked', 'score_design']
        component_labels = ['GC Content', 'Complexity', 'Low Homopolymer',
                          'Optimal Length', 'Low Masked', 'Design Clean']

        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), 
                                subplot_kw=dict(projection='polar'))
        axes = axes.flatten()

        # Angles for radar plot
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        for idx, (_, row) in enumerate(top_candidates.iterrows()):
            if idx >= 10:
                break

            ax = axes[idx]

            # Get values
            values = [row[comp] for comp in components]
            values += values[:1]  # Complete the circle

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], 
                   label=row['fragment_id'])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

            # Formatting
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(component_labels, size=8)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(['25', '50', '75', '100'], size=7)
            ax.grid(True)

            # Title with fragment info
            title_text = f"{row['fragment_id'][:30]}\nScore: {row['composite_score_normalized']:.1f}"
            ax.set_title(title_text, size=9, fontweight='bold', pad=20)

        plt.suptitle(f'Score Component Analysis - Top {top_n} Candidates', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / 'Figure2_Radar_Plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_genomic_coverage_heatmap(self):
        """
        Generate heatmap showing fragment coverage across CDS (Figure 3).
        """

        # Create coverage matrix
        cds_length = self.df['end_pos_1based'].max()
        bin_size = 50  # 50 bp bins
        n_bins = int(np.ceil(cds_length / bin_size))

        # Create matrix for different length categories
        length_categories = sorted(self.df['window_length'].unique())
        coverage_matrix = np.zeros((len(length_categories), n_bins))

        for i, length in enumerate(length_categories):
            subset = self.df[self.df['window_length'] == length]
            for _, row in subset.iterrows():
                start_bin = int((row['start_pos_1based'] - 1) / bin_size)
                end_bin = int(row['end_pos_1based'] / bin_size)
                coverage_matrix[i, start_bin:end_bin+1] += 1

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Heatmap
        im = ax1.imshow(coverage_matrix, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest')
        ax1.set_yticks(range(len(length_categories)))
        ax1.set_yticklabels([f'{l} bp' for l in length_categories])
        ax1.set_ylabel('Fragment Length', fontweight='bold', fontsize=12)
        ax1.set_xlabel('CDS Position (bins of 50 bp)', fontweight='bold', fontsize=12)
        ax1.set_title('A. Fragment Coverage Across CDS by Length', 
                     fontweight='bold', fontsize=14, loc='left')

        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Number of Fragments', fontweight='bold')

        # Score distribution along CDS
        bin_scores = []
        bin_positions = []
        for bin_idx in range(n_bins):
            bin_start = bin_idx * bin_size + 1
            bin_end = (bin_idx + 1) * bin_size

            # Find fragments overlapping this bin
            overlapping = self.df[
                (self.df['start_pos_1based'] <= bin_end) &
                (self.df['end_pos_1based'] >= bin_start)
            ]

            if len(overlapping) > 0:
                bin_scores.append(overlapping['composite_score_normalized'].mean())
                bin_positions.append(bin_idx)

        ax2.plot(bin_positions, bin_scores, linewidth=2, color='#6C5CE7')
        ax2.fill_between(bin_positions, bin_scores, alpha=0.3, color='#6C5CE7')
        ax2.set_xlabel('CDS Position (bins of 50 bp)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Avg. Score', fontweight='bold', fontsize=12)
        ax2.set_title('B. Average Fragment Quality Score Along CDS', 
                     fontweight='bold', fontsize=14, loc='left')
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()

        output_file = self.output_dir / 'Figure3_Genomic_Coverage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_pca_analysis(self):
        """
        PCA analysis of fragment features (Figure 4).
        """

        # Select features for PCA
        features = [
            'gc_percentage', 'complexity_score', 'homopolymer_burden',
            'window_length', 'masked_overlap_fraction', 'relative_position'
        ]

        # Prepare data
        X = self.df[features].values
        X_scaled = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Create figure
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, wspace=0.3)

        # A. Scree plot
        ax1 = fig.add_subplot(gs[0, 0])
        variance_ratio = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_ratio)

        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, 
               alpha=0.7, color='steelblue', edgecolor='black', label='Individual')
        ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax1.axhline(y=80, color='green', linestyle='--', label='80% threshold')
        ax1.set_xlabel('Principal Component', fontweight='bold')
        ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
        ax1.set_title('A. PCA Variance Explained', fontweight='bold', loc='left')
        ax1.legend(frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # B. PC1 vs PC2 colored by composite score
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=self.df['composite_score_normalized'],
                            cmap='RdYlGn', s=30, alpha=0.6,
                            edgecolors='black', linewidth=0.5)
        ax2.set_xlabel(f'PC1 ({variance_ratio[0]:.1f}%)', fontweight='bold')
        ax2.set_ylabel(f'PC2 ({variance_ratio[1]:.1f}%)', fontweight='bold')
        ax2.set_title('B. PCA Score Plot', fontweight='bold', loc='left')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Composite Score', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # C. Feature loadings
        ax3 = fig.add_subplot(gs[0, 2])
        loadings = pca.components_[:2, :].T
        feature_labels = ['GC%', 'Complexity', 'Homopoly.', 
                         'Length', 'Masked', 'Position']

        for i, (label, loading) in enumerate(zip(feature_labels, loadings)):
            ax3.arrow(0, 0, loading[0], loading[1], 
                     head_width=0.05, head_length=0.05, 
                     fc='steelblue', ec='black', linewidth=1.5)
            ax3.text(loading[0] * 1.15, loading[1] * 1.15, label,
                    fontweight='bold', fontsize=10, ha='center', va='center')

        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_xlabel(f'PC1 Loading', fontweight='bold')
        ax3.set_ylabel(f'PC2 Loading', fontweight='bold')
        ax3.set_title('C. Feature Loadings', fontweight='bold', loc='left')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_aspect('equal')

        plt.suptitle('Principal Component Analysis of dsRNA Fragments', 
                    fontsize=16, fontweight='bold')

        output_file = self.output_dir / 'Figure4_PCA_Analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_correlation_heatmap(self):
        """
        Correlation heatmap of key features (Figure 5).
        """

        # Select features
        features = {
            'GC Content (%)': 'gc_percentage',
            'Complexity': 'complexity_score',
            'Homopolymer Burden': 'homopolymer_burden',
            'Fragment Length': 'window_length',
            'Masked Overlap': 'masked_overlap_fraction',
            'CDS Position': 'relative_position',
            'Composite Score': 'composite_score_normalized'
        }

        # Create correlation matrix
        df_subset = self.df[[v for v in features.values()]]
        df_subset.columns = features.keys()
        corr_matrix = df_subset.corr()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)

        ax.set_title('Feature Correlation Matrix', 
                    fontweight='bold', fontsize=14, pad=20)

        plt.tight_layout()

        output_file = self.output_dir / 'Figure5_Correlation_Heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_top_candidates_comparison(self, top_n=10):
        """
        Detailed comparison of top candidates (Figure 6).
        """

        top_candidates = self.df.nlargest(top_n, 'composite_score_normalized')

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # A. Composite scores
        ax1 = axes[0, 0]
        y_pos = np.arange(len(top_candidates))
        colors_gradient = plt.cm.RdYlGn(
            top_candidates['composite_score_normalized'].values / 100
        )

        bars = ax1.barh(y_pos, top_candidates['composite_score_normalized'],
                       color=colors_gradient, edgecolor='black', linewidth=1.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{fid[:30]}" for fid in top_candidates['fragment_id']], 
                           fontsize=8)
        ax1.set_xlabel('Composite Score', fontweight='bold')
        ax1.set_title('A. Top Candidates by Composite Score', 
                     fontweight='bold', loc='left')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Add score values
        for i, (bar, score) in enumerate(zip(bars, top_candidates['composite_score_normalized'])):
            ax1.text(score + 1, i, f'{score:.1f}', 
                    va='center', fontweight='bold', fontsize=8)

        # B. Feature comparison heatmap
        ax2 = axes[0, 1]
        features_to_plot = ['score_gc', 'score_complexity', 'score_homopolymer',
                           'score_length', 'score_masked', 'score_design']
        feature_labels = ['GC', 'Complex', 'Homopoly', 'Length', 'Masked', 'Design']

        heat_data = top_candidates[features_to_plot].values
        im = ax2.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

        ax2.set_xticks(range(len(feature_labels)))
        ax2.set_xticklabels(feature_labels, rotation=45, ha='right')
        ax2.set_yticks(range(len(top_candidates)))
        ax2.set_yticklabels([f"{fid[:20]}" for fid in top_candidates['fragment_id']], 
                           fontsize=8)
        ax2.set_title('B. Score Component Breakdown', fontweight='bold', loc='left')

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Component Score', fontweight='bold')

        # C. GC content and complexity
        ax3 = axes[1, 0]
        x = np.arange(len(top_candidates))
        width = 0.35

        bars1 = ax3.bar(x - width/2, top_candidates['gc_percentage'], width,
                       label='GC Content (%)', color='#4ECDC4', 
                       edgecolor='black', alpha=0.8)

        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, top_candidates['complexity_score'] * 100, width,
                            label='Complexity Score', color='#FF6B6B',
                            edgecolor='black', alpha=0.8)

        ax3.set_xlabel('Candidate Rank', fontweight='bold')
        ax3.set_ylabel('GC Content (%)', fontweight='bold', color='#4ECDC4')
        ax3_twin.set_ylabel('Complexity Score', fontweight='bold', color='#FF6B6B')
        ax3.set_title('C. GC Content vs Complexity', fontweight='bold', loc='left')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'#{i+1}' for i in range(len(top_candidates))])
        ax3.tick_params(axis='y', labelcolor='#4ECDC4')
        ax3_twin.tick_params(axis='y', labelcolor='#FF6B6B')
        ax3.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax3.spines['top'].set_visible(False)
        ax3_twin.spines['top'].set_visible(False)

        # D. Fragment positions on CDS
        ax4 = axes[1, 1]
        cds_length = self.df['end_pos_1based'].max()

        for i, (_, row) in enumerate(top_candidates.iterrows()):
            y_pos = i
            start = row['start_pos_1based']
            end = row['end_pos_1based']
            length = end - start

            color = plt.cm.RdYlGn(row['composite_score_normalized'] / 100)
            ax4.barh(y_pos, length, left=start, height=0.8,
                    color=color, edgecolor='black', linewidth=1)

        ax4.set_yticks(range(len(top_candidates)))
        ax4.set_yticklabels([f'#{i+1}' for i in range(len(top_candidates))])
        ax4.set_xlabel('CDS Position (bp)', fontweight='bold')
        ax4.set_ylabel('Candidate Rank', fontweight='bold')
        ax4.set_xlim(0, cds_length)
        ax4.set_title('D. Fragment Positions on CDS', fontweight='bold', loc='left')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.suptitle(f'Top {top_n} dsRNA Candidate Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / 'Figure6_Top_Candidates.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def generate_summary_report(self, top_n=20):
        """
        Generate comprehensive text report.
        """

        report_file = self.output_dir / 'dsRNA_Analysis_Report.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("dsRNA FRAGMENT ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("="*80 + "\n\n")

            # Dataset overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total fragments analyzed: {len(self.df)}\n")
            f.write(f"CDS length: {self.df['end_pos_1based'].max()} bp\n")
            f.write(f"Fragment length range: {self.df['window_length'].min()}-{self.df['window_length'].max()} bp\n")
            f.write(f"\nOverlapping fragments: {len(self.df[self.df['overlap_type'] == 'overlapping'])}\n")
            f.write(f"Non-overlapping fragments: {len(self.df[self.df['overlap_type'] == 'non-overlapping'])}\n")
            f.write(f"\nFragments with design issues: {len(self.df[self.df['has_design_issues'] == 1])}\n")
            f.write(f"Clean fragments: {len(self.df[self.df['has_design_issues'] == 0])}\n\n")

            # Quality metrics
            f.write("2. QUALITY METRICS SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"GC Content:\n")
            f.write(f"  Mean: {self.df['gc_percentage'].mean():.2f}%\n")
            f.write(f"  Std: {self.df['gc_percentage'].std():.2f}%\n")
            f.write(f"  Range: {self.df['gc_percentage'].min():.2f}% - {self.df['gc_percentage'].max():.2f}%\n")
            f.write(f"  Fragments in optimal range (40-60%): {self.df['gc_in_optimal_range'].sum()}\n\n")

            f.write(f"Sequence Complexity:\n")
            f.write(f"  Mean complexity score: {self.df['complexity_score'].mean():.3f}\n")
            f.write(f"  Median: {self.df['complexity_score'].median():.3f}\n")
            f.write(f"  Fragments with high complexity (>0.9): {len(self.df[self.df['complexity_score'] > 0.9])}\n\n")

            f.write(f"Homopolymer Analysis:\n")
            f.write(f"  Fragments with ≥1 homopolymer run (≥6bp): {len(self.df[self.df['homopolymer_runs_6plus'] > 0])}\n")
            f.write(f"  Fragments with ≥1 homopolymer run (≥8bp): {len(self.df[self.df['homopolymer_runs_8plus'] > 0])}\n")
            f.write(f"  Mean homopolymer burden: {self.df['homopolymer_burden'].mean():.2f}\n\n")

            f.write(f"Composite Quality Score:\n")
            f.write(f"  Mean: {self.df['composite_score_normalized'].mean():.2f}\n")
            f.write(f"  Median: {self.df['composite_score_normalized'].median():.2f}\n")
            f.write(f"  Std: {self.df['composite_score_normalized'].std():.2f}\n")
            f.write(f"  Range: {self.df['composite_score_normalized'].min():.2f} - {self.df['composite_score_normalized'].max():.2f}\n")
            f.write(f"  High-quality candidates (score ≥ 70): {len(self.df[self.df['composite_score_normalized'] >= 70])}\n\n")

            # Top candidates
            f.write(f"3. TOP {top_n} CANDIDATE dsRNA FRAGMENTS\n")
            f.write("-" * 80 + "\n\n")

            top_candidates = self.df.nlargest(top_n, 'composite_score_normalized')

            for rank, (_, row) in enumerate(top_candidates.iterrows(), 1):
                f.write(f"RANK #{rank}\n")
                f.write(f"  Fragment ID: {row['fragment_id']}\n")
                f.write(f"  Position: {row['start_pos_1based']}-{row['end_pos_1based']} bp\n")
                f.write(f"  Length: {row['window_length']} bp\n")
                f.write(f"  Overlap type: {row['overlap_type']}\n")
                f.write(f"  \n")
                f.write(f"  Composite Score: {row['composite_score_normalized']:.2f}\n")
                f.write(f"    - GC content: {row['gc_percentage']:.2f}% (score: {row['score_gc']:.1f})\n")
                f.write(f"    - Complexity: {row['complexity_score']:.3f} (score: {row['score_complexity']:.1f})\n")
                f.write(f"    - Homopolymer: {row['homopolymer_runs_6plus']} runs (score: {row['score_homopolymer']:.1f})\n")
                f.write(f"    - Length optimality: (score: {row['score_length']:.1f})\n")
                f.write(f"    - Masked overlap: {row['masked_overlap_percentage']:.2f}% (score: {row['score_masked']:.1f})\n")
                f.write(f"  \n")
                f.write(f"  Design issues: {row['design_issues_str']}\n")
                f.write(f"  Sequence (first 60 bp): {row['sequence'][:60]}...\n")
                f.write("\n" + "-" * 80 + "\n\n")

            # Recommendations
            f.write("4. RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            # Best overall
            best = self.df.loc[self.df['composite_score_normalized'].idxmax()]
            f.write(f"Best overall candidate: {best['fragment_id']}\n")
            f.write(f"  Score: {best['composite_score_normalized']:.2f}\n")
            f.write(f"  Position: {best['start_pos_1based']}-{best['end_pos_1based']} bp\n")
            f.write(f"  Length: {best['window_length']} bp\n\n")

            # Best by length
            f.write("Best candidates by length category:\n")
            for length in sorted(self.df['window_length'].unique()):
                subset = self.df[self.df['window_length'] == length]
                best_in_cat = subset.loc[subset['composite_score_normalized'].idxmax()]
                f.write(f"  {length} bp: {best_in_cat['fragment_id']} ")
                f.write(f"(score: {best_in_cat['composite_score_normalized']:.2f})\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"Saved: {report_file}")

    def save_ranked_candidates(self):
        """
        Save ranked candidates to CSV.
        """

        # Select key columns
        output_columns = [
            'fragment_id', 'gene_name', 'start_pos_1based', 'end_pos_1based',
            'window_length', 'overlap_type', 'composite_score_normalized',
            'score_gc', 'score_complexity', 'score_homopolymer', 
            'score_length', 'score_masked', 'score_design',
            'gc_percentage', 'complexity_score', 'homopolymer_runs_6plus',
            'homopolymer_runs_8plus', 'kmer_duplication_rate',
            'masked_overlap_percentage', 'design_issues_str',
            'position_category', 'sequence'
        ]

        # Sort by composite score
        ranked_df = self.df.sort_values('composite_score_normalized', 
                                        ascending=False)[output_columns]

        # Save
        output_file = self.output_dir / 'ranked_dsRNA_candidates.csv'
        ranked_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

        # Also save top 50 separately
        top50_file = self.output_dir / 'top50_candidates.csv'
        ranked_df.head(50).to_csv(top50_file, index=False)
        print(f"Saved: {top50_file}")

    def run_complete_analysis(self, top_n=20):
        """
        Run complete analysis pipeline.
        """

        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE dsRNA FRAGMENT ANALYSIS")
        print("="*80 + "\n")

        # Calculate scores
        print("Step 1: Calculating composite quality scores...")
        self.calculate_composite_score()

        # Identify top candidates
        print("\nStep 2: Identifying top candidates...")
        top_candidates = self.identify_top_candidates(n=top_n, min_score=70)

        # Generate figures
        print("\nStep 3: Generating publication-quality figures...")
        print("  - Figure 1: Overview panel...")
        self.plot_overview_panel()

        print("  - Figure 2: Score component radar plots...")
        self.plot_score_components_radar(top_n=10)

        print("  - Figure 3: Genomic coverage heatmap...")
        self.plot_genomic_coverage_heatmap()

        print("  - Figure 4: PCA analysis...")
        self.plot_pca_analysis()

        print("  - Figure 5: Correlation heatmap...")
        self.plot_correlation_heatmap()

        print("  - Figure 6: Top candidates comparison...")
        self.plot_top_candidates_comparison(top_n=10)

        # Generate reports
        print("\nStep 4: Generating summary reports...")
        self.generate_summary_report(top_n=top_n)
        self.save_ranked_candidates()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  - 6 publication-quality figures (PNG + PDF)")
        print("  - Comprehensive text report")
        print("  - Ranked candidates CSV")
        print("  - Top 50 candidates CSV")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of dsRNA fragments for candidate selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input fragments TSV file')
    parser.add_argument('-o', '--output-dir', default='dsRNA_analysis',
                       help='Output directory (default: dsRNA_analysis)')
    parser.add_argument('-n', '--top-n', type=int, default=20,
                       help='Number of top candidates to report (default: 20)')

    args = parser.parse_args()

    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    # Run analysis
    analyzer = dsRNAAnalyzer(args.input, args.output_dir)
    analyzer.run_complete_analysis(top_n=args.top_n)


if __name__ == "__main__":
    main()
