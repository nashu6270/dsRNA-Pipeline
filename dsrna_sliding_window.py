#!/usr/bin/env python3
"""
Sliding Window dsRNA Design Variant Generator
Generates comprehensive sets of dsRNA candidates with configurable window lengths
and step sizes, with quality control and sequence annotation.
*********************Done By  C Bose , A Nayak*************************************
"""

from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from collections import Counter
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json


class dsRNADesigner:
    """
    Generate and annotate dsRNA design variants using sliding window approach.
    """

    def __init__(self,
                 cds_sequence: str,
                 gene_name: str,
                 masked_regions: Optional[List[Tuple[int, int]]] = None,
                 mask_tolerance: float = 0.1):
        """
        Initialize the dsRNA designer.

        Args:
            cds_sequence: CDS sequence string (A/C/G/T/N allowed)
            gene_name: Gene identifier for naming fragments
            masked_regions: List of (start, end) tuples for masked regions (1-based, inclusive)
            mask_tolerance: Maximum fraction of window overlap with masked regions (default 0.1 = 10%)
        """
        self.cds_sequence = cds_sequence.upper()
        self.cds_length = len(self.cds_sequence)
        self.gene_name = gene_name
        self.masked_regions = masked_regions if masked_regions else []
        self.mask_tolerance = mask_tolerance
        self.fragments = []

    def check_masked_overlap(self, start: int, end: int) -> Tuple[bool, float]:
        """
        Check if a window overlaps with masked regions beyond tolerance.

        Args:
            start: Window start position (0-based)
            end: Window end position (0-based, exclusive)

        Returns:
            (passes_check, overlap_fraction)
        """
        window_length = end - start
        overlap_bases = 0

        # Convert to 1-based for comparison with masked regions (inclusive)
        window_start_1based = start + 1
        window_end_1based = end  # because end was exclusive 0-based, this equals inclusive 1-based end

        for mask_start, mask_end in self.masked_regions:
            # Calculate overlap (inclusive intervals)
            overlap_start = max(window_start_1based, mask_start)
            overlap_end = min(window_end_1based, mask_end)

            if overlap_start <= overlap_end:
                overlap_bases += (overlap_end - overlap_start + 1)

        overlap_fraction = overlap_bases / window_length if window_length > 0 else 0
        passes = overlap_fraction <= self.mask_tolerance

        return passes, overlap_fraction

    def count_homopolymers(self, sequence: str, min_length: int = 6) -> Dict[str, int]:
        """
        Count homopolymeric runs of specified minimum length.

        Args:
            sequence: DNA sequence
            min_length: Minimum homopolymer length to flag

        Returns:
            Dictionary of base -> count of runs >= min_length
        """
        seq = sequence.upper()
        homopolymers = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

        for base in 'ACGT':
            # pattern like A{6,} - will find runs of 6 or more
            pattern = rf"{base}{{{min_length},}}"
            matches = re.findall(pattern, seq)
            homopolymers[base] = len(matches)

        return homopolymers

    def calculate_dinucleotide_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate dinucleotide frequencies (only A/C/G/T pairs)."""
        seq = re.sub('[^ACGT]', '', sequence.upper())
        dinucs = [seq[i:i+2] for i in range(len(seq)-1)]
        total = len(dinucs)
        counts = Counter(dinucs)
        return {k: v/total for k, v in counts.items()} if total > 0 else {}

    def calculate_trinucleotide_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate trinucleotide frequencies (only A/C/G/T triplets)."""
        seq = re.sub('[^ACGT]', '', sequence.upper())
        trinucs = [seq[i:i+3] for i in range(len(seq)-2)]
        total = len(trinucs)
        counts = Counter(trinucs)
        return {k: v/total for k, v in counts.items()} if total > 0 else {}

    def count_kmer_duplicates(self, sequence: str, k: int = 21) -> Tuple[int, int]:
        """
        Count k-mer duplications within the sequence.

        Args:
            sequence: DNA sequence
            k: k-mer length (default 21 for siRNA-relevant size)

        Returns:
            (total_kmers, duplicate_kmer_types)
        """
        seq = re.sub('[^ACGT]', '', sequence.upper())
        if len(seq) < k:
            return 0, 0

        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        kmer_counts = Counter(kmers)

        total_kmers = len(kmers)
        # number of distinct k-mers that appear more than once
        duplicate_kmer_types = sum(1 for count in kmer_counts.values() if count > 1)

        return total_kmers, duplicate_kmer_types

    def annotate_fragment(self, fragment_seq: str, fragment_id: str,
                         start_pos: int, end_pos: int,
                         overlap_type: str, mask_overlap_fraction: float) -> Dict:
        """
        Annotate a dsRNA fragment with sequence features.

        Args:
            fragment_seq: Fragment sequence
            fragment_id: Fragment identifier
            start_pos: Start position (1-based, inclusive)
            end_pos: End position (1-based, inclusive)
            overlap_type: "overlapping" or "non-overlapping"
            mask_overlap_fraction: Fraction of fragment overlapping masked regions

        Returns:
            Dictionary of annotations
        """
        # Primary sequence metrics (compute GC on A/C/G/T only)
        analysis_seq = re.sub('[^ACGT]', '', fragment_seq.upper())
        length = len(fragment_seq)
        gc_content = gc_fraction(analysis_seq) if len(analysis_seq) > 0 else 0.0

        # Dinucleotide and trinucleotide composition
        dinuc_comp = self.calculate_dinucleotide_composition(fragment_seq)
        trinuc_comp = self.calculate_trinucleotide_composition(fragment_seq)

        # Homopolymer detection
        homopolymers_6 = self.count_homopolymers(fragment_seq, min_length=6)
        homopolymers_8 = self.count_homopolymers(fragment_seq, min_length=8)
        total_homopolymer_runs = sum(homopolymers_6.values())
        flagged_homopolymers = sum(homopolymers_8.values()) > 0

        # K-mer duplication analysis
        total_21mers, duplicate_21mers = self.count_kmer_duplicates(fragment_seq, k=21)
        kmer_duplication_rate = (duplicate_21mers / total_21mers) if total_21mers > 0 else 0

        annotation = {
            'fragment_id': fragment_id,
            'gene_name': self.gene_name,
            'start_pos_1based': start_pos,
            'end_pos_1based': end_pos,
            'length': length,
            'overlap_type': overlap_type,
            'sequence': fragment_seq,

            # GC content
            'gc_content': gc_content,
            'gc_percentage': gc_content * 100,

            # Dinucleotide composition (top 5 most common)
            'dinuc_composition_json': json.dumps(dinuc_comp),
            'top_dinucleotides': json.dumps(dict(sorted(dinuc_comp.items(),
                                                       key=lambda x: x[1],
                                                       reverse=True)[:5])),

            # Trinucleotide composition (stored as JSON for space)
            'trinuc_composition_json': json.dumps(trinuc_comp),

            # Homopolymer analysis
            'homopolymer_runs_6plus': total_homopolymer_runs,
            'homopolymer_runs_8plus': sum(homopolymers_8.values()),
            'flagged_homopolymers': flagged_homopolymers,
            'homopolymer_details': json.dumps(homopolymers_6),

            # K-mer duplication
            'total_21mers': total_21mers,
            'duplicate_21mers': duplicate_21mers,
            'kmer_duplication_rate': kmer_duplication_rate,

            # Masked region overlap
            'masked_overlap_fraction': mask_overlap_fraction,
            'masked_overlap_percentage': mask_overlap_fraction * 100,

            # Quality flags
            'design_issues': []
        }

        # Flag potential design issues
        if flagged_homopolymers:
            annotation['design_issues'].append('homopolymer_8plus')
        if kmer_duplication_rate > 0.1:  # >10% duplicate 21-mers
            annotation['design_issues'].append('high_kmer_duplication')
        if mask_overlap_fraction > 0:
            annotation['design_issues'].append('masked_region_overlap')
        if gc_content < 0.3 or gc_content > 0.7:
            annotation['design_issues'].append('extreme_gc_content')

        annotation['design_issues_str'] = ';'.join(annotation['design_issues']) if annotation['design_issues'] else 'none'

        return annotation

    def generate_fragments(self,
                          window_lengths: List[int] = [200, 250, 300, 400, 500],
                          step_sizes_overlapping: List[int] = [25, 50],
                          generate_non_overlapping: bool = True,
                          generate_overlapping: bool = True) -> pd.DataFrame:
        """
        Generate dsRNA fragments using sliding window approach.

        Args:
            window_lengths: List of window lengths in bp
            step_sizes_overlapping: List of step sizes for overlapping windows
            generate_non_overlapping: Whether to generate non-overlapping (tiling) fragments
            generate_overlapping: Whether to generate overlapping fragments

        Returns:
            DataFrame containing all fragments with annotations
        """
        self.fragments = []

        for L in window_lengths:
            # Generate overlapping fragments
            if generate_overlapping:
                for S in step_sizes_overlapping:
                    fragment_count = 0
                    i = 0

                    while i + L <= self.cds_length:
                        # Extract subsequence (0-based, end-exclusive)
                        subseq = self.cds_sequence[i:i+L]

                        # Check masked region overlap
                        passes_mask_check, mask_overlap = self.check_masked_overlap(i, i+L)

                        if passes_mask_check:
                            # Convert to 1-based coordinates for output
                            start_1based = i + 1
                            end_1based = i + L

                            # Generate fragment ID
                            fragment_id = f"{self.gene_name}_L{L}_S{S}_pos{start_1based}"

                            # Annotate fragment
                            annotation = self.annotate_fragment(
                                subseq, fragment_id, start_1based, end_1based,
                                "overlapping", mask_overlap
                            )
                            annotation['window_length'] = L
                            annotation['step_size'] = S

                            self.fragments.append(annotation)
                            fragment_count += 1

                        i += S

                    print(f"Generated {fragment_count} overlapping fragments (L={L}, S={S})")

            # Generate non-overlapping fragments
            if generate_non_overlapping:
                fragment_count = 0
                i = 0
                S = L  # Step size equals window length

                while i + L <= self.cds_length:
                    subseq = self.cds_sequence[i:i+L]

                    passes_mask_check, mask_overlap = self.check_masked_overlap(i, i+L)

                    if passes_mask_check:
                        start_1based = i + 1
                        end_1based = i + L

                        fragment_id = f"{self.gene_name}_L{L}_NonOverlap_pos{start_1based}"

                        annotation = self.annotate_fragment(
                            subseq, fragment_id, start_1based, end_1based,
                            "non-overlapping", mask_overlap
                        )
                        annotation['window_length'] = L
                        annotation['step_size'] = S

                        self.fragments.append(annotation)
                        fragment_count += 1

                    i += S

                print(f"Generated {fragment_count} non-overlapping fragments (L={L})")

        # Create DataFrame
        df = pd.DataFrame(self.fragments)

        print(f"\nTotal fragments generated: {len(df)}")
        if len(df) > 0:
            print(f"  Overlapping: {len(df[df['overlap_type'] == 'overlapping'])}")
            print(f"  Non-overlapping: {len(df[df['overlap_type'] == 'non-overlapping'])}")
            print(f"  Fragments with design issues: {len(df[df['design_issues_str'] != 'none'])}")
        else:
            print("  (no fragments generated)")

        return df


def load_cds_from_fasta(fasta_file: str) -> Tuple[str, str]:
    """
    Load CDS sequence from FASTA file.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        (gene_name, sequence)
    """
    record = SeqIO.read(fasta_file, "fasta")
    return record.id, str(record.seq)


def load_masked_regions_from_file(mask_file: str) -> List[Tuple[int, int]]:
    """
    Load masked regions from a tab-delimited file.
    Expected format: start\tend (1-based, inclusive, one region per line)
    Lines starting with # are ignored.

    Args:
        mask_file: Path to mask file

    Returns:
        List of (start, end) tuples
    """
    masked_regions = []

    with open(mask_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    start, end = int(parts[0]), int(parts[1])
                    masked_regions.append((start, end))

    return masked_regions


def main():
    parser = argparse.ArgumentParser(
        description='Generate dsRNA design variants using sliding window approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with FASTA input
  python dsrna_sliding_window.py -i gene.fasta -o fragments.tsv

  # With masked regions and custom parameters
  python dsrna_sliding_window.py -i gene.fasta -o fragments.tsv \\
      -m masked_regions.txt -t 0.15 -l 200 250 300 400 500 -s 25 50

  # Generate only non-overlapping fragments
  python dsrna_sliding_window.py -i gene.fasta -o fragments.tsv \\
      --no-overlapping
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input CDS sequence (FASTA format)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file for fragments (TSV format)')
    parser.add_argument('-n', '--name', default=None,
                        help='Gene name (default: use FASTA ID)')
    parser.add_argument('-m', '--masked', default=None,
                        help='Masked regions file (tab-delimited: start end)')
    parser.add_argument('-t', '--tolerance', type=float, default=0.1,
                        help='Masked region overlap tolerance (default: 0.1 = 10%%)')
    parser.add_argument('-l', '--lengths', type=int, nargs='+',
                        default=[200, 250, 300, 400, 500],
                        help='Window lengths in bp (default: 200 250 300 400 500)')
    parser.add_argument('-s', '--steps', type=int, nargs='+',
                        default=[25, 50],
                        help='Step sizes for overlapping windows (default: 25 50)')
    parser.add_argument('--no-overlapping', action='store_true',
                        help='Skip generation of overlapping fragments')
    parser.add_argument('--no-non-overlapping', action='store_true',
                        help='Skip generation of non-overlapping fragments')
    parser.add_argument('--output-fasta', default=None,
                        help='Optional: output FASTA file with fragment sequences')

    args = parser.parse_args()

    # Load CDS sequence
    print(f"Wirtten by *****//////////////A Nayak C Bose S Nanda////////////////////SL1 SL2***********")
    print(f"Loading CDS sequence from {args.input}...")
    gene_name, cds_sequence = load_cds_from_fasta(args.input)

    if args.name:
        gene_name = args.name

    print(f"Gene: {gene_name}")
    print(f"CDS length: {len(cds_sequence)} bp")

    # Load masked regions if provided
    masked_regions = []
    if args.masked:
        print(f"Loading masked regions from {args.masked}...")
        masked_regions = load_masked_regions_from_file(args.masked)
        print(f"Loaded {len(masked_regions)} masked regions")

    # Initialize designer
    designer = dsRNADesigner(
        cds_sequence=cds_sequence,
        gene_name=gene_name,
        masked_regions=masked_regions,
        mask_tolerance=args.tolerance
    )

    # Generate fragments
    print("\nGenerating dsRNA fragments...")
    print(f"Window lengths: {args.lengths}")
    print(f"Step sizes (overlapping): {args.steps}")
    print(f"Masked region tolerance: {args.tolerance * 100}%")
    print()

    fragments_df = designer.generate_fragments(
        window_lengths=args.lengths,
        step_sizes_overlapping=args.steps,
        generate_non_overlapping=not args.no_non_overlapping,
        generate_overlapping=not args.no_overlapping
    )

    # Save results
    print(f"\nSaving results to {args.output}...")
    fragments_df.to_csv(args.output, sep='\t', index=False)

    # Optionally save FASTA
    if args.output_fasta:
        print(f"Saving fragment sequences to {args.output_fasta}...")
        with open(args.output_fasta, 'w') as f:
            for _, row in fragments_df.iterrows():
                f.write(f">{row['fragment_id']}\n")
                f.write(f"{row['sequence']}\n")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal fragments: {len(fragments_df)}")
    print(f"  Overlapping: {len(fragments_df[fragments_df['overlap_type'] == 'overlapping'])}")
    print(f"  Non-overlapping: {len(fragments_df[fragments_df['overlap_type'] == 'non-overlapping'])}")

    print("\nFragments by window length:")
    for L in args.lengths:
        count = len(fragments_df[fragments_df['window_length'] == L])
        print(f"  {L} bp: {count}")

    if len(fragments_df) > 0:
        print("\nGC content statistics:")
        print(f"  Mean: {fragments_df['gc_percentage'].mean():.2f}%")
        print(f"  Std: {fragments_df['gc_percentage'].std():.2f}%")
        print(f"  Range: {fragments_df['gc_percentage'].min():.2f}% - {fragments_df['gc_percentage'].max():.2f}%")
    else:
        print("\nNo fragments to compute GC statistics.")

    print("\nDesign issues:")
    if len(fragments_df) > 0:
        issues = fragments_df['design_issues_str'].value_counts()
        for issue, count in issues.items():
            print(f"  {issue}: {count}")

        clean_fragments = len(fragments_df[fragments_df['design_issues_str'] == 'none'])
        pct_clean = (clean_fragments / len(fragments_df) * 100) if len(fragments_df) > 0 else 0.0
        print(f"\nClean fragments (no issues): {clean_fragments} ({pct_clean:.1f}%)")
    else:
        print("  (no fragments)")

    print("\nDone!")
    print("\**************************Don't Forget To Cite the Work****************")


if __name__ == "__main__":
    main()
