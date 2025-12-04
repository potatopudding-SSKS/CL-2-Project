"""
RQ2: Relationship Type Analysis
Categorize word pairs by semantic relationship type and analyze 
which types are handled well by each algorithm (Lesk, LCH, Wu-Palmer, and Shortest Path).
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import sys
import os
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.lesk import lesk_similarity_max_synsets
from algorithms.lch import lch_similarity_max_synsets
from algorithms.shortest import build_shortest_path_similarity
from algorithms.wup import build_wup_similarity
from algorithms.utils import convert_pos_tag
from nltk.corpus import wordnet as wn

def categorize_relationship(word1, word2, pos):
    """
    Categorize the semantic relationship between two words.
    
    Categories:
    - Synonyms: Very high similarity (same or nearly same meaning)
    - Hypernyms/Hyponyms: Hierarchical relationship (dog-animal)
    - Co-hyponyms: Share same parent (dog-cat both animals)
    - Antonyms: Opposite meaning
    - Meronyms/Holonyms: Part-whole relationship (wheel-car)
    - Functional: Functional/thematic relationship (doctor-nurse)
    - Unrelated: Low similarity, different concepts
    """
    
    # Convert POS tag to WordNet format
    wn_pos = convert_pos_tag(pos)
    
    synsets1 = wn.synsets(word1, wn_pos) if wn_pos else wn.synsets(word1)
    synsets2 = wn.synsets(word2, wn_pos) if wn_pos else wn.synsets(word2)
    
    if not synsets1 or not synsets2:
        return 'unrelated'
    
    # Check for synonyms (same synset)
    if any(s1 == s2 for s1 in synsets1 for s2 in synsets2):
        return 'synonym'
    
    # Check for antonyms
    for s1 in synsets1:
        for lemma1 in s1.lemmas():
            for ant in lemma1.antonyms():
                if any(ant.synset() == s2 for s2 in synsets2):
                    return 'antonym'
    
    # Check for hypernym/hyponym relationships
    for s1 in synsets1:
        # Check if s2 is hypernym of s1
        hypernyms = set(s1.closure(lambda s: s.hypernyms()))
        if any(s2 in hypernyms for s2 in synsets2):
            return 'hypernym-hyponym'
        
        # Check if s1 is hypernym of s2
        for s2 in synsets2:
            hypernyms2 = set(s2.closure(lambda s: s.hypernyms()))
            if s1 in hypernyms2:
                return 'hypernym-hyponym'
    
    # Check for co-hyponyms (siblings - same hypernym)
    for s1 in synsets1:
        hyp1 = s1.hypernyms()
        for s2 in synsets2:
            hyp2 = s2.hypernyms()
            if any(h1 == h2 for h1 in hyp1 for h2 in hyp2):
                return 'co-hyponym'
    
    # Check for meronyms/holonyms
    for s1 in synsets1:
        # Part-whole relationships
        parts = s1.part_meronyms() + s1.substance_meronyms() + s1.member_meronyms()
        wholes = s1.part_holonyms() + s1.substance_holonyms() + s1.member_holonyms()
        
        if any(s2 in parts or s2 in wholes for s2 in synsets2):
            return 'meronym-holonym'
    
    # If none of the above, it's either functional/thematic or unrelated
    # We'll use a simple heuristic: check path similarity
    max_path_sim = 0
    for s1 in synsets1:
        for s2 in synsets2:
            if s1.pos() == s2.pos() and s1.pos() in ['n', 'v']:
                sim = s1.path_similarity(s2)
                if sim and sim > max_path_sim:
                    max_path_sim = sim
    
    # If there's some path similarity but no direct relationship, it's functional
    if max_path_sim > 0.1:
        return 'functional'
    
    return 'unrelated'

def load_and_categorize_simlex():
    """Load SimLex-999 and categorize relationships."""
    datasets_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'SimLex-999.csv')
    df = pd.read_csv(datasets_path)
    
    categories = []
    for _, row in df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        pos = row['POS']
        
        category = categorize_relationship(word1, word2, pos)
        categories.append(category)
    
    df['relationship'] = categories
    return df

def load_and_categorize_wordsim():
    """Load WordSim-353 and categorize relationships."""
    datasets_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'wordsim353crowd.csv')
    df = pd.read_csv(datasets_path)
    
    categories = []
    for _, row in df.iterrows():
        word1 = row['Word 1']
        word2 = row['Word 2']
        
        # Try to determine POS (default to noun if unclear)
        # This is a simplification - could use POS tagging for better accuracy
        category = categorize_relationship(word1, word2, 'N')
        categories.append(category)
    
    df['relationship'] = categories
    return df

def evaluate_by_relationship(df, algorithm_func, algorithm_name, dataset_name, human_col='SimLex999'):
    """Evaluate algorithm performance by relationship type."""
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"{algorithm_name} - {dataset_name}")
    print(f"{'='*80}")
    
    # Overall results
    predictions = []
    human_scores = []
    
    for _, row in df.iterrows():
        if dataset_name == 'SimLex-999':
            word1, word2, pos = row['word1'], row['word2'], row['POS']
        else:
            word1, word2 = row['Word 1'], row['Word 2']
            pos = 'N'  # Default for WordSim
        
        # Convert POS tag to WordNet format
        wn_pos = convert_pos_tag(pos)
        
        pred = algorithm_func(word1, word2, wn_pos)
        if pred is not None:
            predictions.append(pred)
            human_scores.append(row[human_col])
        else:
            predictions.append(None)
            human_scores.append(None)
    
    df['prediction'] = predictions
    
    # Compute overall correlation
    valid_mask = df['prediction'].notna()
    if valid_mask.sum() > 0:
        spearman, _ = spearmanr(df[valid_mask][human_col], df[valid_mask]['prediction'])
        pearson, _ = pearsonr(df[valid_mask][human_col], df[valid_mask]['prediction'])
        coverage = valid_mask.sum() / len(df) * 100
        
        print(f"\nOverall Performance:")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r:  {pearson:.4f}")
        print(f"  Coverage:   {coverage:.2f}% ({valid_mask.sum()}/{len(df)} pairs)")
        
        results['overall'] = {
            'spearman': spearman,
            'pearson': pearson,
            'coverage': coverage,
            'n_pairs': valid_mask.sum()
        }
    
    # Analyze by relationship type
    print(f"\n{'Relationship Type':<20} {'N':<8} {'Coverage':<12} {'Spearman ρ':<12} {'Pearson r':<12}")
    print('-' * 80)
    
    for rel_type in sorted(df['relationship'].unique()):
        rel_df = df[df['relationship'] == rel_type]
        valid_rel = rel_df['prediction'].notna()
        
        if valid_rel.sum() >= 3:  # Need at least 3 pairs for correlation
            try:
                # Check if predictions are constant (would cause correlation warning)
                predictions = rel_df[valid_rel]['prediction']
                if predictions.nunique() == 1:
                    # All predictions are the same - correlation is undefined
                    coverage = valid_rel.sum() / len(rel_df) * 100
                    print(f"{rel_type:<20} {len(rel_df):<8} {coverage:>6.1f}%     {'constant':>8}      {'constant':>8}")
                    results[rel_type] = {
                        'spearman': np.nan,
                        'pearson': np.nan,
                        'coverage': coverage,
                        'n_pairs': len(rel_df),
                        'n_valid': valid_rel.sum(),
                        'note': 'constant predictions'
                    }
                else:
                    # Suppress the ConstantInputWarning if it still occurs
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='An input array is constant')
                        spearman, _ = spearmanr(rel_df[valid_rel][human_col], rel_df[valid_rel]['prediction'])
                        pearson, _ = pearsonr(rel_df[valid_rel][human_col], rel_df[valid_rel]['prediction'])
                    
                    coverage = valid_rel.sum() / len(rel_df) * 100
                    
                    print(f"{rel_type:<20} {len(rel_df):<8} {coverage:>6.1f}%     {spearman:>8.4f}      {pearson:>8.4f}")
                    
                    results[rel_type] = {
                        'spearman': spearman,
                        'pearson': pearson,
                        'coverage': coverage,
                        'n_pairs': len(rel_df),
                        'n_valid': valid_rel.sum()
                    }
            except:
                print(f"{rel_type:<20} {len(rel_df):<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"{rel_type:<20} {len(rel_df):<8} {'Too few':<12} {'N/A':<12} {'N/A':<12}")
    
    return results, df

def show_examples_by_relationship(df, relationship_type, n=5):
    """Show example predictions for a specific relationship type."""
    
    rel_df = df[df['relationship'] == relationship_type].copy()
    rel_df = rel_df[rel_df['prediction'].notna()]
    
    if len(rel_df) == 0:
        return
    
    # Calculate absolute error
    if 'SimLex999' in rel_df.columns:
        human_col = 'SimLex999'
        word1_col, word2_col = 'word1', 'word2'
    else:
        human_col = 'Human (Mean)'
        word1_col, word2_col = 'Word 1', 'Word 2'
    
    rel_df['error'] = abs(rel_df[human_col] - rel_df['prediction'])
    rel_df = rel_df.sort_values('error', ascending=False)
    
    print(f"\n{relationship_type.upper()} - Examples (worst predictions):")
    print(f"{'Word 1':<15} {'Word 2':<15} {'Human':<8} {'Predicted':<10} {'Error':<8}")
    print('-' * 70)
    
    for _, row in rel_df.head(n).iterrows():
        print(f"{row[word1_col]:<15} {row[word2_col]:<15} {row[human_col]:<8.2f} {row['prediction']:<10.4f} {row['error']:<8.4f}")

def main():
    """Main evaluation function."""
    
    print("Loading and categorizing datasets...")
    simlex = load_and_categorize_simlex()
    wordsim = load_and_categorize_wordsim()
    
    print("\nRelationship distribution in SimLex-999:")
    print(simlex['relationship'].value_counts())
    
    print("\nRelationship distribution in WordSim-353:")
    print(wordsim['relationship'].value_counts())
    
    # Evaluate Lesk
    print("\n" + "="*80)
    print("EVALUATING LESK ALGORITHM")
    print("="*80)
    
    lesk_simlex_results, lesk_simlex_df = evaluate_by_relationship(
        simlex.copy(), lesk_similarity_max_synsets, "Lesk", "SimLex-999"
    )
    
    lesk_wordsim_results, lesk_wordsim_df = evaluate_by_relationship(
        wordsim.copy(), lesk_similarity_max_synsets, "Lesk", "WordSim-353", human_col='Human (Mean)'
    )
    
    # Evaluate LCH
    print("\n" + "="*80)
    print("EVALUATING LCH ALGORITHM")
    print("="*80)
    
    lch_simlex_results, lch_simlex_df = evaluate_by_relationship(
        simlex.copy(), lch_similarity_max_synsets, "LCH", "SimLex-999"
    )
    
    lch_wordsim_results, lch_wordsim_df = evaluate_by_relationship(
        wordsim.copy(), lch_similarity_max_synsets, "LCH", "WordSim-353", human_col='Human (Mean)'
    )
    
    # Build similarity functions for algorithms that need it
    shortest_similarity = build_shortest_path_similarity()
    wup_similarity = build_wup_similarity()
    
    # Wrapper functions to match the expected signature
    def shortest_wrapper(word1, word2, pos=None):
        result = shortest_similarity(word1, word2, pos)
        return result['score'] if result else None
    
    def wup_wrapper(word1, word2, pos=None):
        result = wup_similarity(word1, word2, pos)
        return result['score'] if result else None
    
    # Evaluate Shortest Path
    print("\n" + "="*80)
    print("EVALUATING SHORTEST PATH ALGORITHM")
    print("="*80)
    
    shortest_simlex_results, shortest_simlex_df = evaluate_by_relationship(
        simlex.copy(), shortest_wrapper, "Shortest Path", "SimLex-999"
    )
    
    shortest_wordsim_results, shortest_wordsim_df = evaluate_by_relationship(
        wordsim.copy(), shortest_wrapper, "Shortest Path", "WordSim-353", human_col='Human (Mean)'
    )
    
    # Evaluate Wu-Palmer
    print("\n" + "="*80)
    print("EVALUATING WU-PALMER ALGORITHM")
    print("="*80)
    
    wup_simlex_results, wup_simlex_df = evaluate_by_relationship(
        simlex.copy(), wup_wrapper, "Wu-Palmer", "SimLex-999"
    )
    
    wup_wordsim_results, wup_wordsim_df = evaluate_by_relationship(
        wordsim.copy(), wup_wrapper, "Wu-Palmer", "WordSim-353", human_col='Human (Mean)'
    )
    
    # Show examples for key relationship types
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS BY RELATIONSHIP TYPE")
    print("="*80)
    
    algorithms = [
        ('Lesk', lesk_simlex_df),
        ('LCH', lch_simlex_df),
        ('Shortest Path', shortest_simlex_df),
        ('Wu-Palmer', wup_simlex_df)
    ]
    
    for rel_type in ['synonym', 'hypernym-hyponym', 'antonym', 'co-hyponym', 'functional']:
        if rel_type in simlex['relationship'].values:
            for algo_name, algo_df in algorithms:
                print(f"\n{'='*80}")
                print(f"{algo_name} - {rel_type}")
                show_examples_by_relationship(algo_df, rel_type)

if __name__ == "__main__":
    main()
