"""
Evaluation script for Extended Lesk Algorithm.

This script:
1. Loads word pair datasets (SimLex-999, WordSim-353)
2. Computes Lesk similarity for each word pair
3. Calculates evaluation metrics (Spearman's ρ, Pearson's r)
4. Performs analysis by POS category
5. Displays results and example predictions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from algorithms.lesk import lesk_similarity_max_synsets
from algorithms.utils import convert_pos_tag
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_simlex999(filepath):
    """Load SimLex-999 dataset."""
    df = pd.read_csv(filepath)
    # Relevant columns: word1, word2, POS, SimLex999
    return df


def load_wordsim353(filepath):
    """Load WordSim-353 dataset."""
    df = pd.read_csv(filepath)
    # Relevant columns: Word 1, Word 2, Human (Mean)
    df.columns = ['word1', 'word2', 'human_score']
    df['POS'] = None  # WordSim-353 doesn't have POS tags
    return df


def evaluate_lesk(word_pairs, include_relations=True, normalization='geometric'):
    """
    Evaluate Lesk algorithm on word pairs.
    
    Args:
        word_pairs (list): List of (word1, word2, pos, gold_score) tuples
        include_relations (bool): Whether to include related synsets in glosses
        normalization (str): Normalization method
    
    Returns:
        dict: Results including predictions, correlations, and statistics
    """
    predictions = []
    gold_scores = []
    failed_pairs = []
    
    print(f"\nComputing Lesk similarities for {len(word_pairs)} word pairs...")
    
    for word1, word2, pos, gold in tqdm(word_pairs, desc="Processing"):
        # Convert POS tag if available
        wordnet_pos = convert_pos_tag(pos) if pos else None
        
        # Compute Lesk similarity
        pred = lesk_similarity_max_synsets(
            word1, word2, 
            pos=wordnet_pos,
            include_relations=include_relations,
            normalization=normalization
        )
        
        if pred is not None:
            predictions.append(pred)
            gold_scores.append(gold)
        else:
            failed_pairs.append((word1, word2, pos))
    
    # Calculate correlations
    if len(predictions) > 0:
        spearman_rho, spearman_p = spearmanr(predictions, gold_scores)
        pearson_r, pearson_p = pearsonr(predictions, gold_scores)
    else:
        spearman_rho = spearman_p = pearson_r = pearson_p = None
    
    return {
        'predictions': predictions,
        'gold_scores': gold_scores,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'coverage': len(predictions) / len(word_pairs) if len(word_pairs) > 0 else 0,
        'failed_pairs': failed_pairs,
        'total_pairs': len(word_pairs)
    }


def evaluate_by_pos(df, gold_col='SimLex999'):
    """Evaluate Lesk algorithm separately for each POS category."""
    
    if 'POS' not in df.columns or df['POS'].isna().all():
        print("\nNo POS information available for POS-specific evaluation.")
        return {}
    
    pos_results = {}
    pos_categories = df['POS'].dropna().unique()
    
    for pos in pos_categories:
        pos_df = df[df['POS'] == pos]
        word_pairs = [
            (row['word1'], row['word2'], row['POS'], row[gold_col])
            for _, row in pos_df.iterrows()
        ]
        
        print(f"\n{'='*70}")
        print(f"Evaluating POS: {pos} ({len(word_pairs)} pairs)")
        print(f"{'='*70}")
        
        results = evaluate_lesk(word_pairs)
        pos_results[pos] = results
        
        print_results(results, f"Lesk - POS: {pos}")
    
    return pos_results


def print_results(results, title="Lesk Algorithm"):
    """Print evaluation results in a formatted way."""
    
    print(f"\n{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}")
    
    print(f"\nCoverage: {results['coverage']*100:.2f}% ({len(results['predictions'])}/{results['total_pairs']} pairs)")
    
    if results['spearman_rho'] is not None:
        print(f"\n📊 Correlation Metrics:")
        print(f"   Spearman's ρ:  {results['spearman_rho']:.4f} (p-value: {results['spearman_p']:.4e})")
        print(f"   Pearson's r:   {results['pearson_r']:.4f} (p-value: {results['pearson_p']:.4e})")
        
        print(f"\n📈 Prediction Statistics:")
        print(f"   Mean prediction: {np.mean(results['predictions']):.4f}")
        print(f"   Std prediction:  {np.std(results['predictions']):.4f}")
        print(f"   Min prediction:  {np.min(results['predictions']):.4f}")
        print(f"   Max prediction:  {np.max(results['predictions']):.4f}")
        
        print(f"\n🎯 Gold Score Statistics:")
        print(f"   Mean gold: {np.mean(results['gold_scores']):.4f}")
        print(f"   Std gold:  {np.std(results['gold_scores']):.4f}")
        print(f"   Min gold:  {np.min(results['gold_scores']):.4f}")
        print(f"   Max gold:  {np.max(results['gold_scores']):.4f}")
    else:
        print("\n❌ Not enough valid predictions to compute correlations.")
    
    if len(results['failed_pairs']) > 0:
        print(f"\n⚠️  Failed pairs (no WordNet synsets): {len(results['failed_pairs'])}")
        print("   Examples:", results['failed_pairs'][:5])


def show_example_predictions(df, predictions, gold_col='SimLex999', n=10):
    """Show example predictions vs gold scores."""
    
    if len(predictions) == 0:
        return
    
    print(f"\n{'='*70}")
    print(f"Example Predictions (Top {n})".center(70))
    print(f"{'='*70}")
    print(f"{'Word 1':<15} {'Word 2':<15} {'POS':<5} {'Gold':<8} {'Predicted':<10} {'Diff':<8}")
    print("-" * 70)
    
    # Get valid predictions
    valid_indices = []
    pred_idx = 0
    for idx, row in df.iterrows():
        if pred_idx < len(predictions):
            valid_indices.append((idx, pred_idx))
            pred_idx += 1
    
    # Show top n
    for idx, pred_idx in valid_indices[:n]:
        row = df.iloc[idx]
        word1 = row['word1']
        word2 = row['word2']
        pos = row.get('POS', 'N/A')
        gold = row[gold_col]
        pred = predictions[pred_idx]
        diff = abs(gold - pred)
        
        print(f"{word1:<15} {word2:<15} {str(pos):<5} {gold:<8.2f} {pred:<10.4f} {diff:<8.4f}")


def main():
    """Main evaluation function."""
    
    print("\n" + "="*70)
    print("Extended Lesk Algorithm Evaluation".center(70))
    print("="*70)
    
    # Dataset paths
    simlex_path = 'datasets/SimLex-999.csv'
    wordsim_path = 'datasets/wordsim353crowd.csv'
    
    # ==========================
    # Evaluate on SimLex-999
    # ==========================
    print("\n\n" + "🔵 " * 35)
    print("DATASET: SimLex-999")
    print("🔵 " * 35)
    
    simlex_df = load_simlex999(simlex_path)
    print(f"Loaded {len(simlex_df)} word pairs from SimLex-999")
    
    # Overall evaluation
    simlex_pairs = [
        (row['word1'], row['word2'], row['POS'], row['SimLex999'])
        for _, row in simlex_df.iterrows()
    ]
    
    simlex_results = evaluate_lesk(simlex_pairs)
    print_results(simlex_results, "Lesk - SimLex-999 (All POS)")
    show_example_predictions(simlex_df, simlex_results['predictions'], 'SimLex999', n=15)
    
    # POS-specific evaluation
    simlex_pos_results = evaluate_by_pos(simlex_df, gold_col='SimLex999')
    
    # ==========================
    # Evaluate on WordSim-353
    # ==========================
    print("\n\n" + "🟢 " * 35)
    print("DATASET: WordSim-353")
    print("🟢 " * 35)
    
    wordsim_df = load_wordsim353(wordsim_path)
    print(f"Loaded {len(wordsim_df)} word pairs from WordSim-353")
    
    wordsim_pairs = [
        (row['word1'], row['word2'], row['POS'], row['human_score'])
        for _, row in wordsim_df.iterrows()
    ]
    
    wordsim_results = evaluate_lesk(wordsim_pairs)
    print_results(wordsim_results, "Lesk - WordSim-353")
    show_example_predictions(wordsim_df, wordsim_results['predictions'], 'human_score', n=15)
    
    # ==========================
    # Summary Comparison
    # ==========================
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON".center(70))
    print("="*70)
    
    print(f"\n{'Dataset':<20} {'Pairs':<10} {'Coverage':<12} {'Spearman ρ':<15} {'Pearson r':<15}")
    print("-" * 70)
    
    datasets = [
        ("SimLex-999", simlex_results),
        ("WordSim-353", wordsim_results),
    ]
    
    for name, res in datasets:
        coverage = f"{res['coverage']*100:.1f}%"
        spearman = f"{res['spearman_rho']:.4f}" if res['spearman_rho'] is not None else "N/A"
        pearson = f"{res['pearson_r']:.4f}" if res['pearson_r'] is not None else "N/A"
        print(f"{name:<20} {res['total_pairs']:<10} {coverage:<12} {spearman:<15} {pearson:<15}")
    
    # POS breakdown for SimLex-999
    if simlex_pos_results:
        print(f"\n{'SimLex POS':<20} {'Pairs':<10} {'Coverage':<12} {'Spearman ρ':<15} {'Pearson r':<15}")
        print("-" * 70)
        for pos, res in simlex_pos_results.items():
            coverage = f"{res['coverage']*100:.1f}%"
            spearman = f"{res['spearman_rho']:.4f}" if res['spearman_rho'] is not None else "N/A"
            pearson = f"{res['pearson_r']:.4f}" if res['pearson_r'] is not None else "N/A"
            pos_label = {'N': 'Nouns', 'V': 'Verbs', 'A': 'Adjectives'}.get(pos, pos)
            print(f"  {pos_label:<18} {res['total_pairs']:<10} {coverage:<12} {spearman:<15} {pearson:<15}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!".center(70))
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
