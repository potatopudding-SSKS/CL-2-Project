"""
RQ3: Context-Aware WSD Evaluation
Compare context-aware word sense disambiguation against max-over-synsets baseline.
Evaluate on SCWS dataset (or sample dataset with contexts).
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import sys
sys.path.append('/home/yash-more/Downloads/21/cl2/project/CL-2-Project')

from algorithms.lesk import lesk_similarity_max_synsets, lesk_with_context
from algorithms.lch import lch_similarity_max_synsets, lch_similarity_with_context
from algorithms.utils import convert_pos_tag

def load_scws_sample():
    """Load the sample SCWS dataset."""
    df = pd.read_csv(
        '/home/yash-more/Downloads/21/cl2/project/CL-2-Project/datasets/scws_sample.txt',
        sep='\t'
    )
    return df

def evaluate_baseline(df, algorithm_func, algorithm_name):
    """
    Evaluate algorithm without context (max-over-synsets).
    
    Args:
        df: DataFrame with word pairs and contexts
        algorithm_func: Algorithm function (lesk_similarity_max_synsets or lch_similarity_max_synsets)
        algorithm_name: Name of algorithm for display
    
    Returns:
        tuple: (results dict, predictions list)
    """
    print(f"\n{'='*80}")
    print(f"{algorithm_name} - Max-Over-Synsets Baseline (No Context)")
    print(f"{'='*80}")
    
    predictions = []
    human_scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        word1 = row['word1']
        word2 = row['word2']
        pos = row['pos']
        human = row['similarity']
        
        # Convert POS tag
        wn_pos = convert_pos_tag(pos)
        
        # Compute similarity without context
        pred = algorithm_func(word1, word2, wn_pos)
        
        predictions.append(pred)
        human_scores.append(human)
    
    # Compute correlation
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_preds = [predictions[i] for i in valid_indices]
    valid_human = [human_scores[i] for i in valid_indices]
    
    if len(valid_preds) >= 3:
        spearman, _ = spearmanr(valid_human, valid_preds)
        pearson, _ = pearsonr(valid_human, valid_preds)
        coverage = len(valid_preds) / len(df) * 100
        
        print(f"\nResults:")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r:  {pearson:.4f}")
        print(f"  Coverage:   {coverage:.2f}% ({len(valid_preds)}/{len(df)} pairs)")
        
        results = {
            'spearman': spearman,
            'pearson': pearson,
            'coverage': coverage,
            'n_pairs': len(valid_preds)
        }
    else:
        print("Insufficient valid predictions for correlation.")
        results = {}
    
    return results, predictions

def evaluate_context_aware(df, algorithm_func, algorithm_name, predictions_baseline):
    """
    Evaluate algorithm with context-aware WSD.
    
    Args:
        df: DataFrame with word pairs and contexts
        algorithm_func: Context-aware algorithm function
        algorithm_name: Name of algorithm for display
        predictions_baseline: Baseline predictions for comparison
    
    Returns:
        tuple: (results dict, predictions list)
    """
    print(f"\n{'='*80}")
    print(f"{algorithm_name} - Context-Aware WSD")
    print(f"{'='*80}")
    
    predictions = []
    human_scores = []
    improvements = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        word1 = row['word1']
        word2 = row['word2']
        pos = row['pos']
        context1 = row['context1']
        context2 = row['context2']
        human = row['similarity']
        
        # Convert POS tag
        wn_pos = convert_pos_tag(pos)
        
        # Compute context-aware similarity
        pred = algorithm_func(word1, word2, context1, context2, wn_pos)
        
        predictions.append(pred)
        human_scores.append(human)
        
        # Track improvement over baseline
        baseline_pred = predictions_baseline[idx]
        if pred is not None and baseline_pred is not None:
            baseline_error = abs(human - baseline_pred)
            context_error = abs(human - pred)
            improvement = baseline_error - context_error
            improvements.append(improvement)
    
    # Compute correlation
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_preds = [predictions[i] for i in valid_indices]
    valid_human = [human_scores[i] for i in valid_indices]
    
    if len(valid_preds) >= 3:
        spearman, _ = spearmanr(valid_human, valid_preds)
        pearson, _ = pearsonr(valid_human, valid_preds)
        coverage = len(valid_preds) / len(df) * 100
        
        print(f"\nResults:")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r:  {pearson:.4f}")
        print(f"  Coverage:   {coverage:.2f}% ({len(valid_preds)}/{len(df)} pairs)")
        
        if improvements:
            avg_improvement = np.mean(improvements)
            improved_count = sum(1 for imp in improvements if imp > 0)
            print(f"\nContext Impact:")
            print(f"  Avg error reduction: {avg_improvement:.4f}")
            print(f"  Improved predictions: {improved_count}/{len(improvements)} ({improved_count/len(improvements)*100:.1f}%)")
        
        results = {
            'spearman': spearman,
            'pearson': pearson,
            'coverage': coverage,
            'n_pairs': len(valid_preds),
            'avg_improvement': np.mean(improvements) if improvements else 0,
            'improved_count': sum(1 for imp in improvements if imp > 0) if improvements else 0
        }
    else:
        print("Insufficient valid predictions for correlation.")
        results = {}
    
    return results, predictions

def compare_results(baseline_results, context_results, algorithm_name):
    """Compare baseline and context-aware results."""
    
    print(f"\n{'='*80}")
    print(f"{algorithm_name} - Comparison Summary")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Context-Aware':<15} {'Δ':<10}")
    print('-' * 70)
    
    if baseline_results and context_results:
        spearman_delta = context_results['spearman'] - baseline_results['spearman']
        pearson_delta = context_results['pearson'] - baseline_results['pearson']
        
        print(f"{'Spearman ρ':<25} {baseline_results['spearman']:<15.4f} {context_results['spearman']:<15.4f} {spearman_delta:>+9.4f}")
        print(f"{'Pearson r':<25} {baseline_results['pearson']:<15.4f} {context_results['pearson']:<15.4f} {pearson_delta:>+9.4f}")
        print(f"{'Coverage (%)':<25} {baseline_results['coverage']:<15.2f} {context_results['coverage']:<15.2f}")
        
        if 'avg_improvement' in context_results:
            print(f"\nWSD Impact:")
            print(f"  Average error reduction: {context_results['avg_improvement']:.4f}")
            print(f"  Improved predictions: {context_results['improved_count']}/{context_results['n_pairs']}")

def show_examples(df, baseline_preds, context_preds, n=10):
    """Show examples where context made a difference."""
    
    print(f"\n{'='*80}")
    print("Examples: Context Impact on Predictions")
    print(f"{'='*80}")
    
    # Calculate improvements
    examples = []
    for idx, row in df.iterrows():
        baseline = baseline_preds[idx]
        context = context_preds[idx]
        human = row['similarity']
        
        if baseline is not None and context is not None:
            baseline_error = abs(human - baseline)
            context_error = abs(human - context)
            improvement = baseline_error - context_error
            
            examples.append({
                'word1': row['word1'],
                'word2': row['word2'],
                'context1': row['context1'][:50] + '...' if len(row['context1']) > 50 else row['context1'],
                'context2': row['context2'][:50] + '...' if len(row['context2']) > 50 else row['context2'],
                'human': human,
                'baseline': baseline,
                'context': context,
                'improvement': improvement
            })
    
    # Sort by absolute improvement
    examples_sorted = sorted(examples, key=lambda x: abs(x['improvement']), reverse=True)
    
    print(f"\nTop {n} Largest Context Effects:")
    print(f"{'Word Pair':<20} {'Human':<8} {'Baseline':<10} {'Context':<10} {'Δ Error':<10}")
    print('-' * 70)
    
    for ex in examples_sorted[:n]:
        word_pair = f"{ex['word1']}-{ex['word2']}"
        print(f"{word_pair:<20} {ex['human']:<8.2f} {ex['baseline']:<10.4f} {ex['context']:<10.4f} {ex['improvement']:>+9.4f}")
        print(f"  C1: {ex['context1']}")
        print(f"  C2: {ex['context2']}")
        print()

def main():
    """Main evaluation function."""
    
    print("Loading SCWS sample dataset with contexts...")
    scws = load_scws_sample()
    
    print(f"\nDataset: {len(scws)} word pairs with sentence contexts")
    print(f"Columns: {list(scws.columns)}")
    
    # Evaluate Lesk
    print("\n" + "="*80)
    print("EVALUATING LESK ALGORITHM")
    print("="*80)
    
    lesk_baseline_results, lesk_baseline_preds = evaluate_baseline(
        scws, lesk_similarity_max_synsets, "Lesk"
    )
    
    lesk_context_results, lesk_context_preds = evaluate_context_aware(
        scws, lesk_with_context, "Lesk", lesk_baseline_preds
    )
    
    compare_results(lesk_baseline_results, lesk_context_results, "Lesk")
    show_examples(scws, lesk_baseline_preds, lesk_context_preds, n=10)
    
    # Evaluate LCH
    print("\n" + "="*80)
    print("EVALUATING LCH ALGORITHM")
    print("="*80)
    
    lch_baseline_results, lch_baseline_preds = evaluate_baseline(
        scws, lch_similarity_max_synsets, "LCH"
    )
    
    lch_context_results, lch_context_preds = evaluate_context_aware(
        scws, lch_similarity_with_context, "LCH", lch_baseline_preds
    )
    
    compare_results(lch_baseline_results, lch_context_results, "LCH")
    show_examples(scws, lch_baseline_preds, lch_context_preds, n=10)

if __name__ == "__main__":
    main()
