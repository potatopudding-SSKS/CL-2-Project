"""
Test suite for Shortest Path similarity algorithm.
Evaluates performance on SimLex-999 and WordSim-353 datasets.
"""

import os
import sys
import csv
import math
from statistics import mean
from itertools import islice

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.shortest import build_shortest_path_similarity, ensure_nltk_resources
from algorithms.utils import convert_pos_tag, get_all_synsets
from nltk.corpus import wordnet as wn


def safe_float(value):
    """Convert string to float, return nan if invalid."""
    value = (value or "").strip()
    return float(value) if value else float("nan")


def make_record(word1, word2, score, pos=None):
    """Create a record dictionary for word pair."""
    return {
        "word1": word1,
        "word2": word2,
        "score": score,
        "pos": pos,
    }


def load_simlex(root):
    """Load SimLex-999 dataset."""
    path = os.path.join(root, "datasets", "SimLex-999.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            normalized = {key.lstrip("\ufeff"): value for key, value in row.items()}
            word1 = normalized.get("word1")
            word2 = normalized.get("word2")
            if not word1 or not word2:
                continue
            records.append(
                make_record(
                    word1,
                    word2,
                    safe_float(normalized.get("SimLex999", "")),
                    convert_pos_tag(normalized.get("POS")),
                )
            )
        return records


def load_wordsim(root):
    """Load WordSim-353 dataset."""
    path = os.path.join(root, "datasets", "wordsim353crowd.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            normalized = {key.lstrip("\ufeff"): value for key, value in row.items()}
            word1 = normalized.get("Word 1") or normalized.get("word1")
            word2 = normalized.get("Word 2") or normalized.get("word2")
            if not word1 or not word2:
                continue
            records.append(
                make_record(
                    word1,
                    word2,
                    safe_float(normalized.get("Human (Mean)", "")),
                )
            )
        return records


def filter_by_pos(records, target_pos):
    """Filter records by part of speech."""
    if target_pos is None:
        return list(records)

    filtered = []
    for record in records:
        if record["pos"] == target_pos:
            filtered.append(record)
            continue
        synsets1 = get_all_synsets(record["word1"], target_pos)
        synsets2 = get_all_synsets(record["word2"], target_pos)
        if synsets1 and synsets2:
            filtered.append(make_record(record["word1"], record["word2"], record["score"], target_pos))
    return filtered


def pearsonr(x, y):
    """Calculate Pearson correlation coefficient."""
    if not x or not y or len(x) != len(y):
        return float("nan")
    mean_x = mean(x)
    mean_y = mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denom_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denom_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if not denom_x or not denom_y:
        return float("nan")
    return num / (denom_x * denom_y)


def _ranks(values):
    """Convert values to ranks for Spearman correlation."""
    sorted_pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        total_rank = 0
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            total_rank += j + 1
            j += 1
        average_rank = total_rank / (j - i)
        for k in range(i, j):
            ranks[sorted_pairs[k][1]] = average_rank
        i = j
    return ranks


def spearmanr(x, y):
    """Calculate Spearman correlation coefficient."""
    if not x or not y or len(x) != len(y):
        return float("nan")
    return pearsonr(_ranks(x), _ranks(y))


def compute_series(records, similarity, pos):
    """Compute similarity scores for all records."""
    gold = []
    predictions = []
    details = []

    for record in records:
        result = similarity(
            record["word1"],
            record["word2"],
            pos=pos,
        )
        if math.isnan(record["score"]):
            continue
        gold.append(record["score"])
        predictions.append(result["score"])
        details.append(result)

    return gold, predictions, details


def qualitative_errors(records, gold, predictions, top_k=5):
    """Find word pairs with largest prediction errors."""
    scored = sorted(
        zip(records, gold, predictions),
        key=lambda triple: abs(triple[1] - triple[2]),
        reverse=True,
    )[:top_k]
    return [
        {
            "pair": f"{record['word1']}-{record['word2']}",
            "human": human,
            "predicted": predicted,
            "pos": record["pos"],
            "delta": predicted - human,
        }
        for record, human, predicted in scored
    ]


def summarize_results(records, gold, predictions):
    """Summarize evaluation results."""
    return {
        "pearson": pearsonr(predictions, gold),
        "spearman": spearmanr(predictions, gold),
        "coverage": len(predictions) / len(records) if records else 0.0,
        "errors": qualitative_errors(records, gold, predictions),
    }


def evaluate_subset(dataset_name, records, pos_label, pos_tag, similarity):
    """Evaluate similarity on a subset of data filtered by POS."""
    subset = filter_by_pos(records, pos_tag)
    if not subset:
        return {"dataset": dataset_name, "subset": pos_label, "results": {}}

    gold, predictions, details = compute_series(subset, similarity, pos_tag)

    if not gold:
        return {"dataset": dataset_name, "subset": pos_label, "results": {}}

    results = summarize_results(subset, gold, predictions)
    results["sample_details"] = [
        {
            "pair": f"{record['word1']}-{record['word2']}",
            "synsets": (detail["synset1"].name() if detail["synset1"] else None,
                         detail["synset2"].name() if detail["synset2"] else None),
            "score": detail["score"],
            "distance": detail["distance"],
        }
        for record, detail in islice(zip(subset, details), 5)
    ]

    return {"dataset": dataset_name, "subset": pos_label, "results": results}


def run_evaluations(project_root):
    """Run all evaluations for shortest path similarity."""
    ensure_nltk_resources()
    similarity = build_shortest_path_similarity()

    datasets = {
        "SimLex-999": load_simlex(project_root),
        "WordSim-353": load_wordsim(project_root),
    }

    subsets = [("All", None), ("Nouns", wn.NOUN), ("Verbs", wn.VERB)]
    return [
        evaluate_subset(name, records, subset_name, pos_tag, similarity)
        for name, records in datasets.items()
        for subset_name, pos_tag in subsets
    ]


def pretty_print(evaluations):
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("SHORTEST PATH SIMILARITY - EVALUATION RESULTS")
    print("="*80)
    
    for evaluation in evaluations:
        dataset = evaluation["dataset"]
        subset = evaluation["subset"]
        results = evaluation["results"]
        if not results:
            print(f"\n{dataset} / {subset}: no evaluable pairs")
            continue
        print(f"\n=== {dataset} [{subset}] ===")
        pearson = results["pearson"]
        spearman = results["spearman"]
        coverage = results["coverage"]
        print(f"Pearson:  {pearson:.4f}")
        print(f"Spearman: {spearman:.4f}")
        print(f"Coverage: {coverage:.3f}")
        
        errors = results.get("errors", [])
        if errors:
            sample_error = errors[0]
            print(
                f"Top error: {sample_error['pair']} "
                f"human={sample_error['human']:.2f} "
                f"predicted={sample_error['predicted']:.2f} "
                f"Δ={sample_error['delta']:.2f}"
            )


def main():
    """Main entry point for shortest path tests."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    evaluations = run_evaluations(project_root)
    pretty_print(evaluations)
    return evaluations


if __name__ == "__main__":
    main()
