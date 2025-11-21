from __future__ import annotations

import csv
import math
import os
import random
from collections import Counter
from itertools import islice
import importlib.util
from statistics import mean

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")

def _load_utils_module():  # pragma: no cover - fallback for script execution
    module_path = os.path.join(os.path.dirname(__file__), "utils.py")
    spec = importlib.util.spec_from_file_location("shortest_utils", module_path)
    if not spec or not spec.loader:
        raise ImportError("Unable to locate utils.py for shortest path module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # pragma: no cover - runtime convenience for script execution
    from .utils import convert_pos_tag, get_all_synsets, is_valid_synset_pair
except ImportError:  # pragma: no cover
    try:
        from algorithms.utils import convert_pos_tag, get_all_synsets, is_valid_synset_pair
    except ImportError:
        _utils = _load_utils_module()
        convert_pos_tag = _utils.convert_pos_tag
        get_all_synsets = _utils.get_all_synsets
        is_valid_synset_pair = _utils.is_valid_synset_pair


NLTK_PACKAGES = [
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
]


def ensure_nltk_resources():
    for resource_path, package in NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package)


def safe_float(value):
    value = (value or "").strip()
    return float(value) if value else float("nan")


def make_record(word1, word2, score, pos=None, context1=None, context2=None):
    return {
        "word1": word1,
        "word2": word2,
        "score": score,
        "pos": pos,
        "context1": context1,
        "context2": context2,
    }


def make_similarity(score=0.0, distance=None, synset1=None, synset2=None, strategy="no-context"):
    return {
        "score": score,
        "distance": distance,
        "synset1": synset1,
        "synset2": synset2,
        "strategy": strategy,
    }


def build_text_normalizer():
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def normalize(text):
        tokens = (tok for tok in word_tokenize(text.lower()) if tok.isalpha())
        lemmas = (lemmatizer.lemmatize(tok) for tok in tokens)
        return [lemma for lemma in lemmas if lemma not in stop_words]

    return normalize


def build_extended_lesk(normalize, include_relatives=True):
    def extended_gloss(synset):
        segments = [synset.definition(), *synset.examples()]
        if include_relatives:
            segments.extend(rel.definition() for rel in synset.hypernyms() + synset.hyponyms())
        return " ".join(segments)

    def signature(synset):
        return Counter(normalize(extended_gloss(synset)))

    def context_tokens(context, fallback_word):
        base_text = context if context and context.strip() else fallback_word
        return Counter(normalize(base_text))

    def disambiguate(word, context, pos):
        synsets = get_all_synsets(word, pos)
        if not synsets:
            return None

        context_counter = context_tokens(context, word)
        if not context_counter:
            return synsets[0]

        context_len = sum(context_counter.values())
        scored = []
        for synset in synsets:
            syn_signature = signature(synset)
            if not syn_signature:
                continue
            overlap = sum(min(syn_signature[token], context_counter[token]) for token in syn_signature)
            if overlap:
                scored.append((overlap / context_len, synset))

        return max(scored, default=(None, synsets[0]))[1]

    return disambiguate


def build_shortest_path_similarity(lesk=None):
    def max_pair(synsets1, synsets2):
        best = (0.0, None, None, None)
        for syn1 in synsets1:
            for syn2 in synsets2:
                if not is_valid_synset_pair(syn1, syn2):
                    continue
                distance = syn1.shortest_path_distance(syn2)
                if distance is None:
                    continue
                score = 1.0 / (distance + 1)
                if score > best[0]:
                    best = (score, distance, syn1, syn2)
        return make_similarity(best[0], best[1], best[2], best[3], "no-context")

    def context_pair(word1, word2, pos, context1, context2):
        if not lesk:
            return make_similarity(0.0, None, None, None, "with-context")

        syn1 = lesk(word1, context1, pos)
        syn2 = lesk(word2, context2, pos)
        if not is_valid_synset_pair(syn1, syn2):
            return make_similarity(0.0, None, syn1, syn2, "with-context")

        distance = syn1.shortest_path_distance(syn2)
        if distance is None:
            return make_similarity(0.0, None, syn1, syn2, "with-context")

        return make_similarity(1.0 / (distance + 1), distance, syn1, syn2, "with-context")

    def similarity(word1, word2, pos=None,
                   strategy="no-context",
                   context1=None,
                   context2=None):
        synsets1 = get_all_synsets(word1, pos)
        synsets2 = get_all_synsets(word2, pos)

        if not synsets1 or not synsets2:
            return make_similarity(0.0, None, None, None, strategy)

        if strategy == "with-context":
            return context_pair(word1, word2, pos, context1, context2)

        return max_pair(synsets1, synsets2)

    return similarity


def pearsonr(x, y):
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
    if not x or not y or len(x) != len(y):
        return float("nan")
    return pearsonr(_ranks(x), _ranks(y))


def bootstrap_significance(gold, scores_a, scores_b,
                           metric_fn, iterations=1000, seed=13):
    if not gold:
        return {"delta_mean": float("nan"), "p": 1.0, "ci_low": float("nan"), "ci_high": float("nan")}

    rng = random.Random(seed)
    indices = range(len(gold))
    diffs = [
        metric_fn([scores_a[i] for i in sample], [gold[i] for i in sample])
        - metric_fn([scores_b[i] for i in sample], [gold[i] for i in sample])
        for sample in (rng.choices(indices, k=len(gold)) for _ in range(iterations))
    ]

    diffs.sort()
    ci_low = diffs[int(0.025 * iterations)]
    ci_high = diffs[int(0.975 * iterations)]
    positive = sum(diff >= 0 for diff in diffs)
    p_value = 2 * min(positive, iterations - positive) / iterations
    return {"delta_mean": mean(diffs), "p": p_value, "ci_low": ci_low, "ci_high": ci_high}


def load_simlex(root):
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
            filtered.append(make_record(record["word1"], record["word2"], record["score"], target_pos,
                                        record["context1"], record["context2"]))
    return filtered


def compute_series(records, similarity, pos, strategy):
    gold = []
    predictions = []
    details = []

    for record in records:
        result = similarity(
            record["word1"],
            record["word2"],
            pos=pos,
            strategy=strategy,
            context1=record["context1"],
            context2=record["context2"],
        )
        if math.isnan(record["score"]):
            continue
        gold.append(record["score"])
        predictions.append(result["score"])
        details.append(result)

    return gold, predictions, details


def summarize_strategy(records, gold, predictions, strategy):
    return {
        "strategy": strategy,
        "pearson": pearsonr(predictions, gold),
        "spearman": spearmanr(predictions, gold),
        "coverage": len(predictions) / len(records) if records else 0.0,
        "errors": qualitative_errors(records, gold, predictions),
    }


def qualitative_errors(records, gold,
                       predictions, top_k=5):
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


def evaluate_subset(dataset_name, records, pos_label, pos_tag,
                    similarity_full, similarity_ablated, similarity_no_context):
    subset = filter_by_pos(records, pos_tag)
    if not subset:
        return {"dataset": dataset_name, "subset": pos_label, "results": []}

    gold_nc, preds_nc, _ = compute_series(subset, similarity_no_context, pos_tag, "no-context")
    gold_wc, preds_wc, details_wc = compute_series(subset, similarity_full, pos_tag, "with-context")
    _, preds_wc_abl, _ = compute_series(subset, similarity_ablated, pos_tag, "with-context")

    if not gold_nc:
        return {"dataset": dataset_name, "subset": pos_label, "results": []}

    metrics = [summarize_strategy(subset, gold_nc, preds_nc, "no-context")]

    wc_summary = summarize_strategy(subset, gold_wc, preds_wc, "with-context")
    wc_summary.update(
        {
            "bootstrap": {
                "pearson": bootstrap_significance(gold_nc, preds_wc, preds_nc, pearsonr),
                "spearman": bootstrap_significance(gold_nc, preds_wc, preds_nc, spearmanr),
            },
            "ablation": {
                "pearson": pearsonr(preds_wc_abl, gold_wc),
                "spearman": spearmanr(preds_wc_abl, gold_wc),
            },
            "sample_details": [
                {
                    "pair": f"{record['word1']}-{record['word2']}",
                    "synsets": (detail["synset1"].name() if detail["synset1"] else None,
                                 detail["synset2"].name() if detail["synset2"] else None),
                    "score": detail["score"],
                    "distance": detail["distance"],
                }
                for record, detail in islice(zip(subset, details_wc), 5)
            ],
        }
    )
    metrics.append(wc_summary)

    return {"dataset": dataset_name, "subset": pos_label, "results": metrics}


def run_evaluations(project_root):
    normalize = build_text_normalizer()
    lesk_full = build_extended_lesk(normalize, include_relatives=True)
    lesk_ablated = build_extended_lesk(normalize, include_relatives=False)

    similarity_full = build_shortest_path_similarity(lesk_full)
    similarity_ablated = build_shortest_path_similarity(lesk_ablated)
    similarity_no_context = build_shortest_path_similarity()

    datasets = {
        "SimLex-999": load_simlex(project_root),
        "WordSim-353": load_wordsim(project_root),
    }

    subsets = [("All", None), ("Nouns", wn.NOUN), ("Verbs", wn.VERB)]
    return [
        evaluate_subset(name, records, subset_name, pos_tag,
                        similarity_full, similarity_ablated, similarity_no_context)
        for name, records in datasets.items()
        for subset_name, pos_tag in subsets
    ]


def pretty_print(evaluations):
    for evaluation in evaluations:
        dataset = evaluation["dataset"]
        subset = evaluation["subset"]
        results = evaluation["results"]
        if not results:
            print(f"{dataset} / {subset}: no evaluable pairs")
            continue
        print(f"\n=== {dataset} [{subset}] ===")
        for result in results:
            strategy = result["strategy"]
            pearson = result["pearson"]
            spearman = result["spearman"]
            coverage = result["coverage"]
            print(f"{strategy:12s} | pearson={pearson:.4f} | spearman={spearman:.4f} | coverage={coverage:.3f}")
            if strategy == "with-context":
                boot = result.get("bootstrap", {})
                pearson_boot = boot.get("pearson", {})
                spearman_boot = boot.get("spearman", {})
                print(
                    "  bootstrap pearson Δ={:.4f} (p={:.3f}, ci=[{:.4f},{:.4f}])".format(
                        pearson_boot.get("delta_mean", float("nan")),
                        pearson_boot.get("p", float("nan")),
                        pearson_boot.get("ci_low", float("nan")),
                        pearson_boot.get("ci_high", float("nan")),
                    )
                )
                print(
                    "  bootstrap spearman Δ={:.4f} (p={:.3f}, ci=[{:.4f},{:.4f}])".format(
                        spearman_boot.get("delta_mean", float("nan")),
                        spearman_boot.get("p", float("nan")),
                        spearman_boot.get("ci_low", float("nan")),
                        spearman_boot.get("ci_high", float("nan")),
                    )
                )
                ablation = result.get("ablation", {})
                print(
                    "  ablation (no relatives) pearson={:.4f}, spearman={:.4f}".format(
                        ablation.get("pearson", float("nan")),
                        ablation.get("spearman", float("nan")),
                    )
                )
                errors = result.get("errors", [])
                if errors:
                    sample_error = errors[0]
                    print(
                        "  top error: {pair} human={human:.2f} predicted={predicted:.2f} Δ={delta:.2f}".format(
                            **sample_error
                        )
                    )


def run_shortest_path_pipeline(project_root=None):
    root = project_root if project_root else os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    evaluations = run_evaluations(root)
    pretty_print(evaluations)
    return evaluations


if __name__ == "__main__":
    run_shortest_path_pipeline()