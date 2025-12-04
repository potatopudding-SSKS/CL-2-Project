from __future__ import annotations

import os
import importlib.util

import nltk
from nltk.corpus import wordnet as wn


def _load_utils_module():  # pragma: no cover - fallback for script execution
	module_path = os.path.join(os.path.dirname(__file__), "utils.py")
	spec = importlib.util.spec_from_file_location("wup_utils", module_path)
	if not spec or not spec.loader:
		raise ImportError("Unable to locate utils.py for WUP module")
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
]



def ensure_nltk_resources():
	for resource_path, package in NLTK_PACKAGES:
		try:
			nltk.data.find(resource_path)
		except LookupError:
			nltk.download(package)


def make_similarity(score=0.0, depth=None, synset1=None, synset2=None):
	return {
		"score": score,
		"distance": depth,
		"synset1": synset1,
		"synset2": synset2,
	}


def build_wup_similarity():
	"""Build Wu-Palmer similarity function using max-over-synsets strategy."""
	def wup_score(syn1, syn2):
		builtin = syn1.wup_similarity(syn2)
		lcs_candidates = syn1.lowest_common_hypernyms(syn2)
		if not lcs_candidates:
			return (builtin if builtin is not None else 0.0), None
		lcs = max(lcs_candidates, key=lambda syn: syn.min_depth())
		depth_lcs = lcs.min_depth()
		depth1 = syn1.min_depth()
		depth2 = syn2.min_depth()
		denom = depth1 + depth2
		manual = (2 * depth_lcs) / denom if denom else 0.0
		score = builtin if builtin is not None else manual
		return score, depth_lcs

	def similarity(word1, word2, pos=None):
		synsets1 = get_all_synsets(word1, pos)
		synsets2 = get_all_synsets(word2, pos)

		if not synsets1 or not synsets2:
			return make_similarity(0.0, None, None, None)

		# Max-over-synsets: try all combinations and pick the best
		best = (0.0, None, None, None)
		for syn1 in synsets1:
			for syn2 in synsets2:
				if not is_valid_synset_pair(syn1, syn2):
					continue
				score, depth = wup_score(syn1, syn2)
				if score > best[0]:
					best = (score, depth, syn1, syn2)
		
		return make_similarity(best[0], best[1], best[2], best[3])

	return similarity
