"""
Leacock-Chodorow (LCH) Similarity Implementation.

The LCH similarity measure uses the shortest path length between two synsets
and the maximum depth of the taxonomy to compute semantic similarity.

Formula: LCH(c1, c2) = -log(shortest_path(c1, c2) / (2 * max_depth))

References:
- Leacock, C., & Chodorow, M. (1998). Combining Local Context and WordNet 
  Similarity for WSD. In WordNet: An electronic lexical database.
"""

import math
from nltk.corpus import wordnet as wn
from algorithms.utils import get_all_synsets, is_valid_synset_pair


# Cache for maximum depths by POS
_MAX_DEPTH_CACHE = {}


def get_max_depth(pos):
    """
    Get the maximum depth of the WordNet taxonomy for a given POS.
    
    The maximum depth is the longest path from the root to any leaf node
    in the taxonomy hierarchy.
    
    Args:
        pos (str): Part of speech (NOUN, VERB, ADJ, ADV)
    
    Returns:
        int: Maximum depth for the taxonomy, or None if POS not supported
    """
    # LCH only works for NOUN and VERB (hierarchical taxonomies)
    if pos not in [wn.NOUN, wn.VERB]:
        return None
    
    # Return cached value if available
    if pos in _MAX_DEPTH_CACHE:
        return _MAX_DEPTH_CACHE[pos]
    
    # Compute maximum depth by finding the deepest synset
    max_depth = 0
    
    for synset in wn.all_synsets(pos):
        depth = synset.max_depth()
        if depth > max_depth:
            max_depth = depth
    
    # Cache the result
    _MAX_DEPTH_CACHE[pos] = max_depth
    
    return max_depth


def lch_similarity(synset1, synset2):
    """
    Compute Leacock-Chodorow similarity between two synsets.
    
    LCH uses the shortest path distance between synsets and the maximum
    depth of the taxonomy:
    
    LCH(c1, c2) = -log(shortest_path(c1, c2) / (2 * max_depth))
    
    Higher scores indicate greater similarity.
    
    Args:
        synset1: First WordNet synset
        synset2: Second WordNet synset
    
    Returns:
        float: LCH similarity score (higher is more similar), or None if invalid
    
    Notes:
        - Only works for NOUN and VERB synsets (hierarchical taxonomies)
        - Returns None if synsets are from different POS categories
        - Returns None if no path exists between synsets
    """
    if not is_valid_synset_pair(synset1, synset2):
        return None
    
    # Check that both synsets have the same POS
    if synset1.pos() != synset2.pos():
        return None
    
    pos = synset1.pos()
    
    # LCH only works for nouns and verbs
    if pos not in ['n', 'v']:
        return None
    
    # Convert pos to WordNet constant
    wordnet_pos = wn.NOUN if pos == 'n' else wn.VERB
    
    # Get maximum depth for this POS
    max_depth = get_max_depth(wordnet_pos)
    
    if max_depth is None or max_depth == 0:
        return None
    
    # Get shortest path distance
    # shortest_path_distance returns None if no path exists
    path_distance = synset1.shortest_path_distance(synset2)
    
    if path_distance is None:
        return None
    
    # Handle special case: same synset (distance = 0)
    # Avoid log(0) by using a small epsilon
    if path_distance == 0:
        # Maximum similarity: use smallest possible distance
        path_distance = 1
    
    # Apply LCH formula: -log(path / (2 * max_depth))
    try:
        lch_score = -math.log(path_distance / (2.0 * max_depth))
    except (ValueError, ZeroDivisionError):
        return None
    
    return lch_score


def lch_similarity_max_synsets(word1, word2, pos=None):
    """
    Compute maximum LCH similarity between two words across all synset pairs.
    
    This handles polysemy by trying all possible sense combinations and
    returning the maximum similarity (best-case matching).
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        pos (str): Optional POS tag to filter synsets (NOUN or VERB only)
    
    Returns:
        float: Maximum LCH similarity score, or None if no valid synsets found
    
    Notes:
        - Only works for nouns and verbs
        - If POS is not NOUN or VERB, returns None
        - Returns None if either word has no synsets
    """
    # LCH only works for NOUN and VERB
    if pos is not None and pos not in [wn.NOUN, wn.VERB]:
        return None
    
    # Get all synsets for both words
    synsets1 = get_all_synsets(word1, pos)
    synsets2 = get_all_synsets(word2, pos)
    
    # Filter to only noun and verb synsets if no POS specified
    if pos is None:
        synsets1 = [s for s in synsets1 if s.pos() in ['n', 'v']]
        synsets2 = [s for s in synsets2 if s.pos() in ['n', 'v']]
    
    # Check if we have synsets for both words
    if not synsets1 or not synsets2:
        return None
    
    # Compute similarity for all synset pairs and find maximum
    max_similarity = None
    
    for s1 in synsets1:
        for s2 in synsets2:
            similarity = lch_similarity(s1, s2)
            
            if similarity is not None:
                if max_similarity is None or similarity > max_similarity:
                    max_similarity = similarity
    
    return max_similarity


def lch_similarity_with_context(word1, word2, context1=None, context2=None, pos=None):
    """
    Compute LCH similarity with optional context-aware sense selection.
    
    If contexts are provided, uses Lesk-based WSD to select synsets first,
    then computes LCH similarity between selected synsets.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        context1 (str): Optional sentence context for word1
        context2 (str): Optional sentence context for word2
        pos (str): Optional POS tag (NOUN or VERB)
    
    Returns:
        float: LCH similarity score, or None if computation fails
    """
    # If no context provided, fall back to max synsets approach
    if context1 is None or context2 is None:
        return lch_similarity_max_synsets(word1, word2, pos)
    
    # Import here to avoid circular dependency
    from algorithms.lesk import context_aware_lesk
    
    # Disambiguate both words using context
    synset1 = context_aware_lesk(word1, context1, pos)
    synset2 = context_aware_lesk(word2, context2, pos)
    
    # Compute LCH similarity between selected synsets
    return lch_similarity(synset1, synset2)
