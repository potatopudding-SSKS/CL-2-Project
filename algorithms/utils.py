"""
Utility functions for WordNet-based similarity measures.
Shared helper functions across different algorithms.
"""

from nltk.corpus import wordnet as wn


def get_all_synsets(word, pos=None):
    """
    Get all synsets for a word, optionally filtered by POS.
    
    Args:
        word (str): The word to look up
        pos (str): Optional POS tag (NOUN, VERB, ADJ, ADV)
    
    Returns:
        list: List of synsets for the word
    """
    if pos:
        return wn.synsets(word, pos=pos)
    return wn.synsets(word)


def convert_pos_tag(tag):
    """
    Convert various POS tag formats to WordNet POS constants.
    
    Supports:
    - Single letter tags (N, V, A, R)
    - Penn Treebank tags (NN, VB, JJ, RB, etc.)
    
    Args:
        tag (str): POS tag in various formats
    
    Returns:
        str or None: WordNet POS constant (NOUN, VERB, ADJ, ADV) or None
    """
    if not tag:
        return None
    
    tag = tag.upper()
    
    # Single letter format (from datasets like SimLex)
    if tag == 'N':
        return wn.NOUN
    elif tag == 'V':
        return wn.VERB
    elif tag == 'A':
        return wn.ADJ
    elif tag == 'R':
        return wn.ADV
    
    # Penn Treebank format
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    
    return None


def is_valid_synset_pair(synset1, synset2):
    """
    Check if two synsets are valid for similarity computation.
    
    Args:
        synset1: First synset
        synset2: Second synset
    
    Returns:
        bool: True if both synsets are valid, False otherwise
    """
    return synset1 is not None and synset2 is not None
