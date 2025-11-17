"""
Extended Lesk Algorithm Implementation for Measuring Semantic Similarity.

The Extended Lesk algorithm measures similarity between synsets by computing
the overlap between their extended glosses (definitions, examples, and related synsets).

References:
- Lesk, M. (1986). Automatic sense disambiguation using machine readable dictionaries.
- Banerjee, S., & Pedersen, T. (2003). Extended gloss overlaps as a measure of semantic relatedness.
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import math
from algorithms.utils import get_all_synsets, is_valid_synset_pair


# Initialize NLTK components
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

try:
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocess text for gloss overlap computation.
    
    Steps:
    1. Lowercase
    2. Tokenize
    3. Remove punctuation
    4. Remove stopwords
    5. Lemmatize
    
    Args:
        text (str): Raw text to preprocess
    
    Returns:
        set: Set of preprocessed tokens
    """
    if not text:
        return set()
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords, then lemmatize
    processed_tokens = set()
    for token in tokens:
        # Skip punctuation
        if token in string.punctuation:
            continue
        # Skip stopwords
        if token in STOP_WORDS:
            continue
        # Lemmatize and add
        lemma = LEMMATIZER.lemmatize(token)
        processed_tokens.add(lemma)
    
    return processed_tokens


def get_extended_gloss(synset, include_relations=True):
    """
    Get extended gloss for a synset.
    
    Extended gloss includes:
    - Synset definition
    - Example sentences
    - (Optional) Definitions of related synsets (hypernyms, hyponyms, etc.)
    
    Args:
        synset: WordNet synset
        include_relations (bool): Whether to include related synsets' glosses
    
    Returns:
        set: Set of preprocessed tokens from the extended gloss
    """
    if synset is None:
        return set()
    
    # Collect all text
    gloss_text = []
    
    # Add synset definition
    gloss_text.append(synset.definition())
    
    # Add examples
    for example in synset.examples():
        gloss_text.append(example)
    
    # Add related synsets' glosses if requested
    if include_relations:
        # Hypernyms (more general concepts)
        for hypernym in synset.hypernyms():
            gloss_text.append(hypernym.definition())
        
        # Hyponyms (more specific concepts)
        for hyponym in synset.hyponyms():
            gloss_text.append(hyponym.definition())
        
        # Meronyms (part-of relationships)
        for meronym in synset.part_meronyms():
            gloss_text.append(meronym.definition())
        
        # Holonyms (whole-of relationships)
        for holonym in synset.part_holonyms():
            gloss_text.append(holonym.definition())
        
        # Similar-to (for adjectives)
        for similar in synset.similar_tos():
            gloss_text.append(similar.definition())
    
    # Combine all text and preprocess
    combined_text = ' '.join(gloss_text)
    return preprocess_text(combined_text)


def lesk_similarity(synset1, synset2, include_relations=True, normalization='geometric'):
    """
    Compute Lesk similarity between two synsets.
    
    Similarity is based on the overlap of tokens in their extended glosses.
    
    Args:
        synset1: First WordNet synset
        synset2: Second WordNet synset
        include_relations (bool): Whether to include related synsets in glosses
        normalization (str): Normalization method ('geometric', 'arithmetic', or 'none')
    
    Returns:
        float: Similarity score (higher is more similar), or None if invalid input
    """
    if not is_valid_synset_pair(synset1, synset2):
        return None
    
    # Get extended glosses
    gloss1 = get_extended_gloss(synset1, include_relations)
    gloss2 = get_extended_gloss(synset2, include_relations)
    
    # Handle empty glosses
    if not gloss1 or not gloss2:
        return 0.0
    
    # Compute overlap
    overlap = gloss1.intersection(gloss2)
    overlap_count = len(overlap)
    
    # Apply normalization
    if normalization == 'geometric':
        # Geometric mean: sqrt(len1 * len2)
        norm_factor = math.sqrt(len(gloss1) * len(gloss2))
    elif normalization == 'arithmetic':
        # Arithmetic mean: (len1 + len2) / 2
        norm_factor = (len(gloss1) + len(gloss2)) / 2.0
    elif normalization == 'none':
        norm_factor = 1.0
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Avoid division by zero
    if norm_factor == 0:
        return 0.0
    
    return overlap_count / norm_factor


def lesk_similarity_max_synsets(word1, word2, pos=None, include_relations=True, normalization='geometric'):
    """
    Compute maximum Lesk similarity between two words across all synset pairs.
    
    This handles polysemy by trying all possible sense combinations and
    returning the maximum similarity (best-case matching).
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        pos (str): Optional POS tag to filter synsets (NOUN, VERB, ADJ, ADV)
        include_relations (bool): Whether to include related synsets in glosses
        normalization (str): Normalization method
    
    Returns:
        float: Maximum similarity score, or None if no valid synsets found
    """
    # Get all synsets for both words
    synsets1 = get_all_synsets(word1, pos)
    synsets2 = get_all_synsets(word2, pos)
    
    # Check if we have synsets for both words
    if not synsets1 or not synsets2:
        return None
    
    # Compute similarity for all synset pairs and find maximum
    max_similarity = 0.0
    
    for s1 in synsets1:
        for s2 in synsets2:
            similarity = lesk_similarity(s1, s2, include_relations, normalization)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    
    return max_similarity if max_similarity > 0 else None


def context_aware_lesk(word, context_sentence, pos=None):
    """
    Select the best synset for a word given its context using Lesk WSD.
    
    Classic Lesk word sense disambiguation: finds the synset whose gloss
    has maximum overlap with the context sentence.
    
    Args:
        word (str): Target word to disambiguate
        context_sentence (str): Sentence containing the word
        pos (str): Optional POS tag to filter synsets
    
    Returns:
        Synset: Best matching synset, or None if no synsets found
    """
    # Get all synsets for the word
    synsets = get_all_synsets(word, pos)
    
    if not synsets:
        return None
    
    # Preprocess context
    context_tokens = preprocess_text(context_sentence)
    
    if not context_tokens:
        # No context available, return first synset as default
        return synsets[0]
    
    # Find synset with maximum overlap with context
    max_overlap = 0
    best_synset = synsets[0]  # Default to first synset
    
    for synset in synsets:
        gloss = get_extended_gloss(synset, include_relations=True)
        overlap = len(gloss.intersection(context_tokens))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_synset = synset
    
    return best_synset


def lesk_with_context(word1, word2, context1, context2, include_relations=True, normalization='geometric'):
    """
    Compute Lesk similarity using context-aware sense selection.
    
    First disambiguates each word using its context, then computes
    similarity between the selected synsets.
    
    Useful for contextualized datasets like SCWS.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        context1 (str): Sentence context for word1
        context2 (str): Sentence context for word2
        include_relations (bool): Whether to include related synsets in glosses
        normalization (str): Normalization method
    
    Returns:
        float: Similarity score, or None if disambiguation fails
    """
    # Disambiguate both words using context
    synset1 = context_aware_lesk(word1, context1)
    synset2 = context_aware_lesk(word2, context2)
    
    # Compute similarity between selected synsets
    return lesk_similarity(synset1, synset2, include_relations, normalization)
