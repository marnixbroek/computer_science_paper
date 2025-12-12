import re

import numpy as np

from modelWordsFunctions import extractModelWordsFromTitle

def TMWMsimilarity(data, key1, key2, alpha, beta):
    """
    Calculate the Title Model Words Method Similarity between two products.
    It calculates the cosine similarity of two product names.
    It returns a similarity of 1 if the original TMWM would cluster the two products,
    and either -1 or the cosine similarity otherwise, based on thresholds alpha and beta.
    """ 
    
    # First obtain the titles of both products
    title1 = data[key1]['title'].lower()
    title2 = data[key2]['title'].lower()

    # Calculate cosine similarity
    words1 = extractModelWordsFromTitle(title1, method='MSMP+')
    words2 = extractModelWordsFromTitle(title2, method='MSMP+')

    # Union of all words
    all_words = set(words1).union(words2)

    # Build binary vectors (1 if word is present, 0 otherwise)
    vec1 = np.array([1 if word in words1 else 0 for word in all_words])
    vec2 = np.array([1 if word in words2 else 0 for word in all_words])

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)

    # Determine if TMWM would cluster the two products
    if cosine_similarity >= alpha:
        return 1
    else:
        # Return -1 if smaller than beta
        if cosine_similarity < beta:
            return -1
        else:
            return cosine_similarity
        
def minFeatures(data, key1, key2):
    """Return the minimum number of features between two products."""
    features1 = data[key1]['featuresMap']
    features2 = data[key2]['featuresMap']
    return min(len(features1), len(features2))