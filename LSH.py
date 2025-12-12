from collections import defaultdict
from itertools import combinations
import random

import numpy as np

random.seed(11)

def compute_minhash_signatures(b_matrix):
    b_matrix = np.array(b_matrix)

    r, N = b_matrix.shape

    # Determine suitable k. In paper, they set 50% of r, but that doesn't always result in a divisible k.
    k = find_suitable_k(r)

    # Choose a large prime > r
    p = 50009

    # Random hash parameters
    a = np.random.randint(1, p-1, size=k)
    b = np.random.randint(0, p-1, size=k)

    # Precompute hash values for all rows
    row_ids = np.arange(r)

    # Compute all hash functions on all row IDs
    hashes = ((a[:, None] * row_ids + b[:, None]) % p)   # Shape: (num_hashes, r)

    # Signature: for each hash function, find the minimum hashed row where b_matrix is 1
    signatures = np.full((k, N), np.inf)

    for row in range(r):
        mask = b_matrix[row] == 1
        signatures[:, mask] = np.minimum(signatures[:, mask], hashes[:, row][:, None])

    return signatures.astype(int)

def lsh(signatures, b, r_band):
    """
    signatures: k x N (rows = hash functions, columns = products)
    b: number of bands
    r_band: rows per band (band size)
    """
    k, N = signatures.shape
    assert k == b * r_band, "Signatures must have k = b * r rows."

    candidate_pairs = set()
    band_buckets = defaultdict(list)
    
    # Populate all buckets across all bands
    for product in range(N):
        sig = signatures[:, product]
        
        for band in range(b):
            start = band * r_band
            end = start + r_band
            band_tuple = tuple(sig[start:end])
            
            # The key is (band index, hash of band slice)
            band_buckets[(band, band_tuple)].append(product) 
            
    # Find all pairs in all populated buckets
    for bucket in band_buckets.values():
        if len(bucket) > 1:
            # Generate all unique pairs from the current bucket
            for i, j in combinations(bucket, 2):
                # Ensure the pair is stored consistently (i < j)
                candidate_pairs.add(tuple(sorted((i, j))))

    return candidate_pairs

def lsh_threshold(b, r_band):
    return (1 / b) ** (1 / r_band)

def find_suitable_k(r):
    k0 = r // 2
    search_limit = 20
    r_min = 4
    r_max = 20
    min_pairs = 6
    def factor_pairs(k):
        pairs = []
        for r in range(1, int(np.sqrt(k)) + 1):
            if k % r == 0:
                b = k // r
                pairs.append((b, r))
                if b != r:
                    pairs.append((r, b))
        return pairs

    for delta in range(search_limit):
        k_try = k0 + delta
        pairs = [(b, r) for (b, r) in factor_pairs(k_try)
                 if r_min <= r <= r_max]

        if len(pairs) >= min_pairs:
            return k_try
    
    return k0

def find_b_r_for_threshold(k, target_threshold):
    best_b = k
    best_r = 1
    best_diff = float('inf')

    for r in range(1, k+1):
        if k % r != 0:
            continue
        b = k // r
        threshold = lsh_threshold(b, r)
        diff = abs(threshold - target_threshold)
        if diff < best_diff:
            best_diff = diff
            best_b, best_r = b, r
    return best_b, best_r