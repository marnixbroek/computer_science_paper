from collections import Counter
from itertools import combinations
from LSH import compute_minhash_signatures, find_b_r_for_threshold, lsh
from dataFunctions import loadData
import os
import numpy as np
import matplotlib.pyplot as plt

from modelWordsFunctions import extract_model_id, extractModelWordsFromTitle, extractModelWordsFromValue, getAllModelWords, getMatchingModelWordsPercentage
from clusterFunctions import TMWMsimilarity, minFeatures
from sklearn.cluster import AgglomerativeClustering

# Global cache
_QGRAM_CACHE = {}
_QGRAM_DISTANCE_CACHE = {}
_QGRAM_SIM_CACHE = {}

# MODEL PARAMETERS

# Number of bootstrap samples
num_bootstrap_samples = 5
# Cluster distance threshold -> lower means fewer clusters, higher pair quality but lower pair completeness
epsilon = 0.522

def main():
    # Build a path to the JSON file relative to this script so
    # the file is found regardless of the current working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(here, "TVs-all-merged.json")
    data = loadData(filepath)    

    print(f"Loaded {len(data)} entries from {filepath}")

    np.random.seed(11)

    # Implement bootstrapping here.
    # Consider 5 bootstrap samples. First build the samples by drawing instances with replacement.
    
    n = len(data)
    thresholds = np.arange(0, 1.0001, 0.05)

    # Store LSH results per bootstrap sample and fraction of comparisons
    LSH_results = []
    clustering_results = []
    
    for b in range(num_bootstrap_samples):
        # Create bootstrap samples
        training_keys = np.random.choice(list(data.keys()), size=n, replace=True)
        unique_training_keys = set(training_keys)
        print(f"Bootstrap sample {b+1}: Training size = {len(unique_training_keys)}")        

        # Construct the training data dictionary
        training_data = {key: data[key] for key in unique_training_keys}

        # Build the set of all true duplicate pairs in the training dataset
        modelID_to_keys = mapModelIDsToKeys(training_data)
        true_duplicates = set()
        for keys in modelID_to_keys.values():
            if len(keys) > 1:
                for pair in combinations(keys, 2):
                    true_duplicates.add(tuple(sorted(pair)))
        
        # Perform LSH for thresholds between 0 and 1 with 0.05 step size:
        for threshold in thresholds:
            for method_name in ['MSMP+', 'MSMP+_modelID', 'MSMSP+', 'MSMSP+_modelID']:
                if method_name == 'MSMP+_modelID':
                    method = 'MSMP+'
                    add_modelID = True
                elif method_name == 'MSMSP+_modelID':
                    method = 'MSMSP+'
                    add_modelID = True
                else:
                    method = method_name
                    add_modelID = False
                
                candidate_pairs = obtainPairsLSH(training_data, threshold, method=method)
                
                # Add any pairs not found by LSH but having the same modelID based on extract_model_id
                if add_modelID:
                    candidate_pairs = addModelIDCandidatePairs(training_data, candidate_pairs)

                matched_true_duplicates = countMatchedTrueDuplicates(training_data, true_duplicates, candidate_pairs)

                # Find statistics
                fraction_comparisons = len(candidate_pairs) / (len(training_data) * (len(training_data) - 1) / 2)
                PQ_LSH = matched_true_duplicates / len(candidate_pairs) if len(candidate_pairs) > 0 else 0
                PC_LSH = matched_true_duplicates / len(true_duplicates) if len(true_duplicates) > 0 else 0
                F1_LSH = 2 * PQ_LSH * PC_LSH / (PQ_LSH + PC_LSH) if (PQ_LSH + PC_LSH) > 0 else 0

                LSH_results.append((method_name, threshold, fraction_comparisons, PQ_LSH, PC_LSH, F1_LSH))

                print("Bootstrap sample", b+1, "Threshold:", threshold, 
                    "Fraction comparisons:", fraction_comparisons,
                    "LSH PQ:", PQ_LSH, "PC:", PC_LSH, "F1*:", F1_LSH)

                # Now perform clustering
                distanceMatrix = computeDistances(training_data, alpha=0.602, beta=0.0, gamma=0.756, mu=0.650, candidate_pairs=candidate_pairs, add_modelID=add_modelID)
                clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=epsilon)
                clustering.fit(distanceMatrix)

                clusters = []
                for label in np.unique(clustering.labels_):
                    clusters.append(np.where(clustering.labels_ == label)[0])

                Dn = len(true_duplicates)

                # Compute Df: duplicates found in your clustering

                key_list = list(training_data.keys())

                Df = 0
                Nc = 0
                for cluster in clusters:
                    if len(cluster) < 2:
                        continue
                    # Map indices to actual keys
                    cluster_keys = [key_list[i] for i in cluster]
                    for pair in combinations(cluster_keys, 2):
                        if tuple(sorted(pair)) in true_duplicates:
                            Df += 1
                        Nc += 1

                # 4. Compute PQ, PC, F1
                PQ = Df / Nc if Nc > 0 else 0
                PC = Df / Dn if Dn > 0 else 0
                F1 = 2 * PQ * PC / (PQ + PC) if (PQ + PC) > 0 else 0

                print("Clustering results - PQ:", PQ, "PC:", PC, "F1:", F1)

                clustering_results.append((method_name, threshold, fraction_comparisons, PQ, PC, F1))

    computePlots(LSH_results, clustering_results)

def computePlots(LSH_results, clustering_results):
    # Extract method names (strings)
    method_names = list(set([result[0] for result in LSH_results]))

    # Convert lists → numpy arrays for easier slicing

    LSH_numeric = np.array([r[1:] for r in LSH_results], dtype=float)
    
    # Columns: method, threshold, frac_comparisons, PQ, PC, F1*
    frac_LSH   = LSH_numeric[:, 1]
    PQ_LSH     = LSH_numeric[:, 2]
    PC_LSH     = LSH_numeric[:, 3]
    F1_LSH     = LSH_numeric[:, 4]

    # Plot fraction of comparisons against average PC, PQ and F1* score for LSH and LSH + modelID-based pairs across all bootstrap samples

    # Pair completeness
    plt.figure(figsize=(8,5))
    for method in method_names:
        idx = [i for i, m in enumerate([r[0] for r in LSH_results]) if m == method]
        x = frac_LSH[idx]
        y = PC_LSH[idx]
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], y[sort_idx], label=f"LSH {method}")

    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair Completeness (PC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pair_completeness_vs_fraction_comparisons.png")

    # Pair quality
    plt.figure(figsize=(8,5))

    for method in method_names:
        # Indices of rows in original LSH_results with this method
        idx_method = [i for i, r in enumerate(LSH_results) if r[0] == method]

        # x, y for this method
        x = frac_LSH[idx_method]
        y = PQ_LSH[idx_method]

        # Sort by x to ensure left→right line
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], y[sort_idx], label=f"LSH {method}")

    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair Quality (PQ)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 0.25)
    plt.savefig("pair_quality_vs_fraction_comparisons.png")

    # F1*
    plt.figure(figsize=(8,5))

    for method in method_names:
        # Indices of rows in original LSH_results with this method
        idx_method = [i for i, r in enumerate(LSH_results) if r[0] == method]

        # x, y for this method
        x = frac_LSH[idx_method]
        y = F1_LSH[idx_method]

        # Sort by x to ensure left→right line
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], y[sort_idx], label=f"LSH {method}")

    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1*")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("F1_vs_fraction_comparisons.png")

    # Plot fraction of comparisons against average F1 score for clustering across all bootstrap samples
    plt.figure(figsize=(8,5))

    clustering_numeric = np.array([r[1:] for r in clustering_results], dtype=float)
    frac_vals_all = clustering_numeric[:,1]  # fraction of comparisons
    F1_vals_all   = clustering_numeric[:,4]  # F1

    for method in method_names:
        # Indices of rows in original clustering_results with this method
        idx_method = [i for i, r in enumerate(clustering_results) if r[0] == method]

        # x, y for this method
        x = frac_vals_all[idx_method]
        y = F1_vals_all[idx_method]

        # Sort by x to ensure left→right line
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], y[sort_idx], label=f"{method}")

    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Average F1-score (clustering)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("clustering_average_F1_vs_fraction_comparisons.png")

    auc_results = {
        "PC": {},
        "PQ": {},
        "F1": {},
        "ClusterF1": {}
    }

    # ---- AUC for PC, PQ, F1 ----
    for method in method_names:
        idx = [i for i, r in enumerate(LSH_results) if r[0] == method]

        x = frac_LSH[idx]

        auc_results["PC"][method] = compute_auc(x, PC_LSH[idx])
        auc_results["PQ"][method] = compute_auc(x, PQ_LSH[idx])
        auc_results["F1"][method] = compute_auc(x, F1_LSH[idx])

    # ---- AUC for clustering F1 ----
    for method in method_names:
        idx = [i for i, r in enumerate(clustering_results) if r[0] == method]
        x = frac_vals_all[idx]
        y = F1_vals_all[idx]
        auc_results["ClusterF1"][method] = compute_auc(x, y)

    # === Print pairwise percentage differences ===
    def print_differences(metric_name):
        print(f"\n=== {metric_name}: Pairwise AUC percentage differences ===")
        methods = list(auc_results[metric_name].keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                m1, m2 = methods[i], methods[j]
                auc1, auc2 = auc_results[metric_name][m1], auc_results[metric_name][m2]
                pct_diff = 100 * (auc1 - auc2) / auc2 if auc2 != 0 else float('inf')
                print(f"{m1} vs {m2}: {pct_diff:.2f}%")

    print_differences("PC")
    print_differences("PQ")
    print_differences("F1")
    print_differences("ClusterF1")

def compute_auc(x, y):
        """Compute area under the curve using trapezoidal rule."""
        x = np.array(x)
        y = np.array(y)
        sort_idx = np.argsort(x)
        return np.trapezoid(y[sort_idx], x[sort_idx])

def mapModelIDsToKeys(data):    
    modelID_to_keys = {}
    for key in data.keys():
        modelID = data[key]['modelID']
        if modelID not in modelID_to_keys:
            modelID_to_keys[modelID] = []
        modelID_to_keys[modelID].append(key)
    return modelID_to_keys

def addModelIDCandidatePairs(data, candidate_pairs):
    modelID_to_indices = {}
    for index, key in enumerate(data.keys()):
        modelID = extract_model_id(data[key]['title'])
        if modelID is not None:
            if modelID not in modelID_to_indices:
                modelID_to_indices[modelID] = []
            modelID_to_indices[modelID].append(index)
    for indices in modelID_to_indices.values():
        if len(indices) > 1:
            for pair in combinations(indices, 2):
                candidate_pairs.add(tuple(sorted(pair)))
    return candidate_pairs
    
def obtainPairsLSH(data, target_threshold, method):
    MW_title = set()
    MW_value = set()
    brands = set()
    count = 0
    for product in data.values():
        modelWordsTitle = extractModelWordsFromTitle(product['title'], method)
        MW_title.update(modelWordsTitle)
        modelWordsValueTemp = set()
        for feature, value in product['featuresMap'].items():
            modelWordsValue = extractModelWordsFromValue(feature, value, True, method)
            MW_value.update(modelWordsValue)
            modelWordsValueTemp.update(modelWordsValue)

    # Sort model words for consistent ordering
    MW_title = sorted(MW_title)
    MW_value = sorted(MW_value)

    b_matrix = []

    for product in data.values():
        # Create binary vector for product for each model word.
        # Should be indexed by model word.
        b_vector = []

        values = product['featuresMap'].values()
        for mw in MW_title:
            if mw in product['title'].lower() or mw in values:
                b_vector.append(1)
            else:
                b_vector.append(0)
        
        for mw in MW_value:
            if mw in values:
                b_vector.append(1)
            else:
                b_vector.append(0)
        
        
        b_matrix.append(b_vector)

    b_matrix = np.array(b_matrix).T


    signatures = compute_minhash_signatures(b_matrix)
    k, N = signatures.shape
    b, r_band = find_b_r_for_threshold(k, target_threshold)

    candidate_pairs = lsh(signatures, b, r_band)

    return candidate_pairs

def computeDistances(data, alpha, beta, gamma, mu, candidate_pairs, add_modelID):
    '''Perform clustering on the data based on certain criteria.'''

    distanceMatrix = {}
    
    # Number of products
    n = len(data)
    i = 0

    key_list = list(data.keys())

    # Loop over all candidate pairs
    for index_key, index_other_key in candidate_pairs:
        key = key_list[index_key]
        other_key = key_list[index_other_key]

        # Compare key and other_key products here

        # Extract model ids from titles
        model_id_key = extract_model_id(data[key]['title'])
        model_id_other_key = extract_model_id(data[other_key]['title'])
        
        # Set distance to infinite if same shop, different brand (brand is not always present) or not a candidate pair
        sameShop = str.lower(data[key]['shop']) == str.lower(data[other_key]['shop'])

        # Check if both have brand info, if not, assume same brand
        # Sometimes, feature name is not exactly 'brand', but it contains 'brand'
        brand_keys1 = [k for k in data[key]['featuresMap'].keys() if 'brand' in k.lower()]
        brand_keys2 = [k for k in data[other_key]['featuresMap'].keys() if 'brand' in k.lower()]

        if brand_keys1 and brand_keys2:
            brand1 = str.lower(data[key]['featuresMap'][brand_keys1[0]])
            brand2 = str.lower(data[other_key]['featuresMap'][brand_keys2[0]])
            brandConflict = brand1 != brand2
        else:
            brandConflict = False
        
        if model_id_key == model_id_other_key and add_modelID:
            distance = 0.0
        elif sameShop or brandConflict:
            distance = float('inf')
        else:
            sim = 0
            avgSim = 0
            m = 0
            w = 0

            # We look at all the non-matching features in featuresMap
            # Obtain list of all non-matching feature keys per product
            features1 = set(data[key]['featuresMap'].keys())
            features2 = set(data[other_key]['featuresMap'].keys())

            non_matching_keys1 = features1 - features2
            non_matching_keys2 = features2 - features1

            for feature_key in features1:
                for feature_key2 in features2:
                    keySim = qGramSimilarity(str(feature_key), str(feature_key2), 3)
                    if keySim > gamma:
                        valueSim = qGramSimilarity(str(data[key]['featuresMap'][feature_key]), str(data[other_key]['featuresMap'][feature_key2]), 3)
                        weight = keySim
                        sim = sim + weight * valueSim
                        m += 1
                        w += weight
                        # Remove matched keys from non-matching sets if necessary
                        non_matching_keys1.discard(feature_key)
                        non_matching_keys2.discard(feature_key2)
            if w > 0:
                avgSim = sim / w
            
            mwPerc = getMatchingModelWordsPercentage(getAllModelWords(data, key, non_matching_keys1), getAllModelWords(data, other_key, non_matching_keys2))
            titleSim = TMWMsimilarity(data, key, other_key, alpha, beta)

            if titleSim == -1:
                theta_1 = m / minFeatures(data, key, other_key)
                theta_2 = 1 - theta_1
                hSim = theta_1 * avgSim + theta_2 * mwPerc
            else:
                theta_1 = (1 - mu) * m / minFeatures(data, key, other_key)
                theta_2 = 1 - mu - theta_1
                hSim = theta_1 * avgSim + theta_2 * mwPerc + mu * titleSim

            distance = 1 - hSim
        
        distanceMatrix[(key, other_key)] = distance

    key_list = list(data.keys())
    n = len(key_list)
    MAX_DIST = 1e6  # Must be > epsilon

    distanceMatrix_np = np.array([
        [
            distanceMatrix.get((key_list[i], key_list[j]),
            distanceMatrix.get((key_list[j], key_list[i]), MAX_DIST))
            for j in range(n)
        ]
        for i in range(n)
    ], dtype=np.float64)

    # Replace any remaining inf/NaN with MAX_DIST
    distanceMatrix_np = np.nan_to_num(distanceMatrix_np, nan=MAX_DIST, posinf=MAX_DIST, neginf=MAX_DIST)

    # Ensure diagonal is 0
    np.fill_diagonal(distanceMatrix_np, 0.0)

    distanceMatrix = distanceMatrix_np
    
    return distanceMatrix

def countMatchedTrueDuplicates(data, true_duplicates, candidate_pairs):
    # Check what percentage of true duplicate pairs are in candidate pairs
    matched_true_duplicates = 0
    key_list = list(data.keys())
    key_to_index = {k: i for i, k in enumerate(data.keys())}
    for pair in true_duplicates:
        index1 = key_to_index[pair[0]]
        index2 = key_to_index[pair[1]]
        if (index1, index2) in candidate_pairs or (index2, index1) in candidate_pairs:
            matched_true_duplicates += 1
    return matched_true_duplicates

def get_qgram_counter_cached(s, q):
    key = (s, q)
    if key not in _QGRAM_CACHE:
        _QGRAM_CACHE[key] = Counter(s[i:i+q] for i in range(len(s) - q + 1))
    return _QGRAM_CACHE[key]

def qgrams(s, q):
    """Generate a list of q-grams for the given string s."""
    return [s[i:i+q] for i in range(len(s) - q + 1)]

def qGramDistance(s1, s2, q):
    """Calculate the q-gram distance between two strings."""
    key = (s1, s2, q)
    if key in _QGRAM_DISTANCE_CACHE:
        return _QGRAM_DISTANCE_CACHE[key]

    c1 = get_qgram_counter_cached(s1, q)
    c2 = get_qgram_counter_cached(s2, q)

    keys = c1.keys() | c2.keys()

    dist = 0
    for k in keys:
        dist += abs(c1.get(k, 0) - c2.get(k, 0))

    _QGRAM_DISTANCE_CACHE[key] = dist
    return dist

def qGramSimilarity(s1, s2, q):
    """Calculate the q-gram similarity between two strings."""
    key = (s1, s2, q)
    if key in _QGRAM_SIM_CACHE:
        return _QGRAM_SIM_CACHE[key]

    n1 = len(s1)
    n2 = len(s2)
    dist = qGramDistance(s1, s2, q)

    sim = (n1 + n2 - dist) / (n1 + n2)
    _QGRAM_SIM_CACHE[key] = sim
    return sim

if __name__ == "__main__":
    print("Running main.py")
    main()