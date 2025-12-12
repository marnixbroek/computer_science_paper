This code is used to evaluate the methods proposed in the paper Improving LSH results for MSMP+ by increasing the specificity in selecting model words and extracting model IDs, written by Marnix Broek for the individual assignment of Computer Science for Business Analytics. 

The code can be executed from main.py and uses functions from the custom libraries clusterFunctions.py (for the calculation of dissimilarities), dataFunctions.py (for loading and cleaning the data), LSH.py (for the implementation of Min Hashing and Locality Sensitive Hashing) and modelWordsFunctions.py (for all functions related to model words).

To run the code, you can set the following parameters:

num_bootstrap_samples: the number of bootstrap samples
epsilon: the cluster distance threshold -> lower means fewer clusters, higher pair quality but lower pair completeness

The other parameters are fixed and should not be changed. As output, four graph pngs will be saved (pair_completeness_vs_fraction_comparisons.png, pair_quality_vs_fraction_comparisons.png, F1_vs_fraction_comparisons.png and clustering_average_F1_vs_fraction_comparisons.png). Also, the differences in area under the curve (AUC) between all different models in all graphs will be printed for ease of comparison.