# MTS-Classification-Kernels
Implementation of a binary classification on a small set of experiments with multiple sensors. The aim is to use problem specific kernels to then perform Kernel SVM.

# Approaches implemented:
1. Create a similarity matrix between experiments, using an Extended Frobenius norm. "A PCA-based Similarity Measure for Multivariate Time Series" by Yang and Shahabi
2. Create a feature vector to be ingested into an SVM Classifier of primal form. The feature vector derives from a kernel matrix, which models the similarity between the individual sensors through a Linear kernel projected into a tangent vector space (type of Manifold). "Multi-variate time series using kernel matrix" by Sun and Niu
3. Fisher Kernel. Note: gives still Nans for some elements of feature vector (less Nans for a better data pre-processing). "USING THE FISHER KERNEL METHOD FOR WEB AUDIO CLASSIFICATION" by Pedro J. Moreno and Ryan Rifkin
