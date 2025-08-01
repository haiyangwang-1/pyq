import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List
from .basic_similarity import BasicSimilarity

__all__ = ['partition_similarity']

def partition_similarity(
    points: np.ndarray, # (n_sample, n_feature) matrix
    partition: List[List[int]], # a partition on the features
    similarity: BasicSimilarity, # a callable to compute the similarity on each partition. 
    minimal_partition:2, # ignore small partitions. 
    ):
    
    '''
    points: np.ndarray
        the points to compute the similarity for. 
        shape: (n_points, n_features)
            
    partition: np.ndarray
        the partition of the features. 
        for each group of features, we compute the similarity between the points in this group. 
        
        then, we aggregate the similarity matrices. 
        
        the partitions is a list of lists, each list contains the indices of the features in the partition. 
    '''
        
    npts, _ = points.shape
    ret = np.zeros((npts,npts))
        
    for p in partition: 
        if len(p) < minimal_partition:
            continue
        ret += similarity(points[:,p])
    return ret    