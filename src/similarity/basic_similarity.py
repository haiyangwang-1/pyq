"""
This module contains the similarity measures for the tree.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from abc import ABC, abstractmethod

__all__ = [
    'GaussianSimilarity',
    'PearsonSimilarity',
    'CorrelationSimilarity',
    'CosineSimilarity',
    'Exponential_Cityblock_Similarity'
    ]


class BasicSimilarity(ABC):
    
    @abstractmethod
    def __call__(self,points: np.ndarray) -> np.ndarray:
        '''
        Given points of shape (n_points, n_features), return a similarity matrix of shape (n_points, n_points).
        
        which is a symmetric matrix with non-negative entries. 
        '''
        
        pass
    
class GaussianSimilarity(BasicSimilarity):
    
    '''
    this is essentially the heat kernel similarity. 
    where the similarity is defined as 
    
        (2*pi*t)**(-d/2) * exp(-r^2 / (2 * sigma**2))
    
    the constant pre-factor is ignored in the implementation. 
    '''
    
    def __init__(self,sigma: float, *args, **kwargs):
        self.sigma = sigma
        
    def __call__(self,points: np.ndarray) -> np.ndarray:
        rsq = pdist(points,metric='sqeuclidean')
        val = np.exp(-rsq / (2 * self.sigma**2))
        return squareform(val)
    
class PearsonSimilarity(BasicSimilarity):
    
    '''
    this is designed for the binary {-1,1} data.
    
    the similarity of two points x1 and x2 is defined as 
    
        p = mean(x1==1)
        q = mean(x2==1)
        j = mean(x1==1 & x2==1)
        
        then the similarity is 
        
        (1)   alpha = (j-pq) / sqrt( (1-p)(1-q)pq ) 
    
        in case the denominator is 0, the similarity is set to 0. 
        
    later, we have a thresholding step to make the similarity matrix positive. 
        
        alpha = max(alpha,0)
    
    '''
    
    
    def __init__(self,threshold: float=0, *args, **kwargs):
    
        assert threshold >= 0, "the threshold must be non-negative"
        self.threshold = threshold
    
    def __call__(self,points: np.ndarray) -> np.ndarray:
        
        assert np.all([i in [-1,1] for i in np.unique(points.flatten()).tolist()]), "the data must be binary {-1,1}"
            
        p = np.mean(points==1,axis=1).reshape(-1,1)
        q = p.reshape(1,-1)
        j = np.mean(points[:,None,:] + points[None,:,:] == 2, axis=2)
        
        # compute the similarity
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = (j-p*q) / np.sqrt((1-p)*(1-q)*p*q)
        
        # set the similarity to 0 if the denominator is 0
        flag = (1-p)*(1-q)*p*q == 0
        similarity[flag] = 0
        
        # threshold the similarity
        similarity = np.maximum(similarity,self.threshold)
        
        return similarity
    
class CorrelationSimilarity(BasicSimilarity):
    '''
    similarity is defined as the threshold value of the correlation coefficient. 
    '''
    
    def __init__(self,threshold,*arg,**kwargs) -> None:
        assert threshold >= 0, 'the threshold must be non-negative'
        self.threshold = threshold

    def __call__(self, points: np.ndarray)->np.ndarray:
        return np.maximum(np.corrcoef(points),self.threshold)
    
class CosineSimilarity(BasicSimilarity):
    def __init__(self, threshold, *args, **kwargs) -> None:
        assert threshold >= 0, 'the threshold must be non-negative'
        self.threshold = threshold

    def __call__(self, points: np.ndarray)->np.ndarray:
        return np.maximum(
            1-squareform(pdist(points,metric='cosine')),
            self.threshold
        )
        
        
class Exponential_Cityblock_Similarity(BasicSimilarity):
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, points: np.ndarray)->np.ndarray:
        r = squareform(pdist(points,metric='cityblock'))
        sigma = 2*np.median(r)
        return np.exp(-r/sigma)
    
    