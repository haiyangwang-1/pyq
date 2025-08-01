from .tree_node import TreeNode
import numpy as np
from sklearn.cluster import KMeans
from diffusion_map import DiffusionMap



def build_binary_tree_by_median_split(
    similarity,
    min_size=10,
    max_depth=5,
):
    
    '''
    Build a binary tree by splitting around the median of the similarity matrix. 
    
    Input: 
        similarity (n_samples, n_samples)
        
    Output: 
        Tree
    
    '''
    
    root = TreeNode(data=np.arange(similarity.shape[0],dtype=int))
    
    to_be_splitted = [root]
    
    n_clusters = 2 # binary tree
    
    while to_be_splitted:
        curr = to_be_splitted.pop(0)
        idx  = curr.data
        
        # bunch of stopping conditions
        if len(idx) <= min_size:
            continue
            
        if curr.distance2root() > max_depth:
            continue
        
        # computes only the top eigenvector
        emb = DiffusionMap(similarity[idx,:][:,idx],n_dim=1).transform(t=0).flatten()
        
        median = np.median(emb)
        
        left  = TreeNode(idx[emb < median])
        right = TreeNode(idx[emb >= median])
        
        curr.add_child(left)
        curr.add_child(right)
        
        to_be_splitted.append(left)
        to_be_splitted.append(right)
        
    return root
    
    

def build_binary_tree_by_kmeans(
    embedding,
    min_size=10,
    max_depth=5):
    
    '''
    Build a binary tree on the embedding. 
    
    Input:
        embedding (n_samples, n_features)
    
    Output:
        root (TreeNode)
    
    '''
    
    root = TreeNode(data=np.arange(embedding.shape[0],dtype=int))
    
    to_be_splitted = [root]
    
    n_clusters = 2 # binary tree
        

    while to_be_splitted:
        curr = to_be_splitted.pop(0)
        idx  = curr.data
    
        # bunch of stopping conditions
        if len(idx) <= min_size:
            continue

        if curr.distance2root() > max_depth:
            continue
    
        clusters = KMeans(n_clusters=n_clusters).fit(embedding[idx,:]).labels_
        
        for c in range(n_clusters):
            elem = idx[clusters==c]
            child = TreeNode(elem)
            curr.add_child(child)
            to_be_splitted.append(child)

    return root
