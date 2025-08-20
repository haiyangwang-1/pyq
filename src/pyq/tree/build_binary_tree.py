import numpy as np
from sklearn.cluster import KMeans

from diffusion_map import DiffusionMap
from .tree import Tree

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
    
    tree = Tree(similarity.shape[0])
    to_be_splitted = list(tree.bfs())
    
    while to_be_splitted:
        node_id = to_be_splitted.pop(0)
        idx  = tree[node_id]['idx']
        
        # bunch of stopping conditions
        if len(idx) <= min_size:
            continue
            
        if tree.level(node_id) > max_depth:
            continue
        
        # computes only the top eigenvector
        emb = DiffusionMap(similarity[idx,:][:,idx],n_dim=1).transform(t=0).flatten()
        
        median = np.median(emb)
        
        left_idx  = idx[emb < median]
        right_idx = idx[emb >= median]
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        
        left_id = tree.add_child(node_id, left_idx)
        right_id = tree.add_child(node_id, right_idx)
        
        to_be_splitted.append(left_id)
        to_be_splitted.append(right_id)
        
    return tree
    
    

def build_binary_tree_by_kmeans(
    embedding,
    min_size=10,
    max_depth=5):

    # TODO 
    raise NotImplementedError('not implemented')

    
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
