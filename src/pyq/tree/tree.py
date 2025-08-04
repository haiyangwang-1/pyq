import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Iterator


class Tree:
    """
    A Tree class using the NetworkX digraph as the backend.
    
    Each node has the following attributes:
    - idx: list[int], marks the columns/rows of the dataset that the node represents. 
    - 
    """
    
    def __init__(self, npts: int, **root_attrs):
        """
        the tree is a tree on npts of data. 
        """
        
        
        assert npts > 0, "npts must be positive"
        assert isinstance(npts, int), "npts must be an integer"
        
        self._npts = npts        
        self._graph = nx.DiGraph()
        self._root_id = 0
        self._next_id = 1
        
        # initialize the root nodes
        root_idx = np.arange(self._npts,dtype=int)
        self._graph.add_node(self._root_id, idx=root_idx, **root_attrs)
    
    @property
    def npts(self) -> int:
        return self._npts
        
    def add_child(self, parent_id: int, child_idx: list[int], **child_attrs) -> int:
        """Add a child to a parent node with data and additional attributes."""
        child_id = self._next_id
        self._next_id += 1
        
        # Add node with data and any additional attributes
        self._graph.add_node(child_id, idx=child_idx, **child_attrs)
        self._graph.add_edge(parent_id, child_id)
        
        return child_id
    
        
    def get_children(self, node_id: int) -> List[int]:
        """Get all children of a node."""
        return list(self._graph.successors(node_id))
    
    def get_parent(self, node_id: int) -> Optional[int]:
        """Get the parent of a node."""
        parents = list(self._graph.predecessors(node_id))
        assert len(parents) <= 1, "A node can have at most one parent"
        return parents[0] if parents else None
        
    def __getitem__(self, node_id: int) -> Any:
        return self._graph.nodes[node_id]
        
    def leaves(self) -> List[int]:
        """Get all leaf nodes (nodes with no children)."""
        return [node for node in self._graph.nodes() if self._graph.out_degree(node) == 0]
    
    def level(self, node_id: int) -> int:
        '''distance to the root node'''
        return nx.shortest_path_length(self._graph, self._root_id, node_id)
    
    def distance2leaf(self, node_id: int) -> int:
        """distance to the nearest leaf"""
        return min(nx.shortest_path_length(self._graph, node_id, leaf) \
            for leaf in self.leaves())
    
    def depth(self, node_id: int) -> int:
        '''distance to the farthest leaf'''
        
        '''
        suppose that there are levels 0, 1, 2, ..., L-1
        then depth of the root node is L-1. 
        '''
        if self.get_children(node_id):
            return max(self.depth(child) for child in self.get_children(node_id)) + 1
        return 0
    
    def bfs(self, node_id: Optional[int] = None) -> Iterator[int]:
        """Breadth-first traversal starting from root."""
        if node_id is None:
            '''starts traversal from the root'''
            node_id = self._root_id
        queue = [node_id]
        while queue:
            curr = queue.pop(0)
            yield curr
            queue.extend(self.get_children(curr))
    
    def compute_local_mean(self, data, label:str):
        
        '''
        given the data in the shape (npts, ...)
        and self is the tree on the npts axis, 
        we express the data as cumsum from root to leaf of the tree. 
        
        This gives the multi-scale resolution that will be used for 
        computing the EMDs. 
        
        '''
        
        assert data.shape[0] == self._npts, 'data must have shape (npts, ...)'

        for node_id in self.bfs():
            idx = self[node_id]['idx']
            mean = np.mean(data[idx,:],axis=0)
            self[node_id][label+'_mean'] = mean
            
    def compute_inverse_cumsum(self, label:str):
        
        '''
        inverse of cumsum. 
        taking cumsum from top to bottom would recover the original data. 
        therefore this is the inverse of cumsum. 
        '''
        
        
        assert label + '_mean' in self[self._root_id], \
            f'local mean of {label} must be computed first'
        
        for node_id in self.bfs():
            parent_id = self.get_parent(node_id)
            parent_mean = self[parent_id][label+'_mean'] if parent_id else 0
            current_mean = self[node_id][label+'_mean']
            self[node_id][label+'_inverse_cumsum'] = current_mean - parent_mean
        
    def compute_cumsum_expansion(self, label:str, 
            alpha:float=1,
            beta: float=0,
            l_start:int=1,
            l_end:Optional[int]=None,
            ):
        
        '''
        weight the local cumsum expansion by size and level of the node, and return the expansion. 
        '''
        
        if l_end is None:
            l_end = self.depth(self._root_id)
        
        assert label + '_inverse_cumsum' in self[self._root_id], \
            f'inverse cumsum of {label} must be computed first'
        
        data_shape = self[self._root_id][label+'_inverse_cumsum'].shape
        coefs = []
        
        nodes_by_level = self.nodes_by_level()
        
        for l in range(l_start,l_end):
            nodes_l = nodes_by_level[l]
            sizes_l = np.array([len(self[node_id]['idx']) for node_id in nodes_l])
            sizes_l = sizes_l/sizes_l.sum()
            
            coef_l_shape = (len(nodes_l),) + data_shape
            coef_l = np.zeros(coef_l_shape)
            
            for i,(node_id,size) in enumerate(zip(nodes_l,sizes_l)):
                coef_l[i] = self[node_id][label+'_inverse_cumsum'] \
                    * size ** beta * 2**(-alpha*l)
            
            coefs.append(coef_l)
        
        # the final shape of coefs is (n_nodes, ...)
        return np.concatenate(coefs,axis=0)
        
    def nodes_by_level(self):
        
        nodes_by_level = []
        for node_id in self.bfs():
            l = self.level(node_id)
            if l == len(nodes_by_level):
                nodes_by_level.append([])
            nodes_by_level[l].append(node_id)
            
        return nodes_by_level
            
    def partition_by_level(self, level:int) -> List[List[int]]:
        
        '''
        partition the points by the all the nodes at the given level, 
        as well as the leaf nodes that are at a lower level (closer to the root). 
        '''
        nodes = []
        for node_id in self.bfs():
            if self.level(node_id) > level:
                break
            elif self.level(node_id) < level:
                if not self.get_children(node_id):
                    # this is a non-leaf node at lower level, ignored. 
                    continue
            else: # self.level(node_id) == level
                pass
            
            nodes.append(node_id)
            
        return [self[n]['idx'] for n in nodes]
    
    def is_binary(self) -> bool:
        '''
        check if the tree is binary. 
        '''
        for node_id in self.bfs():
            if len(self.get_children(node_id)) > 2:
                return False
        return True