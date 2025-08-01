from typing import List
import numpy as np

class TreeNode:
    
    def __init__(self,data):
        self.data = data
        self._children = []
        self.parent = None
        
        
    def add_child(self,child: 'TreeNode'):
        assert child.parent is None, "Child already has a parent"
        child.parent = self
        self._children.append(child)
        
    def remove_child(self,child: 'TreeNode'):
        assert child.parent is self, "Child is not a child of this node"
        child.parent = None
        self._children.remove(child)
    
    def get_children(self) -> List['TreeNode']:
        return self._children
    
    def distance2root(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.distance2root() + 1
    
    def distance2leaf(self) -> int:
        if len(self._children) == 0:
            return 0
        return min([child.distance2leaf() for child in self._children]) + 1
    
    def depth(self) -> int:
        if self.get_children():
            return max([child.depth() for child in self._children]) + 1
        return 0
    
    def leaves(self) -> List['TreeNode']:
        if len(self._children) == 0:
            return [self]
        return sum([child.leaves() for child in self._children], [])
    
    def bfs(self):
        queue = [self]
        while queue:
            current = queue.pop(0)
            yield current
            queue.extend(current.get_children())
            
    def compute_coef_and_dual_embedding(self,data,axis=0): 
        
        '''
        the tree is a tree on the axis of the data. 
        we need to take mean w.r.t. the axis. 
        '''
        
        assert self.parent is None, "only compute coef from the root node"
        assert len(self.data) == data.shape[axis], "data shape mismatch"


        to_be_computed = [self]

        while to_be_computed:
            current = to_be_computed.pop(0)
            
            # TODO: this is a hack to make it work for rows and cols,
            # we should generalize this later. 
            
            assert axis in [0,1], "axis must be 0 or 1"
            
            if axis==1:
                current_mean = np.mean(data[:,current.data],axis=axis)
            else: 
                current_mean = np.mean(data[current.data,:],axis=axis)
            
            prev_mean = current.parent.coef if current.parent else 0
            current_coef = current_mean - prev_mean
            current.coef = current_coef
            
            to_be_computed.extend(current.get_children())
            
        
        nodes = [node for node in self.bfs()]
        nodes_by_level = [[node for node in nodes if node.distance2root() == l] for l in range(self.depth())]

            
        alpha = 1
        beta = 0
        l_start = 1
        l_end = min(5,self.depth())
        
        # computes the dual embedding here 
        coefs = []
        for l in range(l_start,l_end):
            nodes_l = nodes_by_level[l]
            sizes_l = np.array([len(node.data) for node in nodes_l])
            sizes_l = sizes_l/sizes_l.sum()
            
            data_shape = data.shape
            coef_l_shape = data_shape[:axis] + (len(nodes_l),) + data_shape[axis+1:] 
            coef_l = np.zeros(coef_l_shape)
            
            
            for i,(node,size) in enumerate(zip(nodes_l,sizes_l)):
                indices = (slice(None),) * axis + (i,) + (slice(None),) * (data.ndim - axis - 1)


                coef_l[indices] = node.coef * size ** beta * 2**(alpha*(l-l_end)/2)
                
            coefs.append(coef_l)
            
        coefs = np.concatenate(coefs,axis=axis)
        
        return coefs
        
        
        
        
        
        