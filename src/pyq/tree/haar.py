'''
Given a single tree, we can compute the Haar expansion. 

suppose the tree only has the root node, then there is only one 
Haar function, which is the constant. 

The number of Haar expansion equals to the number of leaves. 
'''
import numpy as np


class HaarExpansion:

    def __init__(self, tree):
        
        assert tree.is_binary(), "Haar expansion is only defined for binary trees."
        
        self.tree = tree
        self.compute_expansion()
        
        
    def compute_expansion(self):

        funcs = []

        # compute the expansion for the root node        
        func = np.ones(self.tree.npts) 
        func = func/np.sqrt(np.dot(func,func)) # L2-normalized 
        funcs.append(func)

        # compute further expansions down the tree. . 
        to_be_computed = [self.tree._root_id]
        while to_be_computed:
            curr_id = to_be_computed.pop(0)

            children = self.tree.get_children(curr_id)
            match len(children):
                case 0: 
                    continue
                case 1: 
                    assert False, "This should not happen."
                case 2: 
                    pass
                case _: 
                    assert False, "This should not happen."
            # compute the expansion for the left and right children. 
            
            left_id, right_id = children
            
            left_size = len(self.tree[left_id]['idx'])
            right_size = len(self.tree[right_id]['idx'])
            
            func = np.zeros(self.tree.npts)
            func[self.tree[left_id]['idx']] = -right_size
            func[self.tree[right_id]['idx']] = left_size
            func = func/np.sqrt(np.dot(func,func)) # L2-normalized 
            funcs.append(func)
            
            # add the left and right children to the to_be_computed list. 
            to_be_computed.extend([left_id, right_id])
        
        self.haar_funcs = np.array(funcs)
        
