class row_geometry:
    pass

class col_geometry:
    pass


class questionnaire:
    
    '''
    This class implements the questionnaire algorithm in the paper in doc/Ankenman**.pdf
    
    The main input is a dataset of shape (n_rows, n_cols, ...)
    and then, the algorithm will adapt to the geometry of the dataset, building trees on the 
    rows and columns leveraging diffusion geometry and Earth Mover's Distance. 
    
    The main output is the adaptive embedding of the dataset, which is called a 
    bi-Haar expansion. 
    '''
    
    def __init__(self,):
        pass
