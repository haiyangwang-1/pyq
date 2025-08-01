import plotly.express as px

# Assume 'embedding' is a numpy array of shape (n_samples, n_features)
# Plot the first three dimensions

def scatter_3d(embedding,labels,dims=[0,1,2],marker_size=4,title=None):
    
    
    fig = px.scatter_3d(
        x=embedding[:, dims[0]],
        y=embedding[:, dims[1]],
        z=embedding[:, dims[2]],
        title=title,
        labels={
            'x': f'Dim {dims[0]}', 
            'y': f'Dim {dims[1]}', 
            'z': f'Dim {dims[2]}'
        },
        color=labels,
    )
    
    fig.update_traces(marker_size=marker_size)
    
    return fig
