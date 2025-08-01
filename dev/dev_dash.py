import dash
from dash import dcc, html, Input, Output, dash_table
import dash_cytoscape as cyto
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import socket

# --- 1. Create a More Meaningful Sample Dataset ---
# Create a dataset with clear clusters for better tree visualization
np.random.seed(42)
n_samples = 80
n_features = 4

# Create clustered data for more meaningful tree splits
X, y = make_blobs(n_samples=n_samples, centers=4, n_features=n_features, 
                  random_state=42, cluster_std=1.5)
feature_names = [f'Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['cluster'] = y

# --- 2. Build a More Meaningful Binary Tree Structure ---
# Split based on feature values rather than arbitrary indices

nodes = []
edges = []
node_to_data_indices = {}
node_to_split_info = {}
node_counter = 0

def find_best_split(data_indices):
    """Find the best feature and threshold to split the data."""
    if len(data_indices) <= 1:
        return None, None, None, None
    
    node_data = df.iloc[data_indices]
    best_feature = None
    best_threshold = None
    best_left_indices = []
    best_right_indices = []
    max_separation = 0
    
    # Try splitting on each feature
    for feature in feature_names:
        feature_values = node_data[feature].values
        threshold = np.median(feature_values)
        
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        
        left_indices = [data_indices[i] for i, mask in enumerate(left_mask) if mask]
        right_indices = [data_indices[i] for i, mask in enumerate(right_mask) if mask]
        
        # Ensure both splits have data
        if len(left_indices) > 0 and len(right_indices) > 0:
            # Calculate separation (simple heuristic: difference in means)
            left_mean = np.mean(feature_values[left_mask])
            right_mean = np.mean(feature_values[right_mask])
            separation = abs(left_mean - right_mean)
            
            if separation > max_separation:
                max_separation = separation
                best_feature = feature
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices
    
    return best_feature, best_threshold, best_left_indices, best_right_indices

# Global variable to track positions
node_positions = {}

def build_tree_recursive(indices, level, parent_split_info="Root", x_offset=0):
    """
    Recursively builds the tree with meaningful splits and proper positioning.
    """
    global node_counter, node_positions
    
    if len(indices) == 0:
        return None
    
    # Create the current node
    node_id = f'node-{node_counter}'
    n_samples_node = len(indices)
    
    # Create more informative label
    if level == 0:
        node_label = f'Root\n(n={n_samples_node})'
    else:
        node_label = f'Node {node_counter}\n(n={n_samples_node})\n{parent_split_info}'
    
    # Calculate node statistics
    node_data = df.iloc[indices]
    node_stats = {
        'mean': node_data[feature_names].mean().to_dict(),
        'std': node_data[feature_names].std().to_dict(),
        'min': node_data[feature_names].min().to_dict(),
        'max': node_data[feature_names].max().to_dict()
    }
    
    # Calculate position for better tree layout
    x_position = x_offset
    y_position = level * 200
    
    nodes.append({
        'data': {'id': node_id, 'label': node_label},
        'position': {'x': x_position, 'y': y_position}
    })
    node_to_data_indices[node_id] = indices
    node_to_split_info[node_id] = {
        'level': level,
        'parent_info': parent_split_info,
        'stats': node_stats,
        'n_samples': n_samples_node
    }
    
    current_node_idx = node_counter
    node_counter += 1

    # Stop if we reach the desired depth or have too few samples
    if level >= 2 or len(indices) <= 5:
        return current_node_idx

    # Find best split
    split_feature, threshold, left_indices, right_indices = find_best_split(indices)
    
    if split_feature is None:
        return current_node_idx

    # Store split information
    node_to_split_info[node_id]['split_feature'] = split_feature
    node_to_split_info[node_id]['split_threshold'] = threshold

    # Calculate child positions
    child_spacing = 300 / (2 ** level)  # Decreasing spacing for each level
    
    # Recursively build left child
    if left_indices:
        left_split_info = f"{split_feature} ≤ {threshold:.2f}"
        left_x = x_position - child_spacing
        left_child_idx = build_tree_recursive(left_indices, level + 1, left_split_info, left_x)
        if left_child_idx is not None:
            edges.append({'data': {'source': f'node-{current_node_idx}', 'target': f'node-{left_child_idx}'}})

    # Recursively build right child
    if right_indices:
        right_split_info = f"{split_feature} > {threshold:.2f}"
        right_x = x_position + child_spacing
        right_child_idx = build_tree_recursive(right_indices, level + 1, right_split_info, right_x)
        if right_child_idx is not None:
            edges.append({'data': {'source': f'node-{current_node_idx}', 'target': f'node-{right_child_idx}'}})
        
    return current_node_idx

# Start building the tree from the root node
initial_indices = list(range(n_samples))
build_tree_recursive(initial_indices, 0, "Root", 400)  # Start root at center position

# Combine nodes and edges into a single list for Cytoscape
cyto_elements = nodes + edges

# Debug: Print tree structure to verify it's being built
print(f"🌳 Tree built successfully: {len(nodes)} nodes, {len(edges)} edges")
print(f"📊 Dataset: {n_samples} samples, {n_features} features")
if len(nodes) > 0:
    print(f"✅ Root node: {nodes[0]['data']['id']}")
else:
    print("❌ No nodes created - check tree building logic")

# --- 3. Initialize the Dash App ---
app = dash.Dash(__name__)
server = app.server

# --- 4. Define the App Layout with Better Styling ---
app.layout = html.Div(style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0',
    'padding': '0',
    'backgroundColor': '#f8f9fa'
}, children=[
    html.H1("Interactive Binary Tree Data Explorer", style={
        'textAlign': 'center',
        'color': '#2c3e50',
        'marginBottom': '30px',
        'padding': '20px',
        'backgroundColor': 'white',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    html.Div(className='app-container', style={
        'display': 'flex', 
        'flexDirection': 'row',
        'gap': '20px',
        'padding': '0 20px',
        'minHeight': '80vh'
    }, children=[
        
        # Left Panel: Cytoscape Graph for the Tree
        html.Div(className='left-panel', style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }, children=[
            html.H3("Binary Tree Structure", style={
                'marginBottom': '20px',
                'color': '#34495e',
                'borderBottom': '2px solid #3498db',
                'paddingBottom': '10px'
            }),
            cyto.Cytoscape(
                id='cytoscape-tree',
                elements=cyto_elements,
                style={'width': '100%', 'height': '70vh'},
                layout={
                    'name': 'preset',
                    'padding': 30,
                    'fit': True
                },
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'label': 'data(label)',
                            'background-color': '#3498db',
                            'color': 'white',
                            'text-halign': 'center',
                            'text-valign': 'center',
                            'shape': 'roundrectangle',
                            'width': '120px',
                            'height': '80px',
                            'font-size': '11px',
                            'text-wrap': 'wrap',
                            'text-max-width': '100px',
                            'border-width': '2px',
                            'border-color': '#2980b9'
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'width': 3,
                            'line-color': '#34495e',
                            'target-arrow-color': '#34495e',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    },
                    {
                        'selector': ':selected',
                        'style': {
                            'background-color': '#e74c3c',
                            'border-color': '#c0392b',
                            'line-color': '#e74c3c'
                        }
                    }
                ]
            )
        ]),
        
        # Right Panel: Node Information and Plots
        html.Div(className='right-panel', style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }, children=[
            html.H3("Node Analysis", style={
                'marginBottom': '20px',
                'color': '#34495e',
                'borderBottom': '2px solid #e74c3c',
                'paddingBottom': '10px'
            }),
            
            html.Div(id='node-info-output', style={
                'backgroundColor': '#ecf0f1',
                'padding': '15px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'minHeight': '60px'
            }),
            
            html.Div(style={'marginBottom': '20px'}, children=[
                html.H4("Visualization Options", style={'color': '#34495e', 'marginBottom': '10px'}),
                html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                    html.Div(style={'flex': '1'}, children=[
                        html.Label("Feature:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[{'label': name.replace('_', ' '), 'value': name} for name in feature_names],
                            value=feature_names[0],
                            style={'fontSize': '14px'}
                        )
                    ]),
                    html.Div(style={'flex': '1'}, children=[
                        html.Label("Visualization:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='viz-type-dropdown',
                            options=[
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Statistics Table', 'value': 'stats'},
                                {'label': 'Feature Comparison', 'value': 'comparison'}
                            ],
                            value='histogram',
                            style={'fontSize': '14px'}
                        )
                    ])
                ])
            ]),
            
            html.Div(id='visualization-output')
        ])
    ])
])

# --- 5. Define Enhanced Callbacks for Interactivity ---
@app.callback(
    [Output('node-info-output', 'children'),
     Output('visualization-output', 'children')],
    [Input('cytoscape-tree', 'tapNodeData'),
     Input('feature-dropdown', 'value'),
     Input('viz-type-dropdown', 'value')]
)
def display_node_data(node_data, selected_feature, viz_type):
    # Default state when no node is clicked
    if not node_data:
        default_info = html.Div([
            html.H4("🔍 Select a Node", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Click on any node in the tree to explore its data distribution and statistics.", 
                   style={'color': '#95a5a6', 'textAlign': 'center'})
        ])
        
        default_viz = html.Div([
            html.Div(style={
                'height': '400px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'border': '2px dashed #dee2e6'
            }, children=[
                html.H4("Visualization will appear here", style={'color': '#adb5bd'})
            ])
        ])
        
        return default_info, default_viz

    # When a node is clicked
    node_id = node_data['id']
    
    # Error handling for missing node data
    if node_id not in node_to_data_indices:
        error_info = html.Div([
            html.H4("❌ Error", style={'color': '#e74c3c'}),
            html.P(f"Node {node_id} data not found.", style={'color': '#c0392b'})
        ])
        return error_info, html.Div()
    
    # Get the data for the clicked node
    indices = node_to_data_indices[node_id]
    node_df = df.iloc[indices]
    split_info = node_to_split_info.get(node_id, {})
    
    # Create comprehensive node information
    info_components = [
        html.H4(f"📊 {node_id.replace('-', ' ').title()}", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}, children=[
            html.Div([
                html.Strong("Samples: "), 
                html.Span(f"{len(node_df)}", style={'color': '#3498db', 'fontSize': '18px', 'fontWeight': 'bold'})
            ]),
            html.Div([
                html.Strong("Tree Level: "), 
                html.Span(f"{split_info.get('level', 'Unknown')}", style={'color': '#9b59b6'})
            ])
        ])
    ]
    
    # Add split information if available
    if 'split_feature' in split_info:
        info_components.append(html.Div(style={'marginTop': '10px'}, children=[
            html.Strong("Split Condition: "),
            html.Span(f"{split_info['split_feature']} ≤ {split_info['split_threshold']:.3f}", 
                     style={'backgroundColor': '#e8f4fd', 'padding': '4px 8px', 'borderRadius': '4px'})
        ]))
    
    # Add parent split info
    if split_info.get('parent_info') and split_info['parent_info'] != "Root":
        info_components.append(html.Div(style={'marginTop': '8px'}, children=[
            html.Strong("Node Condition: "),
            html.Span(split_info['parent_info'], 
                     style={'backgroundColor': '#fef9e7', 'padding': '4px 8px', 'borderRadius': '4px'})
        ]))
    
    node_info = html.Div(info_components)
    
    # Create visualization based on selected type
    if viz_type == 'histogram':
        fig = go.Figure(
            data=[go.Histogram(
                x=node_df[selected_feature], 
                name=selected_feature,
                marker_color='#3498db',
                opacity=0.8
            )],
            layout=go.Layout(
                title={
                    'text': f'Distribution of {selected_feature.replace("_", " ")}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title=selected_feature.replace('_', ' '),
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
        )
        visualization = dcc.Graph(figure=fig)
        
    elif viz_type == 'box':
        fig = go.Figure(
            data=[go.Box(
                y=node_df[selected_feature],
                name=selected_feature,
                marker_color='#e74c3c'
            )],
            layout=go.Layout(
                title={
                    'text': f'Box Plot of {selected_feature.replace("_", " ")}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                yaxis_title=selected_feature.replace('_', ' '),
                template='plotly_white',
                height=400
            )
        )
        visualization = dcc.Graph(figure=fig)
        
    elif viz_type == 'stats':
        stats_data = []
        for feature in feature_names:
            feature_data = node_df[feature]
            stats_data.append({
                'Feature': feature.replace('_', ' '),
                'Mean': f"{feature_data.mean():.3f}",
                'Std': f"{feature_data.std():.3f}",
                'Min': f"{feature_data.min():.3f}",
                'Max': f"{feature_data.max():.3f}",
                'Median': f"{feature_data.median():.3f}"
            })
        
        visualization = dash_table.DataTable(
            data=stats_data,
            columns=[{'name': col, 'id': col} for col in stats_data[0].keys()],
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Arial'
            },
            style_header={
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ]
        )
        
    elif viz_type == 'comparison':
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f.replace('_', ' ') for f in feature_names[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
        for i, feature in enumerate(feature_names[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=node_df[feature],
                    name=feature.replace('_', ' '),
                    marker_color=colors[i],
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Feature Comparison",
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        visualization = dcc.Graph(figure=fig)
    
    return node_info, visualization

# --- 6. Helper Function to Find Available Port ---
def find_available_port(preferred_ports=None):
    """
    Find an available port, trying common development ports first.
     
    Args:
        preferred_ports (list): List of preferred ports to try first
     
    Returns:
        int: An available port number, or None if no port is available
    """
    # Common development ports that are usually accessible through firewalls
    if preferred_ports is None:
        preferred_ports = [
            # Dash default
            8050,
            # Very common development ports (least likely to be blocked)
            3000, 3001, 3002, 3003,
            # Flask and general dev server ports
            5000, 5001, 5002,
            # Alternative common ports
            8000, 8001, 8002, 8080,
            # Extended ranges
            4000, 4001, 4002,
            8090, 8091, 8092
        ]
    
    # Additional backup ranges if preferred ports fail
    backup_ranges = [
        range(3000, 3020),  # React/frontend dev servers
        range(5000, 5020),  # Flask/Python dev servers  
        range(8000, 8020),  # General dev servers
        range(4000, 4020),  # Often unrestricted
    ]
    
    def try_port(port):
        """Helper function to test if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    # First, try preferred ports
    for port in preferred_ports:
        if try_port(port):
            return port
    
    # If none of the preferred ports work, try backup ranges
    for port_range in backup_ranges:
        for port in port_range:
            if port not in preferred_ports and try_port(port):
                return port
    
    return None

# --- 7. Run the App ---
if __name__ == '__main__':
    # Find an available port
    available_port = find_available_port()
     
    if available_port is None:
        print("❌ No available ports found in common development port ranges. Please free up some ports and try again.")
        exit(1)
     
    if available_port != 8050:
        print(f"🔄 Port 8050 is in use. Using port {available_port} instead.")
     
    print(f"🚀 Starting Dash app on http://127.0.0.1:{available_port}")
    app.run(debug=True, port=available_port)