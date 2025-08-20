import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional
from .tree import Tree


class TreeVisualizer:
    """
    TLDR: Converts Tree objects into interactive Plotly visualizations.
    Simple point-based nodes with hierarchical layout for Dash and Jupyter.
    """
    
    def __init__(self, tree: Tree, 
                 node_size: int = 8,
                 level_height: float = 0.2,
                 node_spacing: float = 1.0):
        """Initialize visualizer with a Tree object."""
        self.tree = tree
        self.node_size = node_size
        self.level_height = level_height
        self.node_spacing = node_spacing
        self._positions = None
        
    def compute_positions(self) -> Dict[int, Tuple[float, float]]:
        """Compute hierarchical layout positions for all nodes."""
        if self._positions is not None:
            return self._positions
            
        positions = {}
        nodes_by_level = self.tree.nodes_by_level()
        
        for level, nodes in enumerate(nodes_by_level):
            y = -level * self.level_height
            self._position_level_nodes(nodes, y, level, positions)
            
        self._positions = positions
        return positions
    
    def _position_level_nodes(self, nodes: List[int], y: float, level: int,
                            positions: Dict[int, Tuple[float, float]]) -> None:
        """Position nodes at a specific level horizontally with adaptive spacing."""
        n_nodes = len(nodes)
        if n_nodes == 1:
            positions[nodes[0]] = (0, y)
            return
        
        # Adaptive spacing: larger at root, smaller at deeper levels
        level_spacing = self.node_spacing * (2.0 ** (-level))
        total_width = (n_nodes - 1) * level_spacing
        start_x = -total_width / 2
        
        for i, node_id in enumerate(nodes):
            x = start_x + i * level_spacing
            positions[node_id] = (x, y)
    
    def get_node_colors(self) -> List[str]:
        """Generate colors for nodes based on their level."""
        colors = []
        for node_id in self.tree.bfs():
            level = self.tree.level(node_id)
            # Generate colors using a simple hue rotation
            hue = (level * 60) % 360
            colors.append(f'hsl({hue}, 70%, 50%)')
        return colors
    
    def get_node_info(self, node_id: int) -> Dict[str, any]:
        """Generate hover data for a node including level, point count, and indices."""
        node_data = self.tree[node_id]
        idx = node_data['idx']
        level = self.tree.level(node_id)
        
        # Hover info with level, points, and indices
        hover_data = {
            'node_id': node_id,
            'level': level,
            'n_points': len(idx),
            'idx': [int(i) for i in idx] if len(idx) > 0 else None
        }
        return hover_data
    
    def create_node_trace(self) -> go.Scatter:
        """Create the scatter plot trace for tree nodes."""
        positions = self.compute_positions()
        node_ids = list(self.tree.bfs())
        
        x_coords = [positions[node_id][0] for node_id in node_ids]
        y_coords = [positions[node_id][1] for node_id in node_ids]
        colors = self.get_node_colors()
        
        # Create hover template with level, points, and indices
        hover_template = (
            "level: %{customdata[0]}\t points: %{customdata[1]}\t idx: %{customdata[2]}"
            "<extra></extra>"
        )
        
        # Prepare custom data for hover
        custom_data = []
        for node_id in node_ids:
            info = self.get_node_info(node_id)
            # Format first few indices as string
            if info['idx'] is None:
                idx_str = 'None'
            else: 
                idx_str = str(info['idx'][:5])[1:-1]
                idx_str = idx_str + ' ...' if len(info['idx']) > 5 else idx_str
            
            custom_data.append([
                info['level'], 
                info['n_points'], 
                idx_str
            ])
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=self.node_size,
                color=colors,
                line=dict(width=1, color='black')
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            name='Nodes'
        )
    
    def create_edge_trace(self) -> go.Scatter:
        """Create the line trace for tree edges."""
        positions = self.compute_positions()
        x_coords = []
        y_coords = []
        
        for node_id in self.tree.bfs():
            for child_id in self.tree.get_children(node_id):
                self._add_edge_coordinates(positions, node_id, child_id, x_coords, y_coords)
                
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(width=1, color='black'),
            hoverinfo='none',
            showlegend=False,
            name='Edges'
        )
    
    def _add_edge_coordinates(self, positions: Dict[int, Tuple[float, float]], 
                            parent_id: int, child_id: int,
                            x_coords: List[float], y_coords: List[float]) -> None:
        """Add coordinates for a single edge to the coordinate lists."""
        parent_pos = positions[parent_id]
        child_pos = positions[child_id]
        
        # Add line from parent to child
        x_coords.extend([parent_pos[0], child_pos[0], None])
        y_coords.extend([parent_pos[1], child_pos[1], None])
    
    def create_figure(self, title: str = "Tree Visualization", 
                     width: int = 800, height: int = 600) -> go.Figure:
        """Create the complete interactive Plotly figure."""
        edge_trace = self.create_edge_trace()
        node_trace = self.create_node_trace()
        
        fig = go.Figure(data=[edge_trace, node_trace])
        self._configure_layout(fig, title, width, height)
        
        return fig
    
    def _configure_layout(self, fig: go.Figure, title: str, 
                         width: int, height: int) -> None:
        """Configure the figure layout and styling."""
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            ),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                # borderwidth=1,
                font_size=10,
                font_family="monospace",
                align="left"
            )
        )
    
    def reset_layout(self) -> None:
        """Reset cached positions to force layout recomputation."""
        self._positions = None


# Convenience function for quick visualization
def visualize_tree(tree: Tree, **kwargs) -> go.Figure:
    """Quick function to visualize a tree with default settings."""
    visualizer = TreeVisualizer(tree, **kwargs)
    return visualizer.create_figure()