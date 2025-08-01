def plot_binary_tree(root, ax=None, x=0, y=0, dx=1.0, dy=-1.5, level=0):
    """
    TLDR: Plots a binary tree using matplotlib. 
    Recursively traverses the tree and draws nodes and edges.

    Parameters
    ----------
    root : TreeNode
        The root node of the binary tree to plot.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to plot on. If None, a new figure and axes are created.
    x : float, optional
        The x-coordinate for the current node.
    y : float, optional
        The y-coordinate for the current node.
    dx : float, optional
        The horizontal distance between nodes at the same level.
    dy : float, optional
        The vertical distance between levels.
    level : int, optional
        The current depth in the tree (used for recursive plotting).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted tree.

    Plotting Details
    ---------------
    - Each node is plotted as a circle with its data size as label.
    - Edges are drawn between parent and child nodes.
    - The function recursively positions child nodes left and right.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

    # Draw the current node
    ax.scatter(x, y, s=300, c='skyblue', edgecolors='k', zorder=3)
    label = str(len(root.data)) if hasattr(root, 'data') and hasattr(root.data, '__len__') else str(root.data)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=4)

    children = root.get_children()
    n = len(children)
    if n == 2:
        # Binary split: left and right
        offsets = [-dx/(2**level), dx/(2**level)]
        for i, child in enumerate(children):
            child_x = x + offsets[i]
            child_y = y + dy
            # Draw edge
            ax.plot([x, child_x], [y, child_y], c='k', zorder=2)
            plot_binary_tree(child, ax=ax, x=child_x, y=child_y, dx=dx, dy=dy, level=level+1)
    elif n == 1:
        # Only one child: draw straight down
        child_x = x
        child_y = y + dy
        ax.plot([x, child_x], [y, child_y], c='k', zorder=2)
        plot_binary_tree(children[0], ax=ax, x=child_x, y=child_y, dx=dx, dy=dy, level=level+1)
    # else: leaf node, nothing to do

    if ax is not None and level == 0:
        plt.show()
    return ax
