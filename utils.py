import networkx as nx
import matplotlib.pyplot as plt

def ensure_vertex_cover(graph, bitstring):
    """
    Convert a bitstring/list to a valid vertex cover.
    
    Args:
        graph: NetworkX graph
        bitstring: String like "010010", or list/tuple of bits like [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]

    Returns:
        Same type as input - a bitstring (string) or list representing a valid vertex cover
    """
    input_is_string = isinstance(bitstring, str)
    
    if input_is_string:
        V = list(map(int, bitstring))
    else:
        V = list(bitstring)
    
    def is_vertex_cover(graph, V):
        for edge in graph.edges:
            if not (V[edge[0]] == 1 or V[edge[1]] == 1):
                return False
        return True

    while not is_vertex_cover(graph, V):
        for edge in graph.edges:
            if not (V[edge[0]] == 1 or V[edge[1]] == 1):
                V[edge[0]] = 1
                break

    if input_is_string:
        return ''.join(map(str, V))
    return V

def plot_result(graph, bitstring, filename, title):
    """
    Plot a graph with vertices colored based on vertex cover.
    
    Args:
        graph: NetworkX graph
        bitstring: List or string of bits representing the solution
        filename: Name of the file to save the plot (without extension)
        title: Title for the plot
    """
    # Create color map based on bitstring
    color_map = ['red' if bit == 1 else 'lightblue' for bit in bitstring]

    # Draw the graph with colored nodes
    plt.figure(figsize=(10, 8))
    nx.draw(graph, 
            node_color=color_map,
            node_size=500, 
            with_labels=True,
            font_weight='bold',
            font_color='black')
    plt.title(title)
    plt.savefig(f'{filename}.pdf',            # Higher DPI for better quality
                bbox_inches='tight',  # Removes extra whitespace
                format='pdf',         # Specify format explicitly
                transparent=False,    # White background
                pad_inches=0.1)      # Small padding around the figure