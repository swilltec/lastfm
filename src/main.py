import random
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

FILE_PATH = Path(__file__).resolve().parents[1] / "edges.csv"


class MenuOption(Enum):
    GRAPH_INFO = 0
    VISUALIZE_GRAPH = 1
    DETECT_CYCLES = 2
    CHECK_CONNECTED = 3
    CHECK_PATH = 4
    RUN_BFS_DFS = 5
    DISPLAY_ADJ_MATRIX = 6
    EXIT = 7


def get_graph_info(G: nx.Graph):
    """Calculate and print various metrics for the graph."""

    # Basic metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_connected = nx.is_connected(G)
    connected_components = nx.number_connected_components(G)

    # Degree metrics
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0

    # Cycle counts
    triangle_count = (
        sum(nx.triangles(G).values()) // 3
    )  # Each triangle is counted three times
    four_cycle_count = sum(1 for cycle in nx.cycle_basis(G) if len(cycle) == 4)
    cycles = nx.cycle_basis(G)

    # Longest cycle
    longest_cycle = max(cycles, key=len) if cycles else None
    longest_cycle_length = len(longest_cycle) if longest_cycle else 0

    # Average shortest path length
    avg_shortest_path_length = (
        nx.average_shortest_path_length(G) if is_connected else -1
    )

    # Diameter (max shortest path length)
    diameter = nx.diameter(G) if is_connected else -1

    # BFS Tree Depth: maximum distance from a node to all others
    bfs_tree_depth = (
        max(
            len(path)
            for node in G
            for path in nx.all_shortest_paths(G, source=node, target=list(G.nodes())[0])
        )
        - 1
        if num_nodes > 1
        else 0
    )

    # DFS Tree Depth: maximum depth in the DFS tree
    dfs_tree = nx.dfs_tree(
        G, source=list(G.nodes())[0]
    )  # Get the DFS tree starting from an arbitrary node
    dfs_tree_depth = max(dict(dfs_tree.degree()).values()) - 1 if num_nodes > 0 else 0

    # Print the metrics
    print("\nGraph Metrics:")
    print(f"Total Nodes: {num_nodes}")
    print(f"Total Edges: {num_edges}")
    print(f"Connected Components: {connected_components}")
    print(f"Average Degree: {avg_degree:.2f}")
    print(f"Triangle Count: {triangle_count}")
    print(f"4-Cycle Count: {four_cycle_count}")
    print(f"Longest Cycle Length: {longest_cycle_length}")
    print(
        f"Average Shortest Path Length: {avg_shortest_path_length:.2f}"
        if is_connected
        else "Average Shortest Path Length: N/A (disconnected)"
    )
    print(
        f"Max Shortest Path Length (Diameter): {diameter}"
        if is_connected
        else "Max Shortest Path Length (Diameter): N/A (disconnected)"
    )
    print(f"BFS Tree Depth (from sample node): {bfs_tree_depth}")
    print(f"DFS Tree Depth (from same node): {dfs_tree_depth}")


def load_graph_with_pandas(filepath: str) -> nx.Graph:
    """Load an undirected graph from an edge list CSV using pandas."""
    df = pd.read_csv(filepath)
    if df.shape[1] != 2:
        raise ValueError("Edge list must have two columns.")
    source_col, target_col = df.columns[:2]
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
    return G


def visualize_graph(G: nx.Graph):
    """Visualize a sample of the graph using Matplotlib."""
    # Get the graph size
    graph_size = len(G.nodes)

    # Ask the user for the sample size or use the default (graph size, with a minimum of 1000)
    sample_size_input = input(
        f"Enter the sample size (default is {max(1000, graph_size)}): "
    )

    # Set the sample size based on user input or default to graph size or 1000
    try:
        sample_size = (
            int(sample_size_input) if sample_size_input else max(1000, graph_size)
        )
    except ValueError:
        print("Invalid input. Using the default sample size.")
        sample_size = max(1000, graph_size)

    # If the graph has fewer nodes than the requested sample size, adjust the sample size
    sample_size = min(sample_size, graph_size)

    # Select a random sample of nodes
    sub_nodes = random.sample(list(G.nodes), sample_size)
    subgraph = G.subgraph(sub_nodes)

    # Plot the subgraph
    plt.figure(figsize=(10, 8))
    nx.draw(subgraph, with_labels=False, node_size=50, font_size=8)
    plt.title(f"Graph Visualization (Sample of {sample_size} nodes)")
    plt.show()


def detect_cycles(G: nx.Graph):
    """Detect and categorize cycles in the graph, using triangles() for triangles and cycle_basis() for others."""

    # Detect triangles using the triangles() function
    triangle_count = sum(1 for v in G.nodes() if nx.triangles(G, v) > 0)
    print(f"Triangles (3-cycles): {triangle_count}")

    # Detect all cycles using cycle_basis (for lengths 4 and greater)
    cycles = nx.cycle_basis(G)
    total_cycles = len(cycles)
    print(f"Total cycles found: {total_cycles}")

    # Count quadrilaterals (4-cycles) and longer cycles (> 4 nodes)
    quadrilateral_count = sum(1 for cycle in cycles if len(cycle) == 4)
    long_cycle_count = sum(1 for cycle in cycles if len(cycle) > 4)

    print(f"Quadrilaterals (4-cycles): {quadrilateral_count}")
    print(f"Longer cycles (>4 nodes): {long_cycle_count}")


def is_connected(G: nx.Graph):
    """Check if the graph is connected."""
    return nx.is_connected(G)


def path_exists(G: nx.Graph, node1: str, node2: str) -> bool:
    """Check if a path exists between two nodes."""
    try:
        has_path = nx.has_path(G, int(node1), int(node2))
    except (ValueError, nx.NodeNotFound) as ex:
        print(ex)
        has_path = False
    return has_path


def run_bfs_dfs(G: nx.Graph, start_node: str):
    """Run BFS and DFS from a starting node and save each tree visualization as a separate image."""

    bfs_tree = nx.bfs_tree(G, start_node)
    dfs_tree = nx.dfs_tree(G, start_node)

    # Draw and save BFS tree
    plt.figure(figsize=(6, 6))
    nx.draw(bfs_tree, with_labels=False, node_size=50, font_size=8)
    plt.title("BFS Tree")
    plt.savefig("bfs_tree.png")
    plt.close()

    # Draw and save DFS tree
    plt.figure(figsize=(6, 6))
    nx.draw(dfs_tree, with_labels=False, node_size=50, font_size=8)
    plt.title("DFS Tree")
    plt.savefig("dfs_tree.png")
    plt.close()


def check_path(G: nx.Graph):
    """Prompt user for two nodes and check if a path exists."""
    node1 = input("Enter first node: ")
    node2 = input("Enter second node: ")
    print(f"Path exists between {node1} and {node2}: {path_exists(G, node1, node2)}")


def run_bfs_dfs_prompt(G: nx.Graph):
    """Prompt user for starting node and run BFS/DFS."""
    start_node = input("Enter starting node for BFS/DFS: ")
    try:
        run_bfs_dfs(G, int(start_node))
    except ValueError as ex:
        print(ex)


def exit_program():
    """Exit the program."""
    print("Exiting...")
    exit()


def draw_adjacency_matrix(G: nx.Graph):
    """
    Draw the adjacency matrix of the graph as an image.

    Parameters:
    - G: networkx.Graph — the input graph.
    - sample_size: int or None — number of nodes to sample; if None, use all nodes.
    """
    nodes = list(G.nodes)
    total_nodes = len(nodes)

    try:
        sample_size = int(input(f"Enter sample size (max: {total_nodes}): "))
        sample_size = min(total_nodes, sample_size)
        sampled_nodes = nodes[:sample_size]
        subgraph = G.subgraph(sampled_nodes)
    except ValueError:
        subgraph = G

    adjacency_matrix = nx.to_numpy_array(subgraph)
    labels = list(subgraph.nodes)

    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Adjacency Matrix of Graph")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("adjacency_matrix.png")
    plt.close()


def main():

    try:
        # Load the graph
        G = load_graph_with_pandas(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file {FILE_PATH} was not found.")
        exit()

    # Mapping menu options to actions
    menu_actions = {
        MenuOption.GRAPH_INFO: lambda: get_graph_info(G),
        MenuOption.VISUALIZE_GRAPH: lambda: visualize_graph(G),
        MenuOption.DETECT_CYCLES: lambda: detect_cycles(G),
        MenuOption.CHECK_CONNECTED: lambda: print(
            f"Graph is connected: {is_connected(G)}"
        ),
        MenuOption.CHECK_PATH: lambda: check_path(G),
        MenuOption.RUN_BFS_DFS: lambda: run_bfs_dfs_prompt(G),
        MenuOption.DISPLAY_ADJ_MATRIX: lambda: draw_adjacency_matrix(G),
        MenuOption.EXIT: lambda: exit_program(),
    }

    # Menu loop
    while True:
        print("\nMenu:")
        for option in MenuOption:
            print(f"{option.value}. {option.name.replace('_', ' ').title()}")

        choice = input("Enter your choice: ")
        try:
            selected_option = MenuOption(int(choice))
            menu_actions[selected_option]()
        except (ValueError, KeyError):
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
