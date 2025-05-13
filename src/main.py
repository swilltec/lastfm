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
    """Get and display basic information about the graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    is_connected = nx.is_connected(G)

    print("\nGraph Information:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Graph density: {density:.4f}")
    print(f"Average node degree: {avg_degree:.2f}")
    print(f"Is the graph connected? {'Yes' if is_connected else 'No'}")


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
    nx.draw(subgraph, with_labels=True, node_size=50, font_size=8)
    plt.title(f"Graph Visualization (Sample of {sample_size} nodes)")
    plt.show()


def detect_cycles(G: nx.Graph):
    """Detect all simple cycles in the graph."""
    cycles = list(nx.cycle_basis(G))
    print(f"Found {len(cycles)} cycles.")
    for i, cycle in enumerate(cycles, 1):
        print(f"Cycle {i}: {cycle}")

    return cycles


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
    """Run BFS and DFS from a starting node and visualize the trees."""
    bfs_tree = nx.bfs_tree(G, start_node)
    dfs_tree = nx.dfs_tree(G, start_node)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    nx.draw(bfs_tree, with_labels=True, node_size=50, font_size=8)
    plt.title("BFS Tree")

    plt.subplot(1, 2, 2)
    nx.draw(dfs_tree, with_labels=True, node_size=50, font_size=8)
    plt.title("DFS Tree")
    plt.savefig("search.png")


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
    """Draw the adjacency matrix of the graph as an image."""
    adjacency_matrix = nx.to_numpy_array(G)
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Adjacency Matrix of Graph")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.xticks(range(len(G.nodes)), G.nodes, rotation=90)
    plt.yticks(range(len(G.nodes)), G.nodes)
    plt.grid(False)
    plt.savefig("adjacency_matrix.png")


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
