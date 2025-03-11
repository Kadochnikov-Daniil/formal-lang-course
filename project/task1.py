import cfpq_data
import networkx


def get_graph_by_name(name: str):
    path = cfpq_data.download(name)
    return cfpq_data.graph_from_csv(path)


def get_graph_info(name: str):
    graph = get_graph_by_name(name)
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq_data.get_sorted_labels(graph),
    )


def create_two_cycle_graph(n, m, labels, path):
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    networkx.drawing.nx_pydot.write_dot(graph, path)
