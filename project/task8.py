from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
import networkx as nx
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata
from scipy.sparse import csr_matrix

def tensor_based_cfpq(
  rsm: RecursiveAutomaton,
  graph: nx.DiGraph,
  start_nodes: set[int] = None,
  final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_nfa = graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    rsm_nfa = rsm_to_nfa(rsm)

    graph_adj = AdjacencyMatrixFA(graph_nfa)
    rsm_adj = AdjacencyMatrixFA(rsm_nfa)

    for nonterminal in rsm.boxes:
        for matrix in graph_adj, rsm_adj:
            if nonterminal not in matrix.bool_decomposition:
                matrix.bool_decomposition[nonterminal] = csr_matrix(
                    (matrix.number_of_states, matrix.number_of_states), dtype=bool
                )

    prev_nonzero = 0
    current_nonzero = None

    while prev_nonzero != current_nonzero:
        intersection = intersect_automata(rsm_adj, graph_adj)
        closure = intersection.get_transitive_closure()

        for row_index, column_index in zip(*closure.nonzero()):
            row_rsm_state, row_graph_node = intersection.index_to_state[row_index].value
            column_rsm_state, column_graph_node = intersection.index_to_state[column_index].value

            (row_symbol, row_rsm_node) = row_rsm_state.value
            (column_symbol, column_rsm_node) = column_rsm_state.value

            dfa = rsm.boxes[row_symbol].dfa
            if (
                row_symbol == column_symbol
                and row_rsm_node in dfa.start_states
                and column_rsm_node in dfa.final_states
            ):

                graph_adj.bool_decomposition[row_symbol][
                    graph_adj.state_to_index[row_graph_node], graph_adj.state_to_index[column_graph_node]
                ] = True

        prev_nonzero = current_nonzero

        current_nonzero = sum(
            graph_adj.bool_decomposition[nonterminal].count_nonzero()
            for nonterminal in graph_adj.bool_decomposition
        )

    result = set()
    for n in graph_adj.start_states:
        for m in graph_adj.final_states:
            if graph_adj.bool_decomposition[rsm.initial_label][n, m]:
                result.add((graph_adj.index_to_state[n], graph_adj.index_to_state[m]))
    return result



def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())

def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)

def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for nonterminal, box in rsm.boxes.items():
        dfa = box.dfa

        for state in dfa.start_states:
            nfa.add_start_state(State((nonterminal, state)))
        for state in dfa.final_states:
            nfa.add_final_state(State((nonterminal, state)))

        transitions = dfa.to_networkx().edges(data="label")
        for start, final, label in transitions:
            nfa.add_transition(State((nonterminal, start)), label, State((nonterminal, final)))

    return nfa
