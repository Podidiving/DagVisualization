import networkx as nx
from typing import Dict, List, Hashable, Optional, Tuple, NamedTuple
import numpy as np
from abc import ABC
from scipy.optimize import linprog


class DistributionResult(NamedTuple):
    node_to_layer: Dict[Hashable, int]
    layers: List[List[Hashable]]


class Algorithm(ABC):
    def distribute(
        self,
        dag: nx.DiGraph,
    ) -> DistributionResult:
        raise NotImplemented

    @staticmethod
    def _get_topsorted_node_list(
        node_to_topsort_label: Dict[Hashable, int]
    ) -> List[Hashable]:
        return [
            node
            for node, _ in sorted(
                node_to_topsort_label.items(), key=lambda p: p[1]
            )
        ]

    @staticmethod
    def _sort_nodes_topologically(dag: nx.DiGraph) -> List[Hashable]:
        node_to_topsort_label = dict()
        nodes_wo_label = set(dag.nodes())

        for next_label in range(dag.number_of_nodes()):
            min_parent_labels_set = [float("inf")]
            argmin_node = None

            for node in nodes_wo_label:
                parents_labels_set = sorted(
                    [
                        node_to_topsort_label.get(parent, float("inf"))
                        for parent in dag.predecessors(node)
                    ],
                    reverse=True,
                )
                if parents_labels_set < min_parent_labels_set:
                    min_parent_labels_set = parents_labels_set
                    argmin_node = node

            assert argmin_node is not None
            node_to_topsort_label[argmin_node] = next_label
            nodes_wo_label.remove(argmin_node)

        return node_to_topsort_label


class FixedWidthAlgorithm(Algorithm):
    def __init__(self, width: int):
        self.max_width = width

    def distribute(
        self,
        dag: nx.DiGraph,
    ) -> DistributionResult:
        node_to_topsort_label = self._sort_nodes_topologically(dag)
        topsorted_nodes = self._get_topsorted_node_list(node_to_topsort_label)
        node_to_layer, layers = self._distribute_nodes(
            dag=dag, topsorted_nodes=topsorted_nodes, max_width=self.max_width
        )
        return DistributionResult(node_to_layer=node_to_layer, layers=layers)

    @staticmethod
    def _distribute_nodes(
        dag: nx.DiGraph, topsorted_nodes: List[Hashable], max_width: int
    ) -> Tuple[Dict[Hashable, int], List[List[Hashable]]]:

        layers = []
        node_to_layer = dict()

        for node in reversed(topsorted_nodes):
            placement_layer = 0
            for child in dag.successors(node):
                placement_layer = max(
                    placement_layer, node_to_layer[child] + 1
                )

            while (
                placement_layer < len(layers)
                and len(layers[placement_layer]) == max_width
            ):
                placement_layer += 1

            if placement_layer >= len(layers):
                layers.append([])
            layers[placement_layer].append(node)
            node_to_layer[node] = placement_layer

        return node_to_layer, layers


class MinimizingDummiesAlgorithm(Algorithm):
    def distribute(
        self,
        dag: nx.DiGraph,
    ) -> DistributionResult:
        node_to_topsort_label = self._sort_nodes_topologically(dag)
        (
            node_to_layer,
            layers,
        ) = self._distribute_nodes(dag, node_to_topsort_label)

        return DistributionResult(node_to_layer=node_to_layer, layers=layers)

    def _distribute_nodes(
        self, dag: nx.DiGraph, node_to_topsort_label: List[Hashable]
    ) -> Tuple[Dict[Hashable, int], List[List[Hashable]]]:
        n_nodes = dag.number_of_nodes()
        boundaries = [(0, None) for i in range(n_nodes)]
        A = []
        b = []
        c = np.zeros(n_nodes)

        for source, sink in dag.edges():
            source_label = node_to_topsort_label[source]
            sink_label = node_to_topsort_label[sink]
            A.append(np.zeros(n_nodes))
            A[-1][source_label] = -1
            A[-1][sink_label] = 1
            b.append(-1)
            c[source_label] += 1
            c[sink_label] -= 1

        linprog_layer_assignment = linprog(
            c, A_ub=A, b_ub=b, bounds=boundaries
        ).x
        node_to_linprog_layer = {
            node: linprog_layer_assignment[topsort_label]
            for node, topsort_label in node_to_topsort_label.items()
        }

        topsorted_nodes = self._get_topsorted_node_list(node_to_topsort_label)
        node_to_layer = dict()
        for node in reversed(topsorted_nodes):
            linprog_layer = node_to_linprog_layer[node]
            placement_layer = -1
            for child in dag.successors(node):
                placement_layer = max(
                    placement_layer, node_to_layer[child] + 1
                )
            if placement_layer == -1:
                placement_layer = int(np.floor(linprog_layer))
            node_to_layer[node] = placement_layer

        layers = [[] for _ in range(max(node_to_layer.values()) + 1)]
        for node, layer in node_to_layer.items():
            layers[layer].append(node)
        return node_to_layer, layers


def distribution_builder(max_width: Optional[int] = None) -> Algorithm:
    if max_width is None:
        return MinimizingDummiesAlgorithm()
    else:
        return FixedWidthAlgorithm(max_width)


__all__ = ["distribution_builder"]
