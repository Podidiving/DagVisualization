import networkx as nx
import itertools
import numpy as np

from typing import Dict, List, Hashable, Tuple, NamedTuple


class RefinmenterResult(NamedTuple):
    dag: nx.DiGraph
    refined_layout: Dict[Hashable, Tuple[float, float]]


class Refinmenter:
    def refine(
        self,
        dag: nx.DiGraph,
        layers: List[List[Hashable]],
        node_to_layer: Dict[Hashable, int],
    ) -> RefinmenterResult:
        dag, layers_w_dummies = self._add_dummy_nodes(
            dag=dag, layers=layers, node_to_layer=node_to_layer
        )
        layout = self._get_layout(layers=layers_w_dummies)
        refined_layout = self._refine_layout(
            dag=dag, layers=layers_w_dummies, layout=layout
        )
        return RefinmenterResult(dag=dag, refined_layout=refined_layout)

    @staticmethod
    def _add_dummy_nodes(
        dag: nx.DiGraph,
        layers: List[List[Hashable]],
        node_to_layer: Dict[Hashable, int],
    ) -> List[List[Hashable]]:

        dag_w_dummies = nx.DiGraph()
        dag_w_dummies.add_nodes_from(dag)
        layers_w_dummies = layers.copy()
        n_dummies = 0

        for edge in dag.edges():

            source, sink = edge
            source_layer = node_to_layer[source]
            sink_layer = node_to_layer[sink]

            assert source_layer >= sink_layer

            margin = source_layer - sink_layer
            if margin == 1:
                dag_w_dummies.add_edge(source, sink)
                continue

            dummy_nodes = []
            layer = source_layer - 1
            for _ in range(margin - 1):
                dummy_node = f"#{n_dummies}"
                dummy_nodes.append(dummy_node)
                layers_w_dummies[layer].append(dummy_node)
                n_dummies += 1
            dag_w_dummies.add_nodes_from(dummy_nodes)

            dummy_path = [source] + dummy_nodes + [sink]

            for dummy_source, dummy_sink in zip(dummy_path, dummy_path[1:]):
                dag_w_dummies.add_edge(dummy_source, dummy_sink)

        return dag_w_dummies, layers_w_dummies

    @staticmethod
    def _get_layout(
        layers: List[List[Hashable]],
    ) -> Dict[Hashable, Tuple[float, float]]:

        layout = dict()

        for layer, nodes_in_layer in enumerate(layers):
            for pos_in_layer, node in enumerate(nodes_in_layer):
                layout[node] = (pos_in_layer, layer)

        return layout

    def _count_crossings(
        self,
        dag: nx.DiGraph,
        layers: List[List[Hashable]],
        layout: Dict[Hashable, Tuple[float, float]],
    ) -> int:
        n_edge_crossings = 0
        for source_layer in layers[:-1]:
            for source_1, source_2 in itertools.combinations(source_layer, 2):
                if layout[source_1][0] > layout[source_2][0]:
                    source_1, source_2 = source_2, source_1
                sinks_1 = list(dag.successors(source_1))
                sinks_2 = list(dag.successors(source_2))
                for sink_1, sink_2 in itertools.product(sinks_1, sinks_2):
                    if sink_1 == sink_2:
                        continue
                    sink_1_x, sink_2_x = layout[sink_1][0], layout[sink_2][0]
                    if sink_1_x > sink_2_x:
                        n_edge_crossings += 1
        return n_edge_crossings

    def _refine_layout(
        self,
        dag: nx.DiGraph,
        layers: List[List[Hashable]],
        layout: Dict[Hashable, Tuple[float, float]],
    ) -> Dict[Hashable, Tuple[float, float]]:
        initial_n_edge_crossings = self._count_crossings(
            dag=dag, layers=layers, layout=layout
        )

        reverse_topsorted_nodes = [
            node
            for node, _ in sorted(layout.items(), key=lambda p: p[1][::-1])
        ]
        refined_layout = layout.copy()

        used_gridpoints = set()
        for node in reverse_topsorted_nodes:
            refined_offset = refined_layout[node][0]

            offsets_of_children = [
                refined_layout[child][0] for child in dag.successors(node)
            ]
            if offsets_of_children:
                refined_offset = np.median(offsets_of_children)

            new_coordinates = (refined_offset, layout[node][1])
            if new_coordinates not in used_gridpoints:
                refined_layout[node] = new_coordinates
            used_gridpoints.add(new_coordinates)

        final_n_edge_crossings = self._count_crossings(
            dag=dag, layers=layers, layout=refined_layout
        )

        if final_n_edge_crossings >= initial_n_edge_crossings:
            return layout

        return refined_layout
