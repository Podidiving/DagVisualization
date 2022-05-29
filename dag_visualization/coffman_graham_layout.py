from typing import Dict, Hashable, Optional, Tuple

import networkx as nx

from .painter import Painter
from .distribution import distribution_builder
from .refinmenter import Refinmenter


class CoffmanGrahamLayout:
    @staticmethod
    def make_visualization(
        dag: nx.DiGraph,
        max_width: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> Dict[Hashable, Tuple[float, float]]:
        _dag = nx.algorithms.dag.transitive_reduction(dag)

        algorithm = distribution_builder(max_width)
        dist_result = algorithm.distribute(_dag)
        node_to_layer, layers = dist_result

        refinmenter = Refinmenter()
        ref_result = refinmenter.refine(_dag, layers, node_to_layer)
        _dag, refined_layout = ref_result

        Painter.draw_dag(dag=_dag, layout=refined_layout, save_path=save_path)
