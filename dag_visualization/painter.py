import networkx as nx
from typing import Dict, Tuple, Optional, Hashable, Literal, Union

import numpy as np

import matplotlib.pyplot as plt

from .config import *


class Painter:
    @staticmethod
    def draw_dag(
        dag: nx.DiGraph,
        layout: Dict[Hashable, Tuple[float, float]],
        axis_type: Union[Literal["on"], Literal["off"]] = "on",
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[str] = None,
    ) -> None:
        coordinates = np.vstack(list(layout.values()))
        xlims = (
            np.floor(np.min(coordinates[:, 0])) - 1,
            np.ceil(np.max(coordinates[:, 0]) + 1),
        )
        ylims = (
            np.floor(np.min(coordinates[:, 1])) - 1,
            np.ceil(np.max(coordinates[:, 1]) + 1),
        )

        _, ax = plt.subplots(figsize=figsize)
        plt.axis(axis_type)
        nx.draw_networkx(
            dag,
            pos=layout,
            node_color=["blue" if node[0] == "#" else "white" for node in dag],
            ax=ax,
            node_size=NODE_SIZE,
            font_size=FONT_SIZE,
            with_labels=WITH_LABELS,
        )

        xgrid = np.arange(xlims[0], xlims[1] + 1, dtype=int)
        ygrid = np.arange(ylims[0], ylims[1] + 1, dtype=int)

        ax.set_xticks(xgrid)
        ax.set_xticklabels(xgrid)
        ax.set_yticks(ygrid)
        ax.set_yticklabels(ygrid)

        ax.collections[0].set_edgecolor("black")
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
