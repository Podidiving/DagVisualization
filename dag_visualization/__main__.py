from argparse import ArgumentParser, Namespace

import networkx as nx
from .coffman_graham_layout import CoffmanGrahamLayout
from loguru import logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", required=False, type=str, default="misc/binary_tree.xml"
    )
    parser.add_argument(
        "--output1", required=False, type=str, default="graph1.png"
    )
    parser.add_argument(
        "--output2", required=False, type=str, default="graph2.png"
    )
    parser.add_argument("--max-width", required=False, type=int, default=3)

    return parser.parse_args()


def main(args: Namespace) -> None:
    dag = nx.read_graphml(args.input)
    max_width = args.max_width
    logger.info(f"Visualizing DAG with max_width={max_width}")
    CoffmanGrahamLayout.make_visualization(
        dag,
        max_width=max_width,
        save_path=args.output1,
    )
    logger.info("Done.")

    logger.info(f"Visualizing DAG without max_width")
    CoffmanGrahamLayout.make_visualization(
        dag, max_width=None, save_path=args.output2
    )
    logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
