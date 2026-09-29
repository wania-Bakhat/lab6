import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# import the graph, coordinates and search algorithms from searchAlgos.py
from searchAlgos import (
    hospital_graph,
    locations,
    gbfs,
    a_star,
)

# ------------------------------------------------------------------
# Page config + title
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Emergency Supply Robot - Informed Search",
    layout="wide",
)

st.title("Emergency Supply Robot: Informed Search Visualizer")
st.write(
    "A hospital robot has to carry an emergency medical supply from one location "
    "to another. Pick a start node, a goal node and a search algorithm "
    "(Greedy Best-First Search or A*), and the app will find a route and "
    "highlight it on the hospital graph."
)

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
nodes = list(hospital_graph.keys())

col1, col2, col3 = st.columns(3)

with col1:
    start = st.selectbox(
        "Select Initial Node",
        nodes,
        index=nodes.index("Pharmacy"),
    )

with col2:
    goal = st.selectbox(
        "Select Goal Node",
        nodes,
        index=nodes.index("Emergency_Ward"),
    )

with col3:
    algorithm = st.selectbox(
        "Select Search Algorithm",
        ["GBFS", "A*"],
    )

# ------------------------------------------------------------------
# Run search
# ------------------------------------------------------------------
if st.button("Run Search"):

    if algorithm == "GBFS":
        path, cost, expansion_order = gbfs(start, goal)
    else:
        path, cost, expansion_order = a_star(start, goal)

    if path is None:
        st.error(
            f"No path found from {start} to {goal}. "
            "Remember the corridors are one-way, so not every node can reach every other node."
        )

    else:
        # --------------------------------------------------------------
        # Build the NetworkX graph
        # --------------------------------------------------------------
        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():
            G.add_node(node)
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = locations

        path_edges = list(zip(path[:-1], path[1:]))
        other_edges = [e for e in G.edges() if e not in path_edges]

        node_colors = []
        for n in G.nodes():
            if n == start:
                node_colors.append("#2ecc71")      # start = green
            elif n == goal:
                node_colors.append("#f1c40f")      # goal = yellow
            elif n in path:
                node_colors.append("#ff8c69")      # on the path = orange
            else:
                node_colors.append("#aec6e8")      # everything else = light blue

        fig, ax = plt.subplots(figsize=(10, 6))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4800, ax=ax,
                               edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color="gray",
                               width=1.5, arrows=True, arrowsize=18, node_size=4800, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red",
                               width=4, arrows=True, arrowsize=22, node_size=4800, ax=ax)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                     label_pos=0.35, ax=ax)

        ax.set_title(f"{algorithm} Solution Path")
        ax.axis("off")

        st.pyplot(fig)

        # --------------------------------------------------------------
        # Results below the graph
        # --------------------------------------------------------------
        st.subheader("Search Result")

        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Solution Path:** {' → '.join(path)}")
        st.write(f"**Total Path Cost:** {cost:.2f}")
        st.write(f"**Node Expansion Order:** {' → '.join(expansion_order)}")
