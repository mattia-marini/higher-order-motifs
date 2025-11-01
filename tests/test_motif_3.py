import _context
import src as app
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import config as cfg
import hypergraphx as hx
from hypergraphx.viz import draw_hypergraph
import os
import numpy as np

def order_3():
    print("Counting motifs of order 3 on dataset high_school")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_3(edges, -1)
    print(f"Classi isomorfismo: {len(output['motifs'])}")
    print(output['motifs'])
    app.utils.plot_dist_motifs(output['motifs'], "hospital_motifs_3", 6)
    print("")

def order_4():
    print("Counting motifs of order 4 on dataset high_school")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_4(edges, -1)
    print(f"Classi isomorfismo: {len(output['motifs'])}")
    print(output['motifs'])
    app.utils.plot_dist_motifs(output['motifs'], "hospital_motifs_4", 5)
    print("")



order_3()
order_4()
# Create a hypergraph
# Nodes: {1, 2, 3, 4, 5}
# Hyperedges: {1, 2, 3}, {3, 4}, and {4, 5}
# hxviz.draw_hypergraph()

# hypergraph = hx.Hypergraph()
# hypergraph.add_nodes([1, 2, 3])
# hypergraph.add_edges([(1, 2, 3)])
# hypergraph.add_edges([(2, 3)])
# hypergraph.add_edges([(1, 2)])
#
# draw_hypergraph(hypergraph)
# plt.show()



# Plot the hypergraph
# hx.visualization.plot(hypergraph)
# def plot_graph(graph, pos):
#     fig, ax = plt.subplots()
#     nx.draw(graph, pos, with_labels=True, ax=ax)
#     ax.axis("off")
#     fig.canvas.draw()
#     plt.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, "test"))
#
# G = nx.Graph()
# G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])
# plot_graph(G, nx.spring_layout(G))
# order_3()
# order_4()
