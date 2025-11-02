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

def order_3_unweighted():
    print("Counting motifs of order 3 on dataset hospital")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_3(edges)
    print(f"Classi isomorfismo: {len(output['motifs'])}")
    # print(output['motifs'])
    # app.plot_utils.plot_dist_motifs(output['motifs'], "hospital_motifs_3", 6)
    app.plot_utils.plot_leading_motifs(output['motifs'], "hospital_leading_motifs_3", 6, limit=10)
    print("")

def order_4_unweighted():
    print("Counting motifs of order 4 on dataset hospital")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_4(edges)
    print(f"Classi isomorfismo: {len(output['motifs'])}")
    # print(output['motifs'])
    # app.plot_utils.plot_dist_motifs(output['motifs'], "hospital_motifs_4", 5)
    app.plot_utils.plot_leading_motifs(output['motifs'], "hospital_leading_motifs_4", 5, limit=10)
    print("")

def order_3_weighted():
    print("Counting motifs of order 3 on dataset hospital")
    edges, tot = app.loaders.load_hospital_duplicates(4)
    app.plot_utils.plot_dist_hyperedges_weights(tot, "hospital_weight_dist")
    app.utils.normalize_weights(edges)

    motifs = app.motifs2.motifs_order_3(edges, weighted=True)
    # app.plot_utils.plot_dist_motifs(output['motifs'], "hospital_motifs_3", 6)
    print(motifs)
    app.plot_utils.plot_leading_motifs(motifs, "hospital_leading_weighted_motifs_3", 6, limit=10)
    print("")

def order_4_weighted():
    print("Counting motifs of order 4 on dataset hospital")
    edges, tot = app.loaders.load_hospital_duplicates(4)
    app.plot_utils.plot_dist_hyperedges_weights(tot, "hospital_weight_dist")
    app.utils.normalize_weights(edges)

    motifs = app.motifs2.motifs_order_4(edges, weighted=True)
    # app.plot_utils.plot_dist_motifs(output['motifs'], "hospital_motifs_3", 6)
    app.plot_utils.plot_leading_motifs(motifs, "hospital_leading_weighted_motifs_4", 5, limit=10)
    print("")


# order_3_unweighted()
# order_4_unweighted()
order_3_weighted()
order_4_weighted()
