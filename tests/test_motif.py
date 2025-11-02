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
    # app.utils.plot_dist_motifs(output['motifs'], "hospital_motifs_3", 6)
    app.plot_utils.plot_leading_motifs(output['motifs'], "hospital_leading_motifs_3", 6, limit=10)
    print("")

def order_4():
    print("Counting motifs of order 4 on dataset high_school")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_4(edges, -1)
    print(f"Classi isomorfismo: {len(output['motifs'])}")
    print(output['motifs'])
    # app.utils.plot_dist_motifs(output['motifs'], "hospital_motifs_4", 5)
    app.plot_utils.plot_leading_motifs(output['motifs'], "hospital_leading_motifs_4", 5, limit=10)
    print("")



order_3()
order_4()
