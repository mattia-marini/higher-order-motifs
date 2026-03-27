import matplotlib.pyplot as plt
import networkx as nx

from tests.util import Colors, Loader, StandardConstructionMethod, time_function

# G = nx.karate_club_graph()  # example graph

hg = Loader("PACS").construction_method(StandardConstructionMethod(weighted=True)).load()

edges = [handle.nodes for handle in hg.get_order_map().get(2, [])]
G = nx.from_edgelist(edges, create_using=nx.Graph())

pos = nx.spring_layout(G, seed=42)  # force-directed layout
nx.draw(G, pos, with_labels=False, node_size=30, font_size=8, width=0.1)
plt.title("NetworkX + Matplotlib")
plt.show()
