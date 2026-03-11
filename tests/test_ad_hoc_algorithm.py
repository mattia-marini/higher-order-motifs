from pprint import pprint

from src.graph import StandardConstructionMethod
from src.motifs.motifs2 import count_motifs as count_motifs2
from src.motifs.motifs3 import count_motifs as count_motifs3
from src.motifs.motifs_count_base import generate_motifs
from tests.util import Loader, time_function_p

hg = Loader("hospital").construction_method(StandardConstructionMethod(weighted=True)).load()


hg = hg.filter_orders([2], retain=True)
# print(len(hg.get_order_map()[2]))


# motifs2 = count_motifs2(hg, 4)
motifs3 = count_motifs3(hg, 4)

rep, rep_map = generate_motifs(4)
# print(rep_map)
count = 0
for motif in rep:
    tri_edge_count = sum([1 if len(edge) == 3 else 0 for edge in motif])
    quad_edge_count = sum([1 if len(edge) == 4 else 0 for edge in motif])
    if quad_edge_count == 0 and tri_edge_count == 0:
        count += 1
        print(motif)
        # print(motifs2[motif])
        print(motifs3[motif])
        print("")
# print(count)

# motifs2 = count_motifs2(hg, 4)
# digraph_motifs = {motif: stat for motif, stat in motifs2.items() if all([len(edge) == 2 for edge in motif])}
#
# time_function_p(lambda: pprint(digraph_motifs))
# print("-" * 100)
# time_function_p(lambda: pprint(count_motifs3(hg, 4)))
# print(f"count_motifs3 {count_motifs3(hg, 3)}")
