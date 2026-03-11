from math import isclose
from pprint import pprint

from src.graph import StandardConstructionMethod
from src.motifs.motifs2 import count_motifs as count_motifs2
from src.motifs.motifs3 import count_motifs as count_motifs3
from src.motifs.motifs_count_base import generate_motifs
from tests.util import Colors, Loader, time_function_p

hg = Loader("high_school").construction_method(StandardConstructionMethod(weighted=True)).load()
# hg = hg.filter_orders([2, 3], retain=True)


print(f"{Colors.DIM}Running motifs2... {Colors.RESET}")
motifs2, _ = time_function_p(lambda: count_motifs2(hg, 4))

print(f"{Colors.DIM}Running motifs3... {Colors.RESET}")
motifs3, _ = time_function_p(lambda: count_motifs3(hg, 4))


rep, rep_map = generate_motifs(4)
count = 0
error_count = 0
for motif in rep:
    tri_edge_count = sum([1 if len(edge) == 3 else 0 for edge in motif])
    quad_edge_count = sum([1 if len(edge) == 4 else 0 for edge in motif])
    if motifs2[motif].count != motifs3[motif].count or not isclose(motifs2[motif].intensity, motifs3[motif].intensity):
        error_count += 1
        print(f"{Colors.RED}Error for motif {motif}{Colors.RESET}")
        print(motifs2[motif])
        print(motifs3[motif])
        print("")

if error_count == 0:
    print(f"{Colors.GREEN}All motifs match!{Colors.RESET}")
