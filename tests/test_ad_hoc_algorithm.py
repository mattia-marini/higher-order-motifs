from pprint import pprint

from src.graph import StandardConstructionMethod
from src.motifs.motifs2 import count_motifs as count_motifs2
from src.motifs.motifs3 import count_motifs as count_motifs3
from src.motifs.motifs_count_base import generate_motifs
from tests.util import Loader

hg = Loader("high_school").construction_method(StandardConstructionMethod(weighted=True)).load()
hg = hg.filter_orders([2], retain=True)

# for i, x in enumerate(["a", "b", "c"]):
#     print(i, x)
rep_list, rep_map = generate_motifs(3)
# pprint(rep_map)
# pprint(len(motifs))
# pprint(labeling)

pprint(count_motifs2(hg, 3))
print("-" * 100)
pprint(count_motifs3(hg, 3))
# print(f"count_motifs3 {count_motifs3(hg, 3)}")
