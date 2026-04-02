from python_core.graph import dump_digraph_to_file
from python_core.loaders import load_hospital

hg = load_hospital()
dump_digraph_to_file(hg, "/tmp/hospital_digraph.txt", map_vertices=True)
