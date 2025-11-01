import _context
import src as app

def t1():
    print("testing on dataset high_school")
    edges = app.loaders.load_hospital(4)
    output = {}
    output['motifs'] = app.motifs2.motifs_order_3(edges, -1)
    print("")

def t2():
    print("testing on dataset high_school")
    app.loaders.load_hospital(4)
    print("")


t1()
# t2()
