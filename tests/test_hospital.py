import _context
import src as app

def t1():
    print("load_hospital")
    app.loaders.load_hospital(4)
    print("")

def t2():
    print("load_hospital_duplicates")
    app.loaders.load_hospital_duplicates(4)
    print("")


t1()
t2()
