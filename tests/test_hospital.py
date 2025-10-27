import _context
import src as app

def t1():
    print("loaders.load_hospital")
    app.loaders.load_hospital(4)

def t2():
    print("load_hospital_duplicates")
    app.loaders.load_hospital_duplicates(4)


t1()
t2()
