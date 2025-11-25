import src as app


def t1():
    print("load_primary_school")
    app.loaders.load_primary_school(4)
    print("")


def t2():
    print("load_primary_school_duplicates")
    app.loaders.load_primary_school_duplicates(4)
    print("")


t1()
t2()
