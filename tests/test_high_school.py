import src as app


def t1():
    print("load_high_school")
    app.loaders.load_high_school(4)
    print("")


def t2():
    print("load_high_school_duplicates")
    app.loaders.load_high_school_duplicates(4)
    print("")


t1()
t2()
