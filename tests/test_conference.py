import src as app


def t1():
    print("load_conference")
    app.loaders.load_conference(4)
    print("")


def t2():
    print("load_conference_duplicates")
    app.loaders.load_conference_duplicates(4)
    print("")


t1()
t2()
