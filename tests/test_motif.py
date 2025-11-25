# from util import Loader, WeightType, time_function

# edges, motifs = Loader("hospital").order(3).weight_type(WeightType.UNWEIGHTED).load()
# time_function(lambda : print(app.aggregate.aggregate(edges, motifs)))

c = {"asdf": 7, "b": "asdfasdf ", "a": 1234}
for order in [3, 4]:
    for weighted in [True, False]:
        for dataset in ["hospital", "conference", "high_school", "primary_school"]:
            print(f"Dataset: {dataset}, Order: {order}, Weighted: {weighted}")
# edges, motifs = Loader("hospital").order(3).weight_type(WeightType.STANDARD).load()
# time_function(lambda : print(app.aggregate.aggregate(edges, motifs)))

# print(motifs)
# print(edges.pop())

# TestBuilder("hospital").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("conference").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("conference").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("high_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("high_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("primary_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("primary_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()
