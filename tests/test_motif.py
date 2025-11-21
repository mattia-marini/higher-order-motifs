import _context
import src as app
from util import Loader, WeightType


edges, motifs = Loader("hospital").order(3).weight_type(WeightType.UNWEIGHTED).load()
# print(motifs)
# print(edges.pop())
print(app.aggregate.aggregate(edges, motifs))

# TestBuilder("hospital").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("conference").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("conference").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("high_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("high_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("primary_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("primary_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()
