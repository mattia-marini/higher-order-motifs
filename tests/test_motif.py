import _context
from util import TestBuilder


TestBuilder("hospital").order([3,4]).weighted(False).with_plots(False).run()
# TestBuilder("hospital").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("conference").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("conference").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("high_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("high_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()

# TestBuilder("primary_school").order([3,4]).weighted(False).normalize_weights(True).with_plots("default").run()
# TestBuilder("primary_school").order([3,4]).weighted(True).normalize_weights(True).with_plots("default").run()
