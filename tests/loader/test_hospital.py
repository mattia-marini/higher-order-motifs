from src.graph import NormalizationMethod, StandardConstructionMethod
from tests.util import Loader

hg, motifs = (
    Loader("hospital")
    .order(3)
    .construction_method(
        StandardConstructionMethod(
            limit_edge_size=4,
            weighted=False,
            normalization_method=NormalizationMethod.NONE,
        )
    )
    .ignore_cache()
    .load()
)
print(hg)
print(motifs)

# hg, motifs = (
#     Loader("hospital")
#     .order(3)
#     .construction_method(
#         StandardConstructionMethod(
#             limit_edge_size=4,
#             weighted=True,
#             normalization_method=NormalizationMethod.DEFAULT,
#         )
#     )
#     .load()
# )
# print(hg)
# print(motifs)
#
# hg, motifs = (
#     Loader("hospital")
#     .order(4)
#     .construction_method(
#         StandardConstructionMethod(
#             limit_edge_size=4,
#             weighted=False,
#             normalization_method=NormalizationMethod.NONE,
#         )
#     )
#     .load()
# )
# print(hg)
# print(motifs)
#
# hg, motifs = (
#     Loader("hospital")
#     .order(4)
#     .construction_method(
#         StandardConstructionMethod(
#             limit_edge_size=4,
#             weighted=True,
#             normalization_method=NormalizationMethod.DEFAULT,
#         )
#     )
#     .load()
# )
# print(hg)
# print(motifs)
