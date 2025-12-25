from src.aggregate import aggregate
from src.graph import NormalizationMethod, StandardConstructionMethod
from tests.util import Colors, OldLoader, time_function_p


def time_aggregate(hg, motifs):
    print(f"{Colors.BLUE}Aggregating... {Colors.RESET}")

    def aggregate_wrapper():
        aggregated = aggregate(hg, motifs)
        for i in range(len(aggregated)):
            print(f"Motif {i}", aggregated[i])

    time_function_p(aggregate_wrapper)


(hg, motifs), _ = time_function_p(
    lambda: OldLoader("hospital")
    .order(3)
    .construction_method(
        StandardConstructionMethod(
            limit_edge_size=4,
            weighted=False,
            normalization_method=NormalizationMethod.NONE,
        )
    )
    .load()
)
time_aggregate(hg, motifs)

(hg, motifs), _ = time_function_p(
    lambda: OldLoader("hospital")
    .order(3)
    .construction_method(
        StandardConstructionMethod(
            limit_edge_size=4,
            weighted=True,
            normalization_method=NormalizationMethod.NONE,
        )
    )
    .load()
)
time_aggregate(hg, motifs)


(hg, motifs), _ = time_function_p(
    lambda: OldLoader("hospital")
    .order(4)
    .construction_method(
        StandardConstructionMethod(
            limit_edge_size=4,
            weighted=False,
            normalization_method=NormalizationMethod.NONE,
        )
    )
    .load()
)
time_aggregate(hg, motifs)


(hg, motifs), _ = time_function_p(
    lambda: OldLoader("hospital")
    .order(4)
    .construction_method(
        StandardConstructionMethod(
            limit_edge_size=4,
            weighted=True,
            normalization_method=NormalizationMethod.NONE,
        )
    )
    .load()
)
time_aggregate(hg, motifs)
