from src.graph import NormalizationMethod, StandardConstructionMethod
from tests.util import Colors, OldLoader, time_function_p

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
    .ignore_cache(True)
    .load()
)
total = 0
for motif, instances in motifs:
    print(motif, len(instances))
    total += len(instances)
print(
    f"{Colors.BOLD}{Colors.GREEN}Total instances: {Colors.RESET}{Colors.GREEN}{total}{Colors.RESET}"
)

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
    .ignore_cache(True)
    .load()
)
total = 0
for motif, instances in motifs:
    print(motif, len(instances))
    total += len(instances)
print(
    f"{Colors.BOLD}{Colors.GREEN}Total instances: {Colors.RESET}{Colors.GREEN}{total}{Colors.RESET}"
)

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
    .ignore_cache(True)
    .load()
)
total = 0
for motif, instances in motifs:
    print(motif, len(instances))
    total += len(instances)
print(
    f"{Colors.BOLD}{Colors.GREEN}Total instances: {Colors.RESET}{Colors.GREEN}{total}{Colors.RESET}"
)

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
    .ignore_cache(True)
    .load()
)
total = 0
for motif, instances in motifs:
    print(motif, len(instances))
    total += len(instances)
print(
    f"{Colors.BOLD}{Colors.GREEN}Total instances: {Colors.RESET}{Colors.GREEN}{total}{Colors.RESET}"
)
