def orca(adj: list[list[int]], sort_degrees: bool = False) -> int:
    c1 = 0
    c2 = 0
    for x in adj:
        c1 += len(x) * (len(x) - 1) // 2
        c2 += len(x) * len(x) - len(x)

    print("C1: ", c1)
    print("C2: ", c2)
    print("C1 * 2: ", c1 * 2)

    print("T1: ", c1 - c2 // 2)
    print("T2: ", 2 * c1 - c2)
    print("T3: ", (2 * c2 - c1) // 3)
    return (c2 - c1) // 3
