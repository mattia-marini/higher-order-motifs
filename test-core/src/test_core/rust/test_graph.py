def test():
    import rust_core.graph as rc_graph
    uw_hg = rc_graph.UnweightedHypergraph()
    uw_hg.insert_hx((1,2,4))
    uw_hg.insert_hx((2,4))

    w_hg = rc_graph.WeightedHypergraph()
    w_hg.insert_hx((1,2,4), 1.5)
    w_hg.insert_hx((2,4), 42)

    print(uw_hg.edges())
    print(w_hg.edges())
