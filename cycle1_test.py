import pytest
import networkx as nx
from cycle1 import has_cycle1, WeightedDiGraph
from testcases import parse_testcases

testcases = parse_testcases("testcases.txt")

def run_testcase(input:str):
    graph = WeightedDiGraph(*input)
    return has_cycle1(graph)
    

@pytest.mark.parametrize("testcase", testcases, ids=[testcase["name"] for testcase in testcases])
def test_cases(testcase):
    actual_output = run_testcase(testcase["input"])
    assert actual_output == testcase["output"], f"Expected {testcase['output']}, got {actual_output}"


def test_empty_graph():
    assert has_cycle1(WeightedDiGraph()) == False

def test_single_node_no_edges():
    g = WeightedDiGraph()
    g.add_node(0)
    assert has_cycle1(g) == False

def test_self_loop_weight_less_than_1():
    assert has_cycle1(WeightedDiGraph([0, 0, 0.5])) == True

def test_self_loop_weight_greater_than_1():
    assert has_cycle1(WeightedDiGraph([0, 0, 2.0])) == False

def test_self_loop_weight_exactly_1():
    assert has_cycle1(WeightedDiGraph([0, 0, 1.0])) == False

def test_two_cycle_product_less_than_1():
    # 0.5 * 0.5 = 0.25 < 1
    assert has_cycle1(WeightedDiGraph([0, 1, 0.5], [1, 0, 0.5])) == True

def test_two_cycle_product_greater_than_1():
    # 2.0 * 2.0 = 4.0 > 1
    assert has_cycle1(WeightedDiGraph([0, 1, 2.0], [1, 0, 2.0])) == False

def test_two_cycle_product_exactly_1():
    # 2.0 * 0.5 = 1.0, not < 1
    assert has_cycle1(WeightedDiGraph([0, 1, 2.0], [1, 0, 0.5])) == False

def test_dag_no_cycle():
    assert has_cycle1(WeightedDiGraph([0, 1, 0.1], [1, 2, 0.1], [0, 2, 0.1])) == False

def test_disconnected_one_component_has_cycle():
    # Component 0-1-2 has product 0.1^3 < 1; component 3-4 has product 2*2 > 1, no full cycle back
    g = WeightedDiGraph([0, 1, 0.1], [1, 2, 0.1], [2, 0, 0.1], [3, 4, 2.0])
    assert has_cycle1(g) == True

def test_disconnected_no_component_has_product_cycle():
    # Component 0-1-2: 2*2*2 = 8 > 1; component 3-4: no cycle
    g = WeightedDiGraph([0, 1, 2.0], [1, 2, 2.0], [2, 0, 2.0], [3, 4, 0.5])
    assert has_cycle1(g) == False

def test_long_cycle_product_less_than_1():
    # Chain 0->1->...->9->0, each weight 0.9; 0.9^10 ≈ 0.349 < 1
    edges = [[i, i+1, 0.9] for i in range(9)] + [[9, 0, 0.9]]
    assert has_cycle1(WeightedDiGraph(*edges)) == True

def test_long_cycle_product_greater_than_1():
    # each weight 1.1; 1.1^10 ≈ 2.59 > 1
    edges = [[i, i+1, 1.1] for i in range(9)] + [[9, 0, 1.1]]
    assert has_cycle1(WeightedDiGraph(*edges)) == False

def test_multiple_cycles_only_one_qualifies():
    # Cycle A: 0->1->0 with product 2.0*2.0 = 4 > 1
    # Cycle B: 2->3->2 with product 0.3*0.3 = 0.09 < 1
    g = WeightedDiGraph([0, 1, 2.0], [1, 0, 2.0], [2, 3, 0.3], [3, 2, 0.3])
    assert has_cycle1(g) == True

def test_performance_large_graph():
    import random, time
    random.seed(42)
    n, m = 1000, 100000
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for _ in range(m):
        u = random.randrange(n)
        v = random.randrange(n)
        w = random.uniform(0.5, 2.0)
        g.add_edge(u, v, weight=w)
    start = time.time()
    has_cycle1(g)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
