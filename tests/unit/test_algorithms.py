#!/usr/bin/env python3
"""
SCIO Algorithm Tests
====================

Comprehensive tests for all algorithm modules.
"""

import pytest
import math
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# SORTING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSortingAlgorithms:
    """Tests for sorting algorithms."""

    @pytest.fixture
    def unsorted_list(self) -> List[int]:
        return [64, 34, 25, 12, 22, 11, 90]

    @pytest.fixture
    def sorted_list(self) -> List[int]:
        return [11, 12, 22, 25, 34, 64, 90]

    @pytest.fixture
    def reverse_sorted(self) -> List[int]:
        return [90, 64, 34, 25, 22, 12, 11]

    def test_bubble_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import bubble_sort
        assert bubble_sort(unsorted_list) == sorted_list

    def test_bubble_sort_reverse(self, unsorted_list, reverse_sorted):
        from scio.algorithms.sorting import bubble_sort
        assert bubble_sort(unsorted_list, reverse=True) == reverse_sorted

    def test_bubble_sort_empty(self):
        from scio.algorithms.sorting import bubble_sort
        assert bubble_sort([]) == []

    def test_selection_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import selection_sort
        assert selection_sort(unsorted_list) == sorted_list

    def test_insertion_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import insertion_sort
        assert insertion_sort(unsorted_list) == sorted_list

    def test_merge_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import merge_sort
        assert merge_sort(unsorted_list) == sorted_list

    def test_quick_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import quick_sort
        assert quick_sort(unsorted_list) == sorted_list

    def test_heap_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import heap_sort
        assert heap_sort(unsorted_list) == sorted_list

    def test_counting_sort(self):
        from scio.algorithms.sorting import counting_sort
        arr = [4, 2, 2, 8, 3, 3, 1]
        assert counting_sort(arr) == [1, 2, 2, 3, 3, 4, 8]

    def test_radix_sort(self):
        from scio.algorithms.sorting import radix_sort
        arr = [170, 45, 75, 90, 802, 24, 2, 66]
        assert radix_sort(arr) == sorted(arr)

    def test_bucket_sort(self):
        from scio.algorithms.sorting import bucket_sort
        arr = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
        result = bucket_sort(arr)
        assert result == sorted(arr)

    def test_tim_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import tim_sort
        assert tim_sort(unsorted_list) == sorted_list

    def test_shell_sort(self, unsorted_list, sorted_list):
        from scio.algorithms.sorting import shell_sort
        assert shell_sort(unsorted_list) == sorted_list

    def test_sort_with_key(self):
        from scio.algorithms.sorting import merge_sort
        data = [{"name": "Charlie", "age": 30}, {"name": "Alice", "age": 25}, {"name": "Bob", "age": 35}]
        sorted_data = merge_sort(data, key=lambda x: x["age"])
        assert [d["name"] for d in sorted_data] == ["Alice", "Charlie", "Bob"]


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCHING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchingAlgorithms:
    """Tests for searching algorithms."""

    @pytest.fixture
    def sorted_array(self) -> List[int]:
        return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    def test_linear_search_found(self, sorted_array):
        from scio.algorithms.searching import linear_search
        assert linear_search(sorted_array, 7) == 3

    def test_linear_search_not_found(self, sorted_array):
        from scio.algorithms.searching import linear_search
        assert linear_search(sorted_array, 8) == -1

    def test_binary_search_found(self, sorted_array):
        from scio.algorithms.searching import binary_search
        assert binary_search(sorted_array, 7) == 3

    def test_binary_search_not_found(self, sorted_array):
        from scio.algorithms.searching import binary_search
        assert binary_search(sorted_array, 8) == -1

    def test_binary_search_left(self):
        from scio.algorithms.searching import binary_search_left
        arr = [1, 2, 2, 2, 3, 4, 5]
        assert binary_search_left(arr, 2) == 1

    def test_binary_search_right(self):
        from scio.algorithms.searching import binary_search_right
        arr = [1, 2, 2, 2, 3, 4, 5]
        assert binary_search_right(arr, 2) == 4

    def test_interpolation_search(self, sorted_array):
        from scio.algorithms.searching import interpolation_search
        assert interpolation_search(sorted_array, 7) == 3

    def test_exponential_search(self, sorted_array):
        from scio.algorithms.searching import exponential_search
        assert exponential_search(sorted_array, 7) == 3

    def test_jump_search(self, sorted_array):
        from scio.algorithms.searching import jump_search
        assert jump_search(sorted_array, 7) == 3

    def test_fibonacci_search(self, sorted_array):
        from scio.algorithms.searching import fibonacci_search
        assert fibonacci_search(sorted_array, 7) == 3

    def test_ternary_search(self, sorted_array):
        from scio.algorithms.searching import ternary_search
        assert ternary_search(sorted_array, 7) == 3

    def test_find_min(self):
        from scio.algorithms.searching import find_min
        arr = [5, 2, 8, 1, 9, 3]
        idx, val = find_min(arr)
        assert val == 1
        assert idx == 3

    def test_find_max(self):
        from scio.algorithms.searching import find_max
        arr = [5, 2, 8, 1, 9, 3]
        idx, val = find_max(arr)
        assert val == 9
        assert idx == 4

    def test_find_kth_smallest(self):
        from scio.algorithms.searching import find_kth_smallest
        arr = [7, 10, 4, 3, 20, 15]
        assert find_kth_smallest(arr, 3) == 7

    def test_find_kth_largest(self):
        from scio.algorithms.searching import find_kth_largest
        arr = [7, 10, 4, 3, 20, 15]
        assert find_kth_largest(arr, 2) == 15


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphAlgorithms:
    """Tests for graph algorithms."""

    @pytest.fixture
    def simple_graph(self):
        from scio.algorithms.graph import Graph
        g = Graph(directed=False)
        g.add_edge('A', 'B', 1)
        g.add_edge('A', 'C', 4)
        g.add_edge('B', 'C', 2)
        g.add_edge('B', 'D', 5)
        g.add_edge('C', 'D', 1)
        return g

    @pytest.fixture
    def directed_graph(self):
        from scio.algorithms.graph import Graph
        g = Graph(directed=True)
        g.add_edge('A', 'B', 1)
        g.add_edge('A', 'C', 2)
        g.add_edge('B', 'D', 3)
        g.add_edge('C', 'D', 1)
        return g

    def test_bfs(self, simple_graph):
        from scio.algorithms.graph import bfs
        distances = bfs(simple_graph, 'A')
        assert distances['A'] == 0
        assert distances['B'] == 1
        assert distances['C'] == 1
        assert distances['D'] == 2

    def test_dfs(self, simple_graph):
        from scio.algorithms.graph import dfs
        visited = dfs(simple_graph, 'A')
        assert 'A' in visited
        assert len(visited) == 4

    def test_dijkstra(self, simple_graph):
        from scio.algorithms.graph import dijkstra
        distances, predecessors = dijkstra(simple_graph, 'A')
        assert distances['A'] == 0
        assert distances['B'] == 1
        assert distances['C'] == 3
        assert distances['D'] == 4

    def test_bellman_ford(self, directed_graph):
        from scio.algorithms.graph import bellman_ford
        distances, predecessors, has_negative = bellman_ford(directed_graph, 'A')
        assert not has_negative
        assert distances['A'] == 0
        assert distances['D'] == 3

    def test_floyd_warshall(self, simple_graph):
        from scio.algorithms.graph import floyd_warshall
        all_pairs = floyd_warshall(simple_graph)
        assert all_pairs['A']['D'] == 4

    def test_kruskal_mst(self, simple_graph):
        from scio.algorithms.graph import kruskal_mst
        mst = kruskal_mst(simple_graph)
        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 4  # A-B(1) + B-C(2) + C-D(1)

    def test_prim_mst(self, simple_graph):
        from scio.algorithms.graph import prim_mst
        mst = prim_mst(simple_graph, 'A')
        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 4

    def test_topological_sort(self, directed_graph):
        from scio.algorithms.graph import topological_sort
        order = topological_sort(directed_graph)
        assert order.index('A') < order.index('B')
        assert order.index('A') < order.index('C')
        assert order.index('B') < order.index('D')

    def test_find_cycle_no_cycle(self, directed_graph):
        from scio.algorithms.graph import find_cycle
        cycle = find_cycle(directed_graph)
        assert cycle is None

    def test_find_cycle_with_cycle(self):
        from scio.algorithms.graph import Graph, find_cycle
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        cycle = find_cycle(g)
        assert cycle is not None
        assert len(cycle) >= 3

    def test_is_bipartite_true(self):
        from scio.algorithms.graph import Graph, is_bipartite
        g = Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        g.add_edge(4, 1)
        is_bip, coloring = is_bipartite(g)
        assert is_bip

    def test_is_bipartite_false(self):
        from scio.algorithms.graph import Graph, is_bipartite
        g = Graph(directed=False)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)  # Triangle - not bipartite
        is_bip, _ = is_bipartite(g)
        assert not is_bip

    def test_strongly_connected_components(self):
        from scio.algorithms.graph import Graph, strongly_connected_components
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        sccs = strongly_connected_components(g)
        assert len(sccs) == 3  # {0,1,2}, {3}, {4}

    def test_a_star(self):
        from scio.algorithms.graph import Graph, a_star
        g = Graph(directed=False)
        g.add_edge('A', 'B', 1)
        g.add_edge('B', 'C', 1)
        g.add_edge('A', 'C', 3)

        def heuristic(node):
            # Simple heuristic
            return 0

        path, cost = a_star(g, 'A', 'C', heuristic)
        assert path == ['A', 'B', 'C']
        assert cost == 2


# ═══════════════════════════════════════════════════════════════════════════════
# STRING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStringAlgorithms:
    """Tests for string algorithms."""

    def test_kmp_search(self):
        from scio.algorithms.string import kmp_search
        text = "ABABDABACDABABCABAB"
        pattern = "ABABCABAB"
        positions = kmp_search(text, pattern)
        assert 10 in positions

    def test_rabin_karp(self):
        from scio.algorithms.string import rabin_karp
        text = "ABABDABACDABABCABAB"
        pattern = "ABAB"
        positions = rabin_karp(text, pattern)
        assert len(positions) > 0

    def test_levenshtein_distance(self):
        from scio.algorithms.string import levenshtein_distance
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "abc") == 0

    def test_longest_common_subsequence(self):
        from scio.algorithms.dp import longest_common_subsequence
        lcs_length, lcs = longest_common_subsequence("ABCDGH", "AEDFHR")
        assert lcs_length == 3

    def test_longest_repeated_substring(self):
        from scio.algorithms.string import longest_repeated_substring
        result = longest_repeated_substring("banana")
        assert result == "ana"

    def test_is_palindrome(self):
        from scio.algorithms.string import is_palindrome
        assert is_palindrome("racecar")
        assert not is_palindrome("hello")

    def test_longest_palindromic_substring(self):
        from scio.algorithms.string import longest_palindromic_substring
        result = longest_palindromic_substring("babad")
        assert result in ["bab", "aba"]


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicProgramming:
    """Tests for dynamic programming algorithms."""

    def test_fibonacci(self):
        from scio.algorithms.dp import fibonacci
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(10) == 55
        assert fibonacci(20) == 6765

    def test_knapsack_01(self):
        from scio.algorithms.dp import knapsack_01
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        max_value, items = knapsack_01(weights, values, capacity)
        assert max_value == 220

    def test_longest_increasing_subsequence(self):
        from scio.algorithms.dp import longest_increasing_subsequence
        arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
        length, lis = longest_increasing_subsequence(arr)
        assert length == 6

    def test_coin_change(self):
        from scio.algorithms.dp import coin_change
        coins = [1, 5, 10, 25]
        amount = 30
        min_coins = coin_change(coins, amount)
        assert min_coins == 2  # 25 + 5

    def test_edit_distance(self):
        from scio.algorithms.dp import edit_distance
        dist = edit_distance("horse", "ros")
        assert dist == 3

    def test_matrix_chain_multiplication(self):
        from scio.algorithms.dp import matrix_chain_multiplication
        dimensions = [10, 30, 5, 60]
        min_ops, order = matrix_chain_multiplication(dimensions)
        assert min_ops == 4500


# ═══════════════════════════════════════════════════════════════════════════════
# MATH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathAlgorithms:
    """Tests for mathematical algorithms."""

    def test_gcd(self):
        from scio.algorithms.math import gcd
        assert gcd(48, 18) == 6
        assert gcd(17, 13) == 1

    def test_lcm(self):
        from scio.algorithms.math import lcm
        assert lcm(4, 6) == 12

    def test_is_prime(self):
        from scio.algorithms.math import is_prime
        assert is_prime(2)
        assert is_prime(17)
        assert not is_prime(1)
        assert not is_prime(4)

    def test_sieve_of_eratosthenes(self):
        from scio.algorithms.math import sieve_of_eratosthenes
        primes = sieve_of_eratosthenes(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_prime_factors(self):
        from scio.algorithms.math import prime_factors
        assert prime_factors(60) == [2, 2, 3, 5]

    def test_mod_pow(self):
        from scio.algorithms.math import mod_pow
        assert mod_pow(2, 10, 1000) == 24
        assert mod_pow(3, 5, 7) == 5

    def test_factorial(self):
        from scio.algorithms.dp import factorial
        assert factorial(0) == 1
        assert factorial(5) == 120

    def test_binomial(self):
        from scio.algorithms.math import binomial
        assert binomial(5, 2) == 10
        assert binomial(10, 0) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataStructures:
    """Tests for data structures."""

    def test_stack(self):
        from scio.algorithms.data_structures import Stack
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        assert s.pop() == 3
        assert s.peek() == 2
        assert s.size() == 2

    def test_queue(self):
        from scio.algorithms.data_structures import Queue
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        assert q.dequeue() == 1
        assert q.front() == 2

    def test_deque(self):
        from scio.algorithms.data_structures import Deque
        d = Deque()
        d.push_back(1)
        d.push_back(2)
        d.push_front(0)
        assert d.pop_front() == 0
        assert d.pop_back() == 2
        assert d.size() == 1

    def test_avl_tree(self):
        from scio.algorithms.data_structures import AVLTree
        avl = AVLTree()
        for val in [5, 3, 7, 1, 4, 6, 8]:
            avl.insert(val)
        assert avl.search(4)
        assert not avl.search(10)
        assert avl.inorder() == [1, 3, 4, 5, 6, 7, 8]

    def test_heap(self):
        from scio.algorithms.data_structures import MinHeap
        h = MinHeap()
        for val in [5, 3, 7, 1, 4]:
            h.push(val)
        assert h.pop() == 1
        assert h.pop() == 3
        assert h.peek() == 4

    def test_trie(self):
        from scio.algorithms.data_structures import Trie
        t = Trie()
        t.insert("apple")
        t.insert("app")
        t.insert("application")
        assert t.search("app")
        assert t.search("apple")
        assert not t.search("appl")
        assert t.starts_with("app")
        words = t.words_with_prefix("app")
        assert len(words) == 3

    def test_union_find(self):
        from scio.algorithms.data_structures import UnionFind
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.connected(0, 1)
        assert not uf.connected(0, 2)
        uf.union(1, 3)
        assert uf.connected(0, 3)

    def test_segment_tree(self):
        from scio.algorithms.data_structures import SegmentTree
        arr = [1, 3, 5, 7, 9, 11]
        st = SegmentTree(arr)
        assert st.query(1, 4) == 15  # 3+5+7
        st.update(2, 10)  # Change 5 to 10
        assert st.query(1, 4) == 20  # 3+10+7

    def test_fenwick_tree(self):
        from scio.algorithms.data_structures import FenwickTree
        ft = FenwickTree.from_array([1, 2, 3, 4, 5])
        assert ft.prefix_sum(2) == 6  # 1+2+3
        assert ft.range_sum(1, 3) == 9  # 2+3+4

    def test_lru_cache(self):
        from scio.algorithms.data_structures import LRUCache
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        cache.put("c", 3)  # Evicts "b"
        assert cache.get("b") is None
        assert cache.get("c") == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
