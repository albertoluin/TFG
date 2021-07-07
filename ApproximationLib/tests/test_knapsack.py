# -*- coding: utf-8 -*-

from approximationLib import knapsack
def test():
    w=[10, 40, 20, 30]
    v=[60, 40, 100, 120]
    c=50
    assert knapsack.greedy(w,v,c)[0] == 160
    assert knapsack.pseudopolynomial(w,v,c) == 220
    assert knapsack.fptas(w,v,c,1.05) == 220
    
    w=[1, 1, 1, 20]
    v=[1, 1, 1, 19]
    c=20
    assert knapsack.greedy(w,v,c)[0] == 3
    assert knapsack.greedy_v2(w,v,c)[0] == 19
    
    
