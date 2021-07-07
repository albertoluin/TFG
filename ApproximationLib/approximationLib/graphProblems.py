# -*- coding: utf-8 -*-
#https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
from collections import defaultdict
import random
import copy


class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self._edges = []
        self.add_connections(connections)
    
    def get_edges(self):
        return self._edges

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)
            self._edges.append((node1, node2))

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None
    
    def arc(self, v, A):
        """Calculate the number of arcs between v and the subset of vertices A
        and v and V\A"""
        
        arcA = 0 #arc(v,V)
        arcV_A = 0 #arc(v,V\A)
        
        for w in self._graph[v]:
            if w in A:
                arcA = arcA + 1
            else:
                arcV_A = arcV_A + 1
        
        return arcA, arcV_A
    
    def cut_value(self, S):
        """Calculate the cut value when S is the subset of vertices"""
        
        cut_val = 0
        for v in S:
            for w in self._graph[v]:
                if w not in S:
                    cut_val = cut_val + 1
        
        return cut_val 
        
        
            
        
        
        
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

def max_cut(graph, seed):
    """ 
    Algoritmo greedy 2-aproximado para el problema del corte máximo. 
    Dado un grafo no dirigido G=(V,E), consiste en partir V en dos conjuntos S
    y V\S de forma que haya un número máximo de arcos entre S y V\S.
    :param graph: objeto de la clase graph. Precondición: El grafo no tiene
    arcos que unen un vértice consigo mismo.
    :param seed: semilla para elegir de forma aleatoria el conjunto S inicial.
    :return: Conjunto de vértices S y el valor del corte.
    """

    #Conjunto de vértices
    V = list(graph._graph)
    
    #Elegimos de forma aleatoria un subconjunto de vértices
    random.seed(seed)
    size = random.choice(range(len(V))) #Tamaño de S
    S = random.sample(V, size)
    
    S_new = S.copy()
    change = True
    while(change):
        for v in V:
            #Obtenemos el número de arcos que hay entre v y S y entre v y V\S
            arcS, arcV_S = graph.arc(v, S_new)
            if v in S:
                if arcV_S < arcS:
                    S_new.remove(v)
            else:
                if arcV_S > arcS:
                    S_new.append(v)
            
        #Comprobamos si ha cambiado
        change = (set(S) != set(S_new))
        S = S_new.copy()
        
    return S, graph.cut_value(S)
        

def vertex_cover(graph, seed):
    """ 
    Algoritmo greedy 2-aproximado para el problema del cubrimiento mínimo por
    vértices. 
    Dado un grafo no dirigido G=(V,E), consiste en encontrar el cubrimiento 
    por vértices C de G de menor número de vértices.
    :param graph: objeto de la clase graph. Precondición: El grafo no tiene
    arcos que unen un vértice consigo mismo.
    :param seed: semilla para elegir de forma aleatoria las aristas.
    :return: Conjunto de vértices C (cubrimiento) y el número de vértices del mismo.
    """                  

    C = []
    g = copy.deepcopy(graph)
    #Conjunto de vértices
    V = list(g._graph)
    #Aristas
    edges = g.get_edges()
    
    while(len(edges)>0):
        
        #Elegimos una arista cualquiera
        random.seed(seed)
        e = random.choice(edges)
        #Añadimos sus extremos a C
        C.append(e[0])
        C.append(e[1])
        #Borramos los dos vértices y sus aristas
        V.remove(e[0])
        V.remove(e[1])
        edges_copy = copy.deepcopy(edges)
        for ed in edges_copy:
            if (e[0] in ed) or (e[1] in ed):
                edges.remove(ed)
    
    return C, len(C)
                
    











