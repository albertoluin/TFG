# -*- coding: utf-8 -*-
from approximationLib import graphProblems
import pprint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

seed = 0


"""MAX-CUT"""
print("MAX-CUT:")
edges =  [('v1', 'v4'), ('v1', 'v2'), ('v2', 'v3'),
               ('v3', 'v4'), ('v5', 'v4'), ('v3', 'v5')]
g = graphProblems.Graph(connections=edges)

pretty_print = pprint.PrettyPrinter()
pretty_print.pprint(g._graph)

#Dibujamos el grafo

print("\nGrafo de entrada:")


G = nx.Graph()
G.add_edges_from(edges)
values = [0 for node in G.nodes()]
options = {"node_size": 700}
cmap=plt.get_cmap('PiYG')
random.seed(seed)
np.random.seed(seed)
nx.draw(G, cmap=cmap, node_color=values, with_labels=True,
        font_color='white', **options)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


S, cut_value = graphProblems.max_cut(g, seed=0)

print("Conjunto S: ", S, "\nValor del corte: ", cut_value)



val_map = {s:1.0 for s in S}

values = [val_map.get(node, 0.5) for node in G.nodes()]
options = {"node_size": 700}
cmap=plt.get_cmap('PiYG')
random.seed(seed)
np.random.seed(seed)
nx.draw(G, cmap=cmap, node_color=values, with_labels=True,
        font_color='white', **options)

cmap=plt.get_cmap('Greys')

plt.scatter([],[],c=[cmap(0)], label='Valor del corte: '+str(cut_value))
plt.legend(prop={'size':13})
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


"""MINIMUM VERTEX COVER"""
print("MINIMUM VERTEX COVER")

edges =  [('v1', 'v4'), ('v1', 'v2'), ('v2', 'v3'),
               ('v3', 'v4'), ('v5', 'v4'), ('v3', 'v5'), ('v3', 'v6')]
g = graphProblems.Graph(connections=edges)

pretty_print = pprint.PrettyPrinter()
pretty_print.pprint(g._graph)

#Dibujamos el grafo

print("\nGrafo de entrada:")


G = nx.Graph()
G.add_edges_from(edges)
values = [0 for node in G.nodes()]
options = {"node_size": 700}
cmap=plt.get_cmap('seismic')
random.seed(seed)
np.random.seed(seed)
nx.draw(G, cmap=cmap, node_color=values, with_labels=True,
        font_color='white', **options)

cmap=plt.get_cmap('Greys')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

C, cover_size = graphProblems.vertex_cover(g, seed=11)
print("Cubrimiento: ", C, "\nNúmero de vértices del cubrimiento ", cover_size)



val_map = {c:2.0 for c in C}

values = [val_map.get(node, 0.25) for node in G.nodes()]
options = {"node_size": 700}
cmap=plt.get_cmap('seismic')
random.seed(seed)
np.random.seed(seed)
nx.draw(G, cmap=cmap, node_color=values, with_labels=True,
        font_color='white', **options)

plt.scatter([],[], c=[cmap(9999)], label='Cubrimiento por vértices')
plt.legend()
plt.show()