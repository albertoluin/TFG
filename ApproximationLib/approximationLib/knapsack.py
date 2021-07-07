# -*- coding: utf-8 -*-
import numpy as np

#Como ejecutar: python setup.py pytest
def greedy(weights, vals, capacity):
    """Algoritmo greedy para el problema de la mochila que consiste en ordenar
     los ítems de forma no creciente respecto al ratio value/weight, e ir introduciendo
     los objetos en ese orden hasta que no quepan más.
     Precondición: Los pesos son distintos de cero, vals[i]<= capacity (se puede suponer
     sin pérdida de generalidad) y weights y vals deben tener el mismo tamaño.
    :param weights: lista de pesos de los objetos [w1, w2, .., wn]
    :param vals: lista de valores de los objetos [v1, v2, .., vn]
    :param capacity: capacidad de la mochila
    :return: una tupla formada por el coste y la configuración final de la mochila
     Por ejemplo: coste, [0,0,1,0,1,1,0] donde un 1 significa que el objeto está dentro
     de la mochila
     
     Eficiencia: O(nlog(n)) debido a que hay que hacer la ordenación por ratio.
     """
    #Almacenamos los pares (ratio, número de objeto) para cada objeto
    ratios = [(i/j,k) for i,j,k in zip(vals, weights, range(len(weights)))] 
    #Ordenamos de mayor a menor los ratios
    #Por ejemplo, este sería un posible resultado: [(4.0, 2), (2.5, 1), (2.0, 0), (0.5, 3)]
    #donde el segundo elemento de cada par es el número del objeto al que corresponde ese ratio
    ratios.sort(reverse=True) 
    
    mochila = np.zeros(len(weights)) #Al principio vacía
    
    valorTotal = 0
    
    for r in ratios:
        if(weights[r[1]] <= capacity):
            mochila[r[1]] = 1
            capacity = capacity - weights[r[1]]
            valorTotal = valorTotal + vals[r[1]]
    
    return valorTotal, mochila
            

def greedy_v2(weights, vals, capacity):
    """Algoritmo greedy para el problema de la mochila que consiste en ordenar
     los ítems de forma no creciente respecto al ratio value/weight, e ir introduciendo
     los objetos en ese orden hasta que no quepan más.
     Además, para asegurar que el algoritmo es 2-aproximado, comprobamos si la solución obtenida
     es mejor que la solución consistente en introducir únicamente el objeto de mayor valor
     Precondición: Los pesos son distintos de cero, vals[i]<= capacity (se puede suponer
     sin pérdida de generalidad) y weights y vals deben tener el mismo tamaño.
    :param weights: lista de pesos de los objetos [w1, w2, .., wn]
    :param vals: lista de valores de los objetos [v1, v2, .., vn]
    :param capacity: capacidad de la mochila
    :return: una tupla formada por el coste y la configuración final de la mochila
     Por ejemplo: coste, [0,0,1,0,1,1,0] donde un 1 significa que el objeto está dentro
     de la mochila
     
     Eficiencia: O(nlog(n)) debido a que hay que hacer la ordenación por ratio.
     """
    #Almacenamos los pares (ratio, número de objeto) para cada objeto
    ratios = [(i/j,k) for i,j,k in zip(vals, weights, range(len(weights)))] 
    #Ordenamos de mayor a menor los ratios
    #Por ejemplo, este sería un posible resultado: [(4.0, 2), (2.5, 1), (2.0, 0), (0.5, 3)]
    #donde el segundo elemento de cada par es el número del objeto al que corresponde ese ratio
    ratios.sort(reverse=True) 
    
    mochila = np.zeros(len(weights)) #Al principio vacía
    
    valorTotal = 0
    
    for r in ratios:
        if(weights[r[1]] <= capacity):
            mochila[r[1]] = 1
            capacity = capacity - weights[r[1]]
            valorTotal = valorTotal + vals[r[1]]
    
    maxvalue = max(vals)
    if(valorTotal < maxvalue): #Si la solución con solo un objeto es mejor, devolvemos esa
        maxpos = vals.index(maxvalue)
        mochila = np.zeros(len(weights))
        mochila[maxpos] = 1
        valorTotal = maxvalue
        
    
    return valorTotal, mochila

#-----------------------------Algoritmo pseudopolinómico recursivo----------------------------
    


def pseudopolynomial (weights, vals, capacity):
    """Algoritmo pseudopolinómico para el problema de la mochila. Es un algoritmo de
     programación dinámica.
     Precondición: Los pesos son distintos de cero, vals[i]<= capacity (se puede suponer
     sin pérdida de generalidad) y weights y vals deben tener el mismo tamaño.
    :param weights: lista de pesos de los objetos [w1, w2, .., wn]
    :param vals: lista de valores de los objetos [v1, v2, .., vn]
    :param capacity: capacidad de la mochila
    :return: una tupla formada por el coste y la configuración final de la mochila
     Por ejemplo: coste, [0,0,1,0,1,1,0] donde un 1 significa que el objeto está dentro
     de la mochila
     
     Eficiencia: O(n^2 V), donde V es el valor máximo de los valores de los objetos.
     El valor de V es exponencial en función del tamaño que se necesita para almacenar V,
     luego este algoritmo es exponencial en función del tamaño de la entrada.
     """
    n = len(weights)
    V = max(vals)
    infinito = float('inf')
    
    #Defino los W[i,j] para i=0,..,n y j=0,..,nV
    W = np.zeros((n+1, n*V+1))
    
    for j in range(1,n*V +1):
        W[0,j]= infinito
    
    for i in range(1,n+1):
        for j in range(1,n*V +1):
            
            if (j==0):
                W[i,j]=0
            elif (j > 0 and i==0):
                W[i,j]=infinito
            elif (i > 0 and j > 0 and vals[i-1] > j):
                W[i,j] = W[i-1,j]
            else:
                W[i,j] = min(W[i-1, j], W[i-1, j-vals[i-1]] + weights[i-1])
                
    valorTotal = 0
    for i in range(n*V, -1, -1):
        if(W[n,i] <= capacity):
            valorTotal = i
            break
                
    return valorTotal
            
                
    
#----------------------------FPTAS----------------------------

def fptas(weights, vals, capacity, delta):
    
    n = len(vals)
    vals_ = np.zeros(n)
    
    V = max(vals)    
    k = 2**int(np.log2(((delta-1)*V)/n))
    
    for i in range(n):
        vals_[i] = k*int(vals[i]/k)
    
    return pseudopolynomial(weights, vals_.astype(np.int64), capacity)
        
    
    
    
    
    
    
    