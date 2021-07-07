# -*- coding: utf-8 -*-
from approximationLib import knapsack
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks", color_codes=True)
#Para el genético
from pyeasyga import pyeasyga
import random

"""Los datasets utilizados se encuentran en:
    http://artemisa.unicauca.edu.co/~johnyortega/instances_01_KP/"""


"""------------------Funciones de lectura de datos--------------------------"""
def leer_fichero(fname, delimiter = ' '):
    return np.loadtxt(fname, delimiter = delimiter)

def leer_datos(data_type):
    """Cada instancia (datos[i]) contiene en la primera línea
    los valores n wmax, referidos al número de ítems y a la capacidad max
    de la mochila"""
    path_dir = "./datasets/" + data_type
    optimum_path = path_dir + "-optimum"
    files = os.listdir(path_dir)
    datos = []
    optimal_values = []
    files_names = []
    for file in files:
        files_names.append(file)
        file_path = os.path.join(path_dir, file)
        #print("Leyendo ", file_path)
        datos.append(leer_fichero(file_path))
        
        file_path_opt = os.path.join(optimum_path, file)
        optimal_values.append(leer_fichero(file_path_opt))
    return datos, optimal_values, files_names

def leer_datos2(data_type):
    """Cada instancia (datos[i]) contiene en la primera línea
    los valores n wmax, referidos al número de ítems y a la capacidad max
    de la mochila"""
    path_dir = "./datasets/" + data_type
    files = os.listdir(path_dir)
    datos = []
    files_names = []
    info = []
    for file in files:
        files_names.append(file)
        file_path = os.path.join(path_dir, file)
        #print("Leyendo ", file_path)
        datos.append(np.loadtxt(file_path,skiprows=2))
        info.append(np.loadtxt(file_path,max_rows=2))
        
    return datos, info, files_names

"""------------------------------------------------------------------------"""

"""----------Funciones auxiliares para el algoritmo genético---------------"""

def transform_input(w,v):
    data = []
    for i in range(len(w)):
        d = {'value': v[i], 'weight': w[i]}
        data.append(d)
    return data
def transform_data_to_lists(data):
    w = []
    v = []
    for d in data:
        w.append(d['weight'])
        v.append(d['value'])
    return w, v

"""------------------------------------------------------------------------"""


print("Vamos a probar y comparar los siguiente algoritmos aproximados \
para el problema de la mochila:")
print("       ->Los que he implementado en la biblioteca approximationLib \
                  para este problema :")
print("              - Greedy")
print("              - Greedy 2-aproximado")
print("              - FPTAS con delta=1.5")
print("              - Pseudopolinómico (algoritmo exacto)")
print("\n       ->Utilizando la biblioteca pyeasyga:")
print("              - Algoritmo genético")
print("              - Algoritmo genético 2-aproximado")


input("\n--- Pulsar tecla para continuar ---\n")


print("\nEjecutamos los algoritmos con el conjunto de datos L-D")
print("Ejecutando... ")

datos, optimos, nombre_archivos = leer_datos("low-dimensional")

val_greedy = []
val_greedy_v2 = []
val_genetico = []
val_genetico_garantia = []
val_fptas = []
val_pseudopoly = []

ratio_gr = []
ratio_gr2 = []
ratio_gen = []
ratio_gen_approx = []
ratio_FPTAS = []
ratio_pseudop = []

tiempoG = []
tiempoG_v2 = []
tiempoGenetico = []
tiempoGeneticoGarantia = []
tiempoFPTAS = []
tiempoPseudo = []

random.seed(18)

for i in range(len(datos)):
    datos_actuales = datos[i]
    optimo = int(optimos[i])
    
    n = datos_actuales[0][0]
    wmax = datos_actuales[0][1]
    
    v = datos_actuales[1:,0]
    v = v.astype(np.int64)
    w = datos_actuales[1:,1]
    w = w.astype(np.int64)
    
    
    t0 = time.perf_counter()
    val_greedy.append(knapsack.greedy(w,v,wmax)[0])
    t1 = time.perf_counter()
    tiempoG.append(t1 - t0)
    
    t0 = time.perf_counter()
    val_greedy_v2.append(knapsack.greedy_v2(w,v,wmax)[0])
    t1 = time.perf_counter()
    tiempoG_v2.append(t1 - t0)
    
    #--------Genético sin garantía----------
    data = transform_input(w,v)

    ga = pyeasyga.GeneticAlgorithm(data, elitism=True)        # initialise the GA with data
    
    

    #Para crear la población inicial
    def create_individual(data):
        return [random.randint(0, 1) for _ in range(len(data))]

    
    ga.create_individual = create_individual
    
    # define a fitness function
    def fitness(individual, data):
        values, weights = 0, 0
        for selected, box in zip(individual, data):
            if selected:
                values += box.get('value')
                weights += box.get('weight')
        if weights > wmax:
            values = 0
        return values
    
    ga.fitness_function = fitness               # set the GA's fitness function
    t0 = time.perf_counter()
    ga.run() 
    t1 = time.perf_counter()
    tiempoGenetico.append(t1 - t0)                                   
    val_genetico.append(ga.best_individual()[0])          
    #------------------------
    
    
    #--------Genético con garantía----------
    data = transform_input(w,v)

    ga_g = pyeasyga.GeneticAlgorithm(data, elitism=True)        # initialise the GA with data
    
    
    first_time = True
    def create_individual_guarantee(data):
        w,v = transform_data_to_lists(data)
        global first_time
        if(first_time):
            sol = knapsack.greedy_v2(w,v,wmax)
            first_time = False
            return list(sol[1])
        else:    
            return [random.randint(0, 1) for _ in range(len(data))]
    
    ga_g.create_individual = create_individual_guarantee
    
    # define a fitness function
    def fitness(individual, data):
        values, weights = 0, 0
        for selected, box in zip(individual, data):
            if selected:
                values += box.get('value')
                weights += box.get('weight')
        if weights > wmax:
            values = 0
        return values
    
    ga_g.fitness_function = fitness               # set the GA's fitness function
    t0 = time.perf_counter()
    ga_g.run()                                      # run the GA
    t1 = time.perf_counter()
    tiempoGeneticoGarantia.append(t1 - t0)                                   
    val_genetico_garantia.append(ga_g.best_individual()[0])            
    #------------------------
    
    t0 = time.perf_counter()
    val_fptas.append(knapsack.fptas(w,v,wmax,1.5))
    t1 = time.perf_counter()
    tiempoFPTAS.append(t1 - t0)
    
    t0 = time.perf_counter()
    val_pseudopoly.append(knapsack.pseudopolynomial(w,v,wmax))
    t1 = time.perf_counter()
    tiempoPseudo.append(t1 - t0)
    
    
    ratio_gr.append(optimo/val_greedy[i])
    ratio_gr2.append(optimo/val_greedy_v2[i])
    ratio_gen.append(optimo/val_genetico[i])
    ratio_gen_approx.append(optimo/val_genetico_garantia[i])
    ratio_FPTAS.append(optimo/val_fptas[i])
    ratio_pseudop.append(optimo/val_pseudopoly[i])
    

    
    
#Obtenemos los ratios y tiempos medios

ratio_medio_gr = np.mean(ratio_gr)
ratio_medio_gr2 = np.mean(ratio_gr2)
ratio_medio_gen = np.mean(ratio_gen)
ratio_medio_gen_approx = np.mean(ratio_gen_approx)
ratio_medio_FPTAS = np.mean(ratio_FPTAS)
ratio_medio_pseudop = np.mean(ratio_pseudop)

tiempo_medio_gr = np.mean(tiempoG)
tiempo_medio_gr2 = np.mean(tiempoG_v2)
tiempo_medio_gen = np.mean(tiempoGenetico)
tiempo_medio_gen_approx = np.mean(tiempoGeneticoGarantia)
tiempo_medio_FPTAS = np.mean(tiempoFPTAS)
tiempo_medio_pseudop = np.mean(tiempoPseudo)



     
print("Diagrama de cajas para el ratio optimo / valor aproximado")
data_ratio = pd.DataFrame({"Greedy": ratio_gr, "Greedy \n 2-approx": ratio_gr2,
                           "Genético": ratio_gen, "Genético\n 2-aprox": ratio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_FPTAS, "Pseudop\n (óptimo)": ratio_pseudop})


sns.boxplot(data=data_ratio).set_title("Ratio óptimo/aproximación")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Ratio medio optimo / valor aproximado")
data_ratio_medio = pd.DataFrame({"Greedy": ratio_medio_gr, "Greedy \n 2-approx": ratio_medio_gr2,
                           "Genético": ratio_medio_gen, "Genético\n 2-aprox": ratio_medio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_medio_FPTAS, 
                           "Pseudop\n (óptimo)": ratio_medio_pseudop},index=[0])


sns.boxplot(data=data_ratio_medio).set_title("Ratio medio óptimo/aproximación")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Tiempo medio (en segundos)")

data_tiempo_medio = pd.DataFrame({"Greedy": tiempo_medio_gr, "Greedy \n 2-approx": tiempo_medio_gr2,
                           "Genético": tiempo_medio_gen, "Genético\n 2-aprox": tiempo_medio_gen_approx,
                           "FPTAS con\n delta=1.5": tiempo_medio_FPTAS, 
                           "Pseudop\n (óptimo)": tiempo_medio_pseudop},index=[0])


sns.boxplot(data=data_tiempo_medio, color=".25").set_title("Tiempo medio (en segundos)")
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

print("\nPodemos ejecutar los algoritmos con instancias de mayor tamaño (hasta 1000 ítems)")
print("Puede tardar unos minutos")
ejecutar = input("Presione la tecla --e-- para ejecutar,\
 o cualquier otra tecla para salir\n")

if(ejecutar == 'e'):
    
    print("Ejecutando... ")
    
    
    datos_, info, nombre_archivos = leer_datos2("different_sizes")
    
    #Ordeno según el número de items
    info = np.asarray(info)
    num_items = info[:,1]
    
    index = sorted(range(len(num_items)), key=lambda k: num_items[k])
    
    num_items = np.sort(num_items)
    
    datos = [datos_[i] for i in index]
    capacidades_ = info[:,0]
    capacidades = [capacidades_[i] for i in index]
    
    val_greedy = []
    val_greedy_v2 = []
    val_genetico = []
    val_genetico_garantia = []
    val_fptas = []
    val_pseudopoly = []
    
    ratio_gr = []
    ratio_gr2 = []
    ratio_gen = []
    ratio_gen_approx = []
    ratio_FPTAS = []
    ratio_pseudop = []
    
    tiempoG = []
    tiempoG_v2 = []
    tiempoGenetico = []
    tiempoGeneticoGarantia = []
    tiempoFPTAS = []
    tiempoPseudo = []
    
    random.seed(2)
    #datos=[datos[3]]
    #info = [info[3]]
    
    for i in range(len(datos)):
        datos_actuales = datos[i]
        info_actual = info[i]
        
        n = num_items[i]
        wmax = capacidades[i]
        
        v = datos_actuales[0:,0]
        v = v.astype(np.int64)
        w = datos_actuales[0:,1]
        w = w.astype(np.int64)
        
        
        t0 = time.perf_counter()
        val_greedy.append(knapsack.greedy(w,v,wmax)[0])
        t1 = time.perf_counter()
        tiempoG.append(t1 - t0)
        
        t0 = time.perf_counter()
        val_greedy_v2.append(knapsack.greedy_v2(w,v,wmax)[0])
        t1 = time.perf_counter()
        tiempoG_v2.append(t1 - t0)
        
        #--------Genético sin garantía----------
        data = transform_input(w,v)
        
        ga = pyeasyga.GeneticAlgorithm(data, elitism=True)        # initialise the GA with data
        
        
        
        #Para crear la población inicial
        def create_individual(data):

            return [random.randint(0, 1) for _ in range(len(data))]
        
        
        ga.create_individual = create_individual
        
        # define a fitness function
        def fitness(individual, data):
            values, weights = 0, 0
            for selected, box in zip(individual, data):
                if selected:
                    values += box.get('value')
                    weights += box.get('weight')
            if weights > wmax:
                values = 0
            return values
        
        ga.fitness_function = fitness               # set the GA's fitness function
        t0 = time.perf_counter()
        ga.run() 
        t1 = time.perf_counter()
        tiempoGenetico.append(t1 - t0)                                   
        val_genetico.append(ga.best_individual()[0])          
        #------------------------
        
        
        #--------Genético con garantía----------
        data = transform_input(w,v)
        
        ga_g = pyeasyga.GeneticAlgorithm(data, elitism=True)        # initialise the GA with data
        
        
        first_time = True
        def create_individual_guarantee(data):
            w,v = transform_data_to_lists(data)
            global first_time
            if(first_time):
                sol = knapsack.greedy_v2(w,v,wmax)
                first_time = False
                return list(sol[1])
            else:    
                return [random.randint(0, 1) for _ in range(len(data))]
        
        ga_g.create_individual = create_individual_guarantee
        
        # define a fitness function
        def fitness(individual, data):
            values, weights = 0, 0
            for selected, box in zip(individual, data):
                if selected:
                    values += box.get('value')
                    weights += box.get('weight')
            if weights > wmax:
                values = 0
            return values
        
        ga_g.fitness_function = fitness               # set the GA's fitness function
        t0 = time.perf_counter()
        ga_g.run()                                      # run the GA
        t1 = time.perf_counter()
        tiempoGeneticoGarantia.append(t1 - t0)                                   
        val_genetico_garantia.append(ga_g.best_individual()[0])            
        #------------------------
        
        t0 = time.perf_counter()
        val_fptas.append(knapsack.fptas(w,v,wmax,1.5))
        t1 = time.perf_counter()
        tiempoFPTAS.append(t1 - t0)
        
        t0 = time.perf_counter()
        val_pseudopoly.append(knapsack.pseudopolynomial(w,v,wmax))
        optimo = val_pseudopoly[i]
        t1 = time.perf_counter()
        tiempoPseudo.append(t1 - t0)
        
        
        ratio_gr.append(optimo/val_greedy[i])
        ratio_gr2.append(optimo/val_greedy_v2[i])
        if(val_genetico[i]>0):
            ratio_gen.append(optimo/val_genetico[i])
        
        ratio_gen_approx.append(optimo/val_genetico_garantia[i])
        ratio_FPTAS.append(optimo/val_fptas[i])
        ratio_pseudop.append(optimo/val_pseudopoly[i])
        
        
        
        
    #Obtenemos los ratios y tiempos medios
    
    ratio_medio_gr = np.mean(ratio_gr)
    ratio_medio_gr2 = np.mean(ratio_gr2)
    ratio_medio_gen = np.mean(ratio_gen)
    ratio_medio_gen_approx = np.mean(ratio_gen_approx)
    ratio_medio_FPTAS = np.mean(ratio_FPTAS)
    ratio_medio_pseudop = np.mean(ratio_pseudop)
    
    tiempo_medio_gr = np.mean(tiempoG)
    tiempo_medio_gr2 = np.mean(tiempoG_v2)
    tiempo_medio_gen = np.mean(tiempoGenetico)
    tiempo_medio_gen_approx = np.mean(tiempoGeneticoGarantia)
    tiempo_medio_FPTAS = np.mean(tiempoFPTAS)
    tiempo_medio_pseudop = np.mean(tiempoPseudo)
    
    print("Diagrama de cajas para el ratio optimo / valor aproximado")
    data_ratio = pd.DataFrame({"Greedy": ratio_gr, "Greedy \n 2-approx": ratio_gr2, "Genético\n 2-aprox": ratio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_FPTAS, "Pseudop\n (óptimo)": ratio_pseudop})
    
    sns.boxplot(data=data_ratio).set_title("Ratio óptimo/aproximación")
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    print("Ratio medio optimo / valor aproximado")
    data_ratio_medio = pd.DataFrame({"Greedy": ratio_medio_gr, "Greedy \n 2-approx": ratio_medio_gr2,
                                 "Genético\n 2-aprox": ratio_medio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_medio_FPTAS, 
                           "Pseudop\n (óptimo)": ratio_medio_pseudop},index=[0])
    
    
    sns.boxplot(data=data_ratio_medio).set_title("Ratio medio óptimo/aproximación")
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")

    print("Ratio óptimo/approximación en función del número de items")
    p1,= plt.plot(ratio_gr2, '--bo', c='b')
    p2, =plt.plot(ratio_gen_approx, '--bo', c='green')
    p3, =plt.plot(ratio_FPTAS, '--bo', c='red')
    p4, =plt.plot(ratio_pseudop, '--bo', c='darkviolet')
    plt.xticks(range(len(num_items)),num_items)
    plt.xlabel("Número de ítems")
    plt.ylabel("Ratio")
    plt.legend([p1,p2,p3,p4],["Greedy","Genético con garantía",
    "FPTAS", "Pseudopolinómico"])
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")

    print("Tiempo empleado en función del número de items")
    p1,= plt.plot(tiempoG_v2, '--bo', c='b')
    p2, =plt.plot(tiempoGeneticoGarantia, '--bo', c='green')
    p3, =plt.plot(tiempoFPTAS, '--bo', c='red')
    p4, =plt.plot(tiempoPseudo, '--bo', c='darkviolet')
    plt.xticks(range(len(num_items)),num_items)
    plt.xlabel("Número de ítems")
    plt.ylabel("Tiempo (s)")
    plt.legend([p1,p2,p3,p4],["Greedy","Genético con garantía",
    "FPTAS", "Pseudopolinómico"])
    plt.show()




