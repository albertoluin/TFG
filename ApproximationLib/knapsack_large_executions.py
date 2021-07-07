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
"""Datasets: http://artemisa.unicauca.edu.co/~johnyortega/instances_01_KP/"""

#Prepara los ficheros de datos para ser leídos
def eliminar_ultima_linea_datos(path_dir):
    files = os.listdir(path_dir)
    for file in files:
        file_path = os.path.join(path_dir, file)
        a_file = open(file_path, "r")
        lines = a_file.readlines()
        a_file.close()
        del lines[len(lines)-1]
        new_file = open(file_path, "w+")
        for line in lines:
            new_file.write(line)
        new_file.close()

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
    info = []
    for file in files:
        files_names.append(file)
        file_path = os.path.join(path_dir, file)
        print("Leyendo ", file_path)
        
        file_path_opt = os.path.join(optimum_path, file)
        optimal_values.append(leer_fichero(file_path_opt))
        datos.append(np.loadtxt(file_path,skiprows=1))
        info.append(np.loadtxt(file_path,max_rows=1))
    return datos, optimal_values, files_names, info

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

print("\nEjecutamos los algoritmos con el conjunto de datos large-scale")
print("Ejecutando... ")

datos_, optimos_, nombre_archivos_, info = leer_datos("large_scale")

#Ordeno todo según el tamaño de las instancias
info = np.asarray(info)
num_items = info[:,0]

index = sorted(range(len(num_items)), key=lambda k: num_items[k])
    
num_items = np.sort(num_items)

datos = [datos_[i] for i in index]
capacidades_ = info[:,1]
capacidades = [capacidades_[i] for i in index]

nombre_archivos = [nombre_archivos_[i] for i in index]

optimos = [optimos_[i] for i in index]

val_greedy = []
val_greedy_v2 = []
val_genetico = []
val_genetico_garantia = []
val_fptas = []
val_pseudopoly = []

ratio_gr = []
ratio_gr2 = []
ratio_gen_approx = []
ratio_FPTAS = []

tiempoG = []
tiempoG_v2 = []
tiempoGeneticoGarantia = []
tiempoFPTAS = []



columns = ['Número de ítems', 'Capacidad máxima', 'Óptimo', 'Greedy', 'Ratio Greedy', 'Tiempo Greedy',
           'Greedy con garantía', 'Ratio Greedy con garantía','Tiempo Greedy con garantía',
           'Genético con garantía', 'Ratio Genético con garantía', 'Tiempo genético con garantía',
           'PTAS con delta=2','Ratio PTAS','Tiempo PTAS']
dataframe = pd.DataFrame(index=nombre_archivos, columns=columns)

random.seed(0)

for i in range(12):
    datos_actuales = datos[i]
    info_actual = info[i]
    
    n = num_items[i]
    wmax = capacidades[i]
    optimo = optimos[i]
    
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
    

    
    
    ratio_gr.append(optimo/val_greedy[i])
    ratio_gr2.append(optimo/val_greedy_v2[i])
    ratio_gen_approx.append(optimo/val_genetico_garantia[i])
    ratio_FPTAS.append(optimo/val_fptas[i])
    
        
    row = [n, wmax, optimo, val_greedy[i], optimo/val_greedy[i], tiempoG[i],
    val_greedy_v2[i], optimo/val_greedy_v2[i], tiempoG_v2[i],
    val_genetico_garantia[i], optimo/val_genetico_garantia[i], tiempoGeneticoGarantia[i],
    val_fptas[i], optimo/val_fptas[i], tiempoFPTAS[i]]
    
    dataframe.iloc[i,:]=row
    
    
dataframe.to_excel("knapsack_large_executions.xlsx")


    
    
#Obtenemos los ratios y tiempos medios

ratio_medio_gr = np.mean(ratio_gr)
ratio_medio_gr2 = np.mean(ratio_gr2)
ratio_medio_gen_approx = np.mean(ratio_gen_approx)
ratio_medio_FPTAS = np.mean(ratio_FPTAS)

tiempo_medio_gr = np.mean(tiempoG)
tiempo_medio_gr2 = np.mean(tiempoG_v2)
tiempo_medio_gen_approx = np.mean(tiempoGeneticoGarantia)
tiempo_medio_FPTAS = np.mean(tiempoFPTAS)



     
print("Diagrama de cajas para el ratio optimo / valor aproximado")
data_ratio = pd.DataFrame({"Greedy": ratio_gr, "Greedy \n 2-approx": ratio_gr2,
                           "Genético\n 2-aprox": ratio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_FPTAS})


sns.boxplot(data=data_ratio).set_title("Ratio óptimo/aproximación")
plt.show()

##input("\n--- Pulsar tecla para continuar ---\n")

print("Ratio medio optimo / valor aproximado")
data_ratio_medio = pd.DataFrame({"Greedy": ratio_medio_gr, "Greedy \n 2-approx": ratio_medio_gr2,
                        "Genético\n 2-aprox": ratio_medio_gen_approx,
                           "FPTAS con\n delta=1.5": ratio_medio_FPTAS}
                                ,index=[0])


sns.boxplot(data=data_ratio_medio).set_title("Ratio medio óptimo/aproximación")
plt.show()

##input("\n--- Pulsar tecla para continuar ---\n")

print("Tiempo medio (en segundos)")

data_tiempo_medio = pd.DataFrame({"Greedy": tiempo_medio_gr, "Greedy \n 2-approx": tiempo_medio_gr2,
                                  "Genético\n 2-aprox": tiempo_medio_gen_approx,
                           "FPTAS con\n delta=1.5": tiempo_medio_FPTAS}
                                 ,index=[0])


sns.boxplot(data=data_tiempo_medio, color=".25").set_title("Tiempo medio (en segundos)")
plt.show()

print("Ratio óptimo/approximación en función del número de items")
p1,= plt.plot(ratio_gr2, '--bo', c='b')
p2, =plt.plot(ratio_gen_approx, '--bo', c='green')
p3, =plt.plot(ratio_FPTAS, '--bo', c='red')
plt.xticks(range(len(num_items)), num_items)
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Número de ítems")
plt.ylabel("Ratio")
plt.legend([p1,p2,p3],["Greedy","Genético con garantía",
"FPTAS"])
plt.show()
    
#input("\n--- Pulsar tecla para continuar ---\n")

print("Tiempo empleado en función del número de items")
p1,= plt.plot(tiempoG_v2, '--bo', c='b')
p2, =plt.plot(tiempoGeneticoGarantia, '--bo', c='green')
p3, =plt.plot(tiempoFPTAS, '--bo', c='red')
plt.xticks(range(len(num_items)),num_items)
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Número de ítems")
plt.ylabel("Tiempo (s)")
plt.legend([p1,p2,p3],["Greedy","Genético con garantía",
"FPTAS"])
plt.show()

