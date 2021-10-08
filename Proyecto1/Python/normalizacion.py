import pandas as pd
import numpy as np
import matplotlib as mt
import random

data = pd.read_csv('copia.txt', sep=",", header=None)
data_input = data.drop(columns=8)


for i in range(0,8):      #normalización de los inputs
  maxim=data_input[i].max()
  minim=data_input[i].min()
  div = maxim - minim
  for aa in range(0,1030):   #normalización de cada variable
    data_input[i][aa] = (data_input[i][aa] - minim) / div
    #print(data_input[i][aa])
  #print("Row "+str(i)+ " printed")

#print(data_input)


entrenamiento=data_input.sample(frac=0.7)          #randomización y división en porcentajes
validacion=data_input.drop(entrenamiento.index)
test=validacion.sample(frac=0.5)
validacion=validacion.drop(test.index)


#print(entrenamiento)

entrenamiento.reset_index(inplace=True)
entrenamiento = entrenamiento.drop(columns="index")


#print("entrenamiento[0][0]: ",entrenamiento[0][0])
#print("entrenamiento[1][0]: ",entrenamiento[1][0])
#print("entrenamiento[2][0]: ",entrenamiento[2][0])
#print("entrenamiento[3][0]: ",entrenamiento[3][0])
#print("entrenamiento[4][0]: ",entrenamiento[4][0])
#print("entrenamiento[5][0]: ",entrenamiento[5][0])
#print("entrenamiento[6][0]: ",entrenamiento[6][0])
#print("entrenamiento[7][0]: ",entrenamiento[7][0])

#print(validacion)
#print(test)

a, b, c, d, e, f, g, h, zz = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)

peso = {'0':[a], '1': [b], '2': [c], '3': [d], '4': [e],'5': [f],'6': [g],'7': [h]}

peso = pd.DataFrame(peso).transpose()
#print("Pesos: ",peso)

#############################################Hiperparámetros#############################################

learning_rate = 0.1
ciclos = 10

#############################################Bucle de ejecución#############################################

dot = np.dot(entrenamiento,peso)
#print("Suma pesos: ",dot[0])

dot += zz
#print("Umbral: ",zz)
#print("Suma pesos y umbral: ",dot[0])
