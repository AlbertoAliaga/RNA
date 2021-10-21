# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib as mt
import random

data = pd.read_csv('copia.txt', sep=",", header=None)
data_input = data.drop(columns=8)
desired_output = data.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7])
# print(desired_output)
desired_output.rename({8: 0}, axis=1, inplace=True)


for i in range(0, 8):  # normalización de los inputs
  maxim = data_input[i].max()
  minim = data_input[i].min()
  div = maxim - minim
  for aa in range(0, 1030):  # normalización de cada variable
    data_input[i][aa] = (data_input[i][aa] - minim) / div
    # print(data_input[i][aa])
  # print("Row "+str(i)+ " printed")

# print(data_input)


# randomización y división en porcentajes
entrenamiento = data_input.sample(frac=0.7)
validacion = data_input.drop(entrenamiento.index)
test = validacion.sample(frac=0.5)
validacion = validacion.drop(test.index)


# print(entrenamiento)

entrenamiento.reset_index(inplace=True)
entrenamiento = entrenamiento.drop(columns="index")


# print("entrenamiento[0][0]: ",entrenamiento[0][0])
# print("entrenamiento[1][0]: ",entrenamiento[1][0])
# print("entrenamiento[2][0]: ",entrenamiento[2][0])
# print("entrenamiento[3][0]: ",entrenamiento[3][0])
# print("entrenamiento[4][0]: ",entrenamiento[4][0])
# print("entrenamiento[5][0]: ",entrenamiento[5][0])
# print("entrenamiento[6][0]: ",entrenamiento[6][0])
# print("entrenamiento[7][0]: ",entrenamiento[7][0])

# print(validacion)
# print(test)

a, b, c, d, e, f, g, h, zz = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(
    0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)

peso = {'0': [a], '1': [b], '2': [c], '3': [
    d], '4': [e], '5': [f], '6': [g], '7': [h]}

peso = pd.DataFrame(peso).transpose()
# print("Pesos: ",peso)

#############################################Hiperparámetros#############################################

learning_rate = 0.1
ciclos_max = 1

#############################################Bucle de ejecución#############################################

output = np.dot(entrenamiento, peso)
# print("Suma pesos: ",output[0])
#print("desired_output: ",desired_output[0][0])

output += zz
# print("Umbral: ",zz)
# print("Suma pesos y umbral: ",output[0])

# BUCLE DE CICLOS
ciclos = 0
while ciclos < ciclos_max:
    # BUCLE DE PATRONES
    ii = 0
    #Lista vacía de errores para el MSE: (d - y)^2
    error = [0] * 721
    while ii < len(output):
      jj = 0
      error[ii] = (desired_output[0][ii] - output[ii])**2
      #print("Error ", ii, ": ", error[ii])
      while jj < 9:
        if jj < 8:
          a = 0
          # wj + ∇p * wj
          #print("jj: ",jj)
          #print(peso[0][jj])
          peso[0][jj] += learning_rate * (desired_output[0][ii] - output[ii]) * entrenamiento[jj][ii]
        else:
          # θ + ∇p * θ
          zz += learning_rate * (desired_output[0][ii] - output[ii])
        jj += 1
      ii += 1

    mse = 0
    #Suma todos los valores para el MSE
    for value in error:
        mse += value
    #MSE = 1/N * suma(d - y)
    mse = mse / 721
    print("MSE ciclo 1: ", mse)
    ciclos += 1

#############################################Fin bucle ejecución#############################################

print("Últimos pesos primer ciclo:")
print(peso[0][0])
print(peso[0][1])
print(peso[0][2])
print(peso[0][3])
print(peso[0][4])
print(peso[0][5])
print(peso[0][6])
print(peso[0][7])

print("Umbral primer ciclo: ", zz)
