import random
import pandas as pd

dato = random.uniform(0,1)

print(dato)
print(type(dato))

peso = {'0': [dato]}
print(peso)

pesos = pd.DataFrame(peso)
print(pesos)