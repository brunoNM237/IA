import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from slp import SLP

df = pd.read_csv('/home/bruno/Desktop/IA/basesDeDados/breast+cancer+wisconsin+original/wdbc.data', header=None)
#descarta a coluna do paciente, e do diagnóstico
X = df.iloc[:,2:].values
##Benigno e Maligno
y_texto = df.iloc[:, 1].values


#converter em binário 
y = np.where(y_texto == 'M', 1, 0)


#divide partes dos dados em treino e teste, 70 % aprender
X_train, X_teste, y_train, y_teste = train_test_split(X,y, test_size=0.3, random_state=100)


#não explodir sigmoide
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_teste = scaler.transform(X_teste)

#transformar em matriz coluna para dar certo a conta
y_train = y_train.reshape(-1,1)
y_teste = y_teste.reshape(-1,1)

input_size = X_train.shape[1]
output_size = 1


modelo = SLP(input_size=input_size, output_size=output_size)

taxa_de_aprendizado = 0.001
epocas = 1000


modelo.train(X_train, y_train, epocas, taxa_de_aprendizado)


probabilidades = modelo.forward(X_teste)


predicoes = (probabilidades >= 0.5)

matriz = confusion_matrix(y_teste, predicoes)


print(matriz)