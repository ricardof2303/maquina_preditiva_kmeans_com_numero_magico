import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score


# Carregando os dados
dataset = pd.read_csv('heart.csv', delimiter=',')

# visualizar as Primeiras Linhas
print(dataset.head())

# Dimensões do Dataset em linhas e colunas respectivamente
print(dataset.shape)

# Verifica o tipo dos Campos
print(dataset.dtypes)

# Informações Gerais do Dataset
print(dataset.info())


# Obtem os valores de cada variável num formato de array
dataset = dataset.values

# Aplica redução de dimensionalidade no array das variáveis
pca = PCA(n_components=2).fit_transform(dataset)

# Determinando um range do Hyperparâmetro  "K"  do Kmeans
k_range = range(5, 10)

# Aplicando o modelo K-Means para cada valor de K (esta célula pode levar bastante tempo para ser executada)
k_means_var = [KMeans(n_clusters=k).fit(pca) for k in k_range]



# Criando um modelo com K = 5
modelo_v1 = KMeans(n_clusters = 5)
modelo_v1.fit(pca)
 
# Silhouette Score
labels = modelo_v1.labels_
k5 = silhouette_score(pca, labels, metric = 'euclidean')



# Criando um modelo com K = 6
modelo_v2 = KMeans(n_clusters=6)
modelo_v2.fit(pca)

# Silhouette Score
labels = modelo_v2.labels_
k6 = silhouette_score(pca, labels, metric='euclidean')


# Criando um modelo com K = 7
modelo_v3 = KMeans(n_clusters = 7)
modelo_v3.fit(pca)

# Silhouette Score
labels = modelo_v3.labels_
k7 = silhouette_score(pca, labels, metric = 'euclidean')

# Criando um modelo com K = 8
modelo_v4 = KMeans(n_clusters = 8)
modelo_v4.fit(pca)

# Silhouette Score
labels = modelo_v4.labels_
k8 = silhouette_score(pca, labels, metric = 'euclidean')

# Criando um modelo com K = 9
modelo_v5 = KMeans(n_clusters = 9)
modelo_v5.fit(pca)

# Silhouette Score
labels = modelo_v5.labels_
k9 = silhouette_score(pca, labels, metric = 'euclidean')

if (k5 > k6 and k5 > k7 and k5 > k8 and k5 > k9):
    print("O modelo_v1 é o melhor. score:", k5)
elif (k6 > k5 and k6 > k7 and k6 > k8 and k6 > k9):
    print("O modelo_v2 é o melhor. score:", k6)
elif (k7 > k5 and k7 > k6 and k7 > k8 and k7 > k9):
    print("O modelo_v3 é o melhor. score:", k7)
elif (k8 > k5 and k8 > k6 and k8 > k7 and k8 > k9):
    print("O modelo_v4 é o melhor. score:", k8)
else:
    print("O modelo_v5 é o melhor. score:", k9)

print("k5",k5)
print("k6",k6)
print("k7",k7)
print("k8",k8)
print("k9",k9)


# Lista com nomes das colunas
names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
         'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']

# Incluindo o n° do cluster na base de clientes
cluster_map = pd.DataFrame(dataset, columns=names)
cluster_map['cluster'] = modelo_v2.labels_

print(cluster_map)


# Calcula a média de idade por cluster
print(cluster_map.groupby('cluster')['age'].mean())
