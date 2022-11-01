import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregando os dados
dataset = pd.read_csv('Mall_Customers.csv', delimiter=',')

# visualizar as Primeiras Linhas
# print(dataset.head())

# Dimensões do Dataset em linhas e colunas respectivamente
# print(dataset.shape)

# Verifica o tipo dos Campos
# print(dataset.dtypes)

# Informações Gerais do Dataset
# print(dataset.info())

# Informações Gerais do Dataset
# print(dataset.describe())

# Remove  as três primeiras colunas (vou utilizar apenas as duas últimas)
clientes = dataset.iloc[0:, [3, 4]]

# Obtem os valores de cada variável num formato de array
vclientes = clientes.values

# função para escolher o melhor valor de k entre os valores do número mágico 7+-2


def calcular_melhor_k(vclientes):
    # Criando um modelo com K = 5
    modelo_v1 = KMeans(n_clusters=5)
    modelo_v1.fit_predict(vclientes)

    # Silhouette Score
    labels = modelo_v1.labels_
    k5 = silhouette_score(vclientes, labels, metric='euclidean')

    # Criando um modelo com K = 6
    modelo_v2 = KMeans(n_clusters=6)
    modelo_v2.fit_predict(vclientes)

    # Silhouette Score
    labels = modelo_v2.labels_
    k6 = silhouette_score(vclientes, labels, metric='euclidean')

    # Criando um modelo com K = 7
    modelo_v3 = KMeans(n_clusters=7)
    modelo_v3.fit_predict(vclientes)

    # Silhouette Score
    labels = modelo_v3.labels_
    k7 = silhouette_score(vclientes, labels, metric='euclidean')

    # Criando um modelo com K = 8
    modelo_v4 = KMeans(n_clusters=8)
    modelo_v4.fit_predict(vclientes)

    # Silhouette Score
    labels = modelo_v4.labels_
    k8 = silhouette_score(vclientes, labels, metric='euclidean')

    # Criando um modelo com K = 9
    modelo_v5 = KMeans(n_clusters=9)
    modelo_v5.fit_predict(vclientes)
    # y_pred = KMeans(n_clusters=5).fit_predict(pca)

    # Silhouette Score
    labels = modelo_v5.labels_
    k9 = silhouette_score(vclientes, labels, metric='euclidean')

    if (k5 > k6 and k5 > k7 and k5 > k8 and k5 > k9):
        modelo = modelo_v1
    elif (k6 > k5 and k6 > k7 and k6 > k8 and k6 > k9):
        modelo = modelo_v2
    elif (k7 > k5 and k7 > k6 and k7 > k8 and k7 > k9):
        modelo = modelo_v3
    elif (k8 > k5 and k8 > k6 and k8 > k7 and k8 > k9):
        modelo = modelo_v4
    else:
        modelo = modelo_v5

    return modelo

###########################################################################


modelo = calcular_melhor_k(vclientes)

# Lista com nomes das colunas
names = ['IDCliente', 'Genero', 'Idade',
         'Renda Anual (k$)', 'Pontuacao de Gastos (1-100)']

# Incluindo o n° do cluster na base de clientes
cluster_map = pd.DataFrame(dataset, columns=names)
cluster_map['cluster'] = modelo.labels_

print(cluster_map)

# Calcula a média de idade por cluster
# print(cluster_map.groupby('cluster')['Idade'].mean())
