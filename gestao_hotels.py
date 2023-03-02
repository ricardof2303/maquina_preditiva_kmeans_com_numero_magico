
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Carregando os dados

dataset = pd.read_csv('la_hotels_data.csv', delimiter=';')

# visualizar as Primeiras Linhas
print(dataset.head(10))


# Dimensões do Dataset em linhas e colunas respectivamente
print(dataset.shape)

# Verifica o tipo dos Campos
print(dataset.dtypes)

# Informações Gerais do Dataset
print(dataset.info())

# Descrições estatísticas básicas gerais do Dataset
print(dataset.describe())

# Remove  a primeira coluna (não é necessária)
hotels = dataset.iloc[0:, 1:]

# visualizar as Primeiras Linhas
print(hotels.head())

# Obtem os valores de cada variável num formato de array
v_hotels = hotels.values

# print(v_hotels)
pca = PCA(n_components=2).fit_transform(v_hotels)

###############################################################


def calcular_melhor_k(pca):
    # Criando um modelo com K = 5
    modelo_v1 = KMeans(n_clusters=5)
    modelo_v1.fit_predict(pca)

    # Silhouette Score
    labels = modelo_v1.labels_
    k5 = silhouette_score(pca, labels, metric='euclidean')

    # Criando um modelo com K = 6
    modelo_v2 = KMeans(n_clusters=6)
    modelo_v2.fit_predict(pca)

    # Silhouette Score
    labels = modelo_v2.labels_
    k6 = silhouette_score(pca, labels, metric='euclidean')

    # Criando um modelo com K = 7
    modelo_v3 = KMeans(n_clusters=7)
    modelo_v3.fit_predict(pca)

    # Silhouette Score
    labels = modelo_v3.labels_
    k7 = silhouette_score(pca, labels, metric='euclidean')

    # Criando um modelo com K = 8
    modelo_v4 = KMeans(n_clusters=8)
    modelo_v4.fit_predict(pca)

    # Silhouette Score
    labels = modelo_v4.labels_
    k8 = silhouette_score(pca, labels, metric='euclidean')

    # Criando um modelo com K = 9
    modelo_v5 = KMeans(n_clusters=9)
    modelo_v5.fit_predict(pca)
    # y_pred = KMeans(n_clusters=5).fit_predict(pca)

    # Silhouette Score
    labels = modelo_v5.labels_
    k9 = silhouette_score(pca, labels, metric='euclidean')

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


modelo = calcular_melhor_k(pca)

# -------------------------------------
# Lista com nomes das colunas
names = ['Nome', 'Pontuacao', 'Limpeza',
         'Conforto', 'Instalacoes', 'Funcionarios', 'Custo-beneficio', 'WiFi_Gratuito', 'Localizacao']

# Incluindo a variável com n° do cluster na base de clientes
cluster_map = pd.DataFrame(dataset, columns=names)
cluster_map['cluster'] = modelo.labels_

print(cluster_map)

# Salvando o novo dataset, agora com com uma variável com o número de cluster

cluster_map.to_csv('hotels_clusters.csv')

# Calcula a média de idade por cluster
print(cluster_map.groupby('cluster')['Pontuacao'].mean())
# print()
print(cluster_map.groupby('cluster')['Idade'].count())
# -----------------------------------------------------------------

clus0 = cluster_map.loc[cluster_map.cluster == 0].cluster
clus1 = cluster_map.loc[cluster_map.cluster == 1].cluster
clus2 = cluster_map.loc[cluster_map.cluster == 2].cluster
clus3 = cluster_map.loc[cluster_map.cluster == 3].cluster
clus4 = cluster_map.loc[cluster_map.cluster == 4].cluster

# -----------------------------------------------------------------
# Calcula a média de outras variáveis (uma por vez) por cluster

print(cluster_map.groupby('cluster')['Pontuacao'].mean())

print(cluster_map.groupby('cluster')['Custo-beneficio'].mean())

print(cluster_map.groupby('cluster')['Conforto'].mean())

# Calcula a média de todas as outras variáveis por cluster
mpc = (cluster_map.groupby('cluster')[['Pontuacao', 'Limpeza', 'Conforto', 'Instalacoes',
                                       'Funcionarios', 'Custo-beneficio', 'WiFi_Gratuito', 'Localizacao']].mean()).round(2)
# print(mpc)

# -------------------------------------------------------------------
# Quantidade de hoteis em cada cluster
print(cluster_map.groupby('cluster')['Nome'].count())
print()
# -------------------------------------------------------------------
# Porcentagem por cluster
g = (cluster_map.groupby('cluster')['Conforto'].count())
s = g.sum()
c0, c1, c2, c3, c4 = clus0.count(), clus1.count(
), clus2.count(), clus3.count(), clus4.count()
pct = [(c0/s), (c1/s), (c2/s), (c3/s), (c4/s)]
print(pct)

# -------------------------------------------------------------------
# Visualizando os grupos

clus = modelo.labels_
plt.scatter(pca[clus == 0, 0], pca[clus ==
            0, 1], s=30, c='red', label='Cluster 0', alpha=0.7)
plt.scatter(pca[clus == 1, 0], pca[clus ==
            1, 1], s=30, c='blue', label='Cluster 1', alpha=0.5)
plt.scatter(pca[clus == 2, 0], pca[clus ==
            2, 1], s=30, c='green', label='Cluster 2', alpha=0.5)
plt.scatter(pca[clus == 3, 0], pca[clus ==
            3, 1], s=30, c='cyan', label='Cluster 3', alpha=0.5)
plt.scatter(pca[clus == 4, 0], pca[clus ==
            4, 1], s=30, c='magenta', label='Cluster 4', alpha=0.5)
plt.scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[
            :, 1], s=50, c='y', edgecolor="black", label='Centroids')
plt.title("Booking hotels | Los Angeles")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim([-4, 8])
plt.ylim([-4, 4])
plt.grid(True)
plt.legend()
plt.show()
