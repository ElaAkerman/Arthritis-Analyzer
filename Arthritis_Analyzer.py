import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from numpy import shape
from sklearn.decomposition import PCA


def CreateHeatmapData():
    df = pd.read_csv("arthritis rma.csv")
    df.set_index('#NAMES', inplace=True)
    df.columns.names = ["Subjects"]
    df.index.names = ["Genes"]

    return df


def CreateHeatmap(i_heatmapData, i_heatmapName):
    # Creating the heatmap window.
    plt.figure(i_heatmapName, figsize=(10, 10))
    plt.title(i_heatmapName, color='#CD3333', size=16, fontstyle='oblique', fontweight='bold')

    # Set the xaxis and yaxis title size.
    plt.gca().xaxis.label.set_size(14)
    plt.gca().yaxis.label.set_size(14)

    heatmap = sns.heatmap(
        data=i_heatmapData,
        cmap='OrRd',
    )

    # Arange the xticks and yticks according to the data-frame labels.
    plt.yticks(np.arange(i_heatmapData.shape[0]), i_heatmapData.index)
    plt.xticks(np.arange(i_heatmapData.shape[1]), i_heatmapData.columns)

    # Resize the x and y ticks sizes.
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=7)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=7)

    plt.show()


def ClusterDataFrame(i_dataFrame):
    retDF = i_dataFrame.copy()
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(i_dataFrame)

    # Adding the cluster.lables_ to a new column in heatmap's data:
    retDF['cluster'] = cluster.labels_  # Adding new cluster column
    retDF = retDF.sort_values(by=['cluster'],
                              ascending=False)  # Sorting - all with cluster=1 at top of the df
    retDF = retDF.drop(columns=['cluster'])  # Deleting the cluster column

    return retDF


def clusteringData(i_heatmapData):
    # Genes Cluster:
    clusteredHMDataByGenes = ClusterDataFrame(i_heatmapData)
    # CreateHeatmap(clusteredHMDataByGenes, "Genes Clustering")  # Creating new heatmap with the current clustered df

    # Subjects cluster:
    heatmapDataTransposed = clusteredHMDataByGenes.transpose()
    clusteredHMDataBySubjects = ClusterDataFrame(heatmapDataTransposed)
    clusteredHMDataBySubjects = clusteredHMDataBySubjects.transpose()
    CreateHeatmap(clusteredHMDataBySubjects,
                  "Arthritis Analyzer Clustering")  # Creating new heatmap with the current clustered df

def normalization():
    df = pd.read_csv("arthritis rma.csv")
    df.set_index('#NAMES', inplace=True)
    df.columns.names = ["Subjects"]
    df.index.names = ["Genes"]
    df_normalized = df.sub(df.mean(1),axis=0).div(df.std(1),axis=0)
    
    return df_normalized

def ClusterDataFrame_Normalized(i_dataFrame):
    retDF = i_dataFrame.copy()
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(i_dataFrame)

    # Adding the cluster.lables_ to a new column in heatmap's data:
    retDF['cluster'] = cluster.labels_  # Adding new cluster column
    retDF = retDF.sort_values(by=['cluster'],
                              ascending=False)  # Sorting - all with cluster=1 at top of the df
    retDF = retDF.drop(columns=['cluster'])  # Deleting the cluster column

    return retDF

def clusteringData_Normalized(i_heatmapData):
    # Genes Cluster:
    clusteredHMDataByGenes = ClusterDataFrame_Normalized(i_heatmapData)
    # CreateHeatmap(clusteredHMDataByGenes, "Genes Clustering")  # Creating new heatmap with the current clustered df

    # Subjects cluster:
    i_heatmapDataTransposed = clusteredHMDataByGenes.transpose()
    clusteredHMDataBySubjects = ClusterDataFrame_Normalized(i_heatmapDataTransposed)
    clusteredHMDataBySubjects = clusteredHMDataBySubjects.transpose()
    CreateHeatmap(clusteredHMDataBySubjects,
                  "Arthritis Analyzer Clustering Normalized")  # Creating new heatmap with the current clustered df
    return clusteredHMDataBySubjects

def Dimensionality_reduction (i_clusteredHMDataBySubjects):
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(i_clusteredHMDataBySubjects.to_numpy().T)
    X_principal = pd.DataFrame(X_principal)
    ac = AgglomerativeClustering(n_clusters = 2)

    plt.figure(figsize = (6,6))
    scatter = plt.scatter(X_principal[0], X_principal[1],c=ac.fit_predict(X_principal), cmap = 'bwr')
    plt.title('Scatter Plot', color='#CD3333', size=16, fontstyle='oblique', fontweight='bold')
    tabs = ['Sick', 'Healthy']
    plt.legend(handles=scatter.legend_elements()[0], labels=tabs)
    plt.show()

def main():
    df = CreateHeatmapData()
    CreateHeatmap(df, 'Arthritis Analyzer')
    clusteringData(df)
    df_normalized = normalization()
    clusteredHMDataBySubjects = clusteringData_Normalized(df_normalized)
    Dimensionality_reduction(clusteredHMDataBySubjects)



if __name__ == "__main__":
    main()
