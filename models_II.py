# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data pre-proccessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# classification models
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import *
import leidenalg

# dimensionality reduction
from sklearn.manifold import TSNE

# metric tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import networkx as nx 
from sklearn.neighbors import NearestNeighbors
import igraph as ig 
import umap.umap_ as umap

# -------------------------- Cluster qualification methods ------------------------
# def parent_get_entropy(trueLabels, pred_label_probabilities, num_components):
#     '''
#     Given a model which uses probabilities to assign classes, calculate prediction certainty with entropy
#     How true labels are distributed within a predicted class
#     '''
#     cluster_entropies = {}
#     cluster_entropies['MAX ENTROPY'] = float(np.log2(num_components)) # define max entropy for comparison

#     pred_num_clusters = pred_label_probabilities.shape[1] # get length of matrix of probabilities
#     # for each predicted class aka col
#     for col in range(pred_num_clusters):

#         # get probabilities for single class across all samples
#         single_cluster_probabilities = pred_label_probabilities[:, col]

#         # store probability per true label
#         target_probabilities = []

#         # for each true class label
#         for true_class_label in np.unique(trueLabels):
#             # mask to find samples where true == pred
#             pred_mask = (trueLabels == true_class_label)

#             # sum of predicted probabilities for class
#             total_single_clust_probs = np.sum(single_cluster_probabilities[pred_mask]) # sum of predicted probabilities for target variable
#             target_probabilities.append(total_single_clust_probs) 

#         target_probabilities = np.array(target_probabilities) # convert list to numpy array for calculation
#         cluster_probabilities = target_probabilities / target_probabilities.sum()

#         cluster_entropy = float(-np.sum(cluster_probabilities * np.log2(cluster_probabilities)))
#         cluster_entropies[col] = f'{cluster_entropy:.2f}'
        
#     return cluster_entropies

def TSNE_visualization(x_data, trueLabels, predLabels):
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    trueLabels_enc = le1.fit_transform(trueLabels)
    predLabels_enc = le2.fit_transform(predLabels)

    # plot to visualize accuracy
    T = TSNE(learning_rate='auto', init='random', perplexity=30, random_state=42)
    X_TSNE = T.fit_transform(x_data) # get points

    # compare clustering
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # true labels
    ax[0].scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=trueLabels_enc, cmap='cividis')
    ax[0].set_xlabel('TSNE component 1')
    ax[0].set_ylabel('TSNE component 2')
    ax[0].set_title('True Clustering')

    # predicted labels
    ax[1].scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=predLabels_enc, cmap='inferno')
    ax[1].set_xlabel('TSNE component 1')
    ax[1].set_title('Predicted Clustering')

    plt.show()

    return

# ---------------------------------- REGRESSION  ----------------------------------
class MN_Logistic_Regression_model:
    def __init__(self, 
                 num_classes: int,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 balance_classes: bool):
        
        self.y_train = np.ravel(y_train)
        self.y_test = np.ravel(y_test)
        self.x_train = x_train
        self.x_test = x_test
        self.num_classes = num_classes

        if balance_classes == True:
            my_model = LogisticRegression(class_weight='balanced')

        else: 
            my_model = LogisticRegression()

        my_model.fit(self.x_train, self.y_train)

        # training predictions
        self.y_train_predLabels = my_model.predict(x_train)

        # testing predictions
        self.y_test_probabilities = my_model.predict_proba(x_test)
        self.y_test_predLabels = my_model.predict(x_test)

    def get_accuracies(self):
        test_acc = accuracy_score(self.y_test, self.y_test_predLabels) * 100
        train_acc = accuracy_score(self.y_train, self.y_train_predLabels) * 100
        print(f"training accuracy: {train_acc:.2f}%")
        print(f"testing accuracy: {test_acc:.2f}%")
        return
    
    def visualize_clustering(self):
        return TSNE_visualization(self.x_test, self.y_test, self.y_test_predLabels)
    
    # def get_entropy(self):
    #     return parent_get_entropy(self.y_test, self.y_test_probabilities, self.num_classes)


# ------------------------------ UNSUPERVISED MODELS ------------------------------
class Leiden_clustering:
    def __init__(self, 
                 k: int,
                 x_data: pd.DataFrame,
                 y_data: pd.DataFrame,
                 resolution=1.0
                 ):
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(x_data)
        distances, indices = nbrs.kneighbors(x_data)

        edges = [(i, j) for i, neigh in enumerate(indices) for j in neigh[1:]]

        My_G = nx.Graph()
        My_G.add_nodes_from(y_data)
        My_G.add_edges_from(edges)

        G = ig.Graph(edges = edges, directed = False)
        self.G = G.simplify()

        pos = nx.spectral_layout(My_G)

        self.x_data = x_data
        self.y_data = y_data

        # for single UMAP
        partition = leidenalg.find_partition(self.G, leidenalg.RBConfigurationVertexPartition, resolution_parameter = resolution)
        self.labels = np.array(partition.membership)

    def get_UMAP(self):
        X_UMAP = umap.UMAP().fit_transform(self.x_data)

        for i in range(np.max(self.labels)+ 1):
            idx = np.argwhere(self.labels == i)
            plt.scatter(X_UMAP[idx, 0], X_UMAP[idx, 1], marker = '.', alpha = 0.3, label = str(i))

        plt.gca().set_aspect('equal', 'datalim')
        plt.title("Leiden Clustering on Digits Dataset (UMAP)", fontsize = 12)
        plt.legend()
        plt.show()

        return
    
    def run_UMAP_resolutions(self, parameters: list):
        plt.figure(figsize=(11, 8))
        fig, ax = plt.subplots(1, len(parameters), figsize=(12, 3))

        for a, p in enumerate(parameters):
            partition = leidenalg.find_partition(self.G, leidenalg.RBConfigurationVertexPartition, resolution_parameter = p)

            labels = np.array(partition.membership)

            X_UMAP = umap.UMAP().fit_transform(self.x_data)

            for i in range(np.max(labels)+ 1):
                idx = np.argwhere(labels == i)
                ax[a].scatter(X_UMAP[idx, 0], X_UMAP[idx, 1], marker = '.', alpha = 0.3, label = str(i))

                ax[a].set_aspect('equal', 'datalim')
                ax[a].set_title(f'resolution={p}', fontsize = 12)
                ax[a].legend()

        plt.show()
    # def get


# ------------------------------ SUPERVISED MODELS ------------------------------
class Naive_Bayes_model:
    def __init__(self, 
                 num_classes: int,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 update_priors: bool):
        
        self.y_train = np.ravel(y_train)
        self.y_test = np.ravel(y_test)
        self.x_train = x_train
        self.x_test = x_test
        self.num_classes = num_classes
    
        if update_priors == True:
            my_model = GaussianNB(priors=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        else: 
            my_model = GaussianNB()

        my_model.fit(self.x_train, self.y_train)

        # training predictions
        self.y_train_predLabels = my_model.predict(x_train)

        # testing predictions
        self.y_test_probabilities = my_model.predict_proba(x_test)
        self.y_test_predLabels = my_model.predict(x_test)

    def get_accuracies(self):
        test_acc = accuracy_score(self.y_test, self.y_test_predLabels) * 100
        train_acc = accuracy_score(self.y_train, self.y_train_predLabels) * 100
        print(f"training accuracy: {train_acc:.2f}%")
        print(f"testing accuracy: {test_acc:.2f}%")
        return
    
    def visualize_clustering(self):
        return TSNE_visualization(self.x_test, self.y_test, self.y_test_predLabels)
    
    # def get_entropy(self):
    #     return parent_get_entropy(self.y_test, self.y_test_probabilities, self.num_classes)
