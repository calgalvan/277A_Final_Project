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
def parent_get_entropy():
    # !! TO DO !!
    return

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

    def get_entropy(self):
        #Obtain the number of classes for normalization
        n_classes = self.pred_label_probabilities.shape[1]

        #Ensure that 0 log2 0 = 0 instead of -inf
        adjustedProbs = np.clip(self.pred_label_probabilities, 1e-12, 1)

        #Calculate Shannon's entropy based on equation for each data point and clusters
        entropy = -np.sum(adjustedProbs * np.log(adjustedProbs), axis = 1)

        #Determine average entropy and normalize for comparison to other models
        return np.mean(entropy) / np.log(n_classes)

    def plot_entropy(self):
        N, Nclass = self.y_test_probabilities.shape
        class_labels = np.unique(self.y_test)

        fig, ax = plt.subplots(Nclass, 1, sharex = True)
        fig.set_figheight(Nclass + 1)
        fig.subplots_adjust(hspace = 1)
        fig.suptitle('Cross Entropy Plot for Multinomial Logistic Regression')
        fig.tight_layout(rect=[0, 0, 1, 1])

        for i, l in enumerate(class_labels):
            idx      = np.array([j for j, t in enumerate(self.y_test) if t == l])
            pclass   = self.y_test_probabilities[idx,i]

            #Count number of each label and calculate weight
            (value, where) = np.histogram(pclass, bins = np.arange(0, 1, 0.01), density = True)
            w = 0.5*(where[1:] + where[:-1])

            ax[i].plot(w, value, 'k-')
            ax[i].set_ylabel('Frequency')
            ax[i].set_title(l)

        ax[Nclass - 1].set_xlabel('Probability')
        plt.show()


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

        G = ig.Graph(n=len(x_data), edges = edges, directed = False)
        self.G = G.simplify()

        pos = nx.spectral_layout(My_G)

        self.x_data = x_data
        self.y_data = y_data

        # for single UMAP
        partition = leidenalg.find_partition(self.G, leidenalg.RBConfigurationVertexPartition, resolution_parameter = resolution)
        self.labels = np.array(partition.membership)

    def get_UMAP(self):
        X_UMAP = umap.UMAP().fit_transform(self.x_data)
        labels = [self.y_data, self.labels]
        titles = ['True labels', 'Predicted labels']

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        for plt_i in range(2):
            for i in range(np.max(labels[plt_i])+ 1):
                idx = np.argwhere(labels[plt_i] == i)
                ax[plt_i].scatter(X_UMAP[idx, 0], X_UMAP[idx, 1], marker = '.', alpha = 0.3, label = str(i))

            ax[plt_i].set_aspect('equal', 'datalim')
            ax[plt_i].set_title(titles[plt_i])
            ax[plt_i].legend()

        plt.tight_layout
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

    def get_entropy(self):
        entropy_total = 0
        n_samples = len(self.y_data)
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            #Determine the indexes for each cluster in pred labels and find corresponding true labels at index
            idx = np.where(self.labels == label)[0]
            labels = self.y_data["sfdm2"].iloc[idx]

            #Count number of each label
            labels_counts = np.bincount(labels)

            #Convert to probabilities
            prob = labels_counts / labels_counts.sum()

            #Ensure that 0 log2 0 = 0 instead of -inf
            prob = np.clip(prob, 1e-12, 1)

            #Shannon's entropy to calculate entropy for this cluster
            entropy = -np.sum(prob * np.log(prob))

            #Calculate weight of this cluster label
            w = len(labels_counts) / n_samples

            #Add to total entropy
            entropy_total += entropy * w

        #Normalize entropy for comparison
        return entropy_total / np.log(len(unique_labels))


class Gaussian_Mixture_model:
    def __init__(self,
                num_components: int,
                rand_state: int,
                x_data: pd.DataFrame,
                y_data: pd.DataFrame,
                target: str):

        self.num_components = num_components
        self.x_data = x_data
        self.y_data = y_data

        my_model = GaussianMixture(n_components = num_components, random_state = rand_state).fit(x_data)

        # extract centers and pred labels
        self.center = my_model.means_
        self.pred_label_probabilities = my_model.predict_proba(x_data)
        self.predLabels = my_model.predict(x_data)
        self.trueLabels = y_data[target].values

    def visualize_clustering(self):
        return TSNE_visualization(self.x_data, self.trueLabels, self.predLabels)

    def get_entropy(self):
        #Obtain the number of classes for normalization
        n_classes = self.pred_label_probabilities.shape[1]

        #Ensure that 0 log2 0 = 0 instead of -inf
        adjustedProbs = np.clip(self.pred_label_probabilities, 1e-12, 1)

        #Calculate Shannon's entropy based on equation for each data point and clusters
        entropy = -np.sum(adjustedProbs * np.log(adjustedProbs), axis = 1)

        #Determine average entropy and normalize for comparison to other models
        return np.mean(entropy) / np.log(n_classes)


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
            p = 1 / num_classes
            priors_list = [p] * num_classes
            my_model = GaussianNB(priors=priors_list)

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

    def get_entropy(self):
        #Obtain the number of classes for normalization
        n_classes = self.y_test_probabilities.shape[1]

        #Ensure that 0 log2 0 = 0 instead of -inf
        adjustedProbs = np.clip(self.y_test_probabilities, 1e-12, 1)

        #Calculate Shannon's entropy based on equation for each data point and clusters
        entropy = -np.sum(adjustedProbs * np.log(adjustedProbs), axis = 1)

        #Determine average entropy and normalize for comparison to other models
        return np.mean(entropy) / np.log(n_classes)



    def plot_entropy(self):

        N, Nclass = self.y_test_probabilities.shape
        class_labels = np.unique(self.y_test)

        fig, ax = plt.subplots(Nclass, 1, sharex = True)
        fig.set_figheight(Nclass + 1)
        fig.subplots_adjust(hspace = 1)
        fig.suptitle('Cross Entropy Plot for Naive Bayes')
        fig.tight_layout(rect=[0, 0, 1, 1])

        for i, l in enumerate(class_labels):
            idx      = np.array([j for j, t in enumerate(self.y_test) if t == l])
            pclass   = self.y_test_probabilities[idx,i]

            #Count number of each label and calculate weight
            (value, where) = np.histogram(pclass,\
                                        bins = np.arange(0,1,0.01),\
                                        density = True)
            w = 0.5*(where[1:] + where[:-1])

            ax[i].plot(w, value, 'k-')
            ax[i].set_ylabel('frequency')
            ax[i].set_title(class_labels[i])

        ax[Nclass-1].set_xlabel('probability')
        plt.show()