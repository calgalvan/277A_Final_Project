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

# dimensionality reduction
from sklearn.manifold import TSNE

# metric tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

# -------------------------- Cluster qualification methods ------------------------
def parent_get_entropy(trueLabels, pred_label_probabilities, num_components):
    '''
    Given a model which uses probailities to assign classes, calculate prediction certainty with entropy
    '''
    cluster_entropies = {}
    cluster_entropies['MAX ENTROPY'] = float(np.log2(num_components)) # !!!

    pred_num_clusters = pred_label_probabilities.shape[1] # get length of matrix of probabilities
    for col in range(pred_num_clusters):
        single_cluster_probabilities = pred_label_probabilities[:, col]
        target_probabilities = []

        for pred_cluster_label in np.unique(trueLabels):
            pred_mask = (trueLabels == pred_cluster_label) # get indices of points in predicted cluster i
            total_single_clust_probs = np.sum(single_cluster_probabilities[pred_mask]) # sum of predicted probabilities for target variable
            target_probabilities.append(total_single_clust_probs) 

        target_probabilities = np.array(target_probabilities) # convert list to numpy array for calculation
        cluster_probabilities = target_probabilities / target_probabilities.sum()

        cluster_entropy = -np.sum(cluster_probabilities * np.log2(cluster_probabilities))
        cluster_entropies[col] = cluster_entropy
        
    return cluster_entropies

def TSNE_visualization(x_data, trueLabels, predLabels):
    # plot to visualize accuracy
    T = TSNE(learning_rate='auto', init='random', perplexity=30, random_state=42)
    X_TSNE = T.fit_transform(x_data) # get points

    # compare clustering
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # true labels
    ax[0].scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=trueLabels, cmap='cividis')
    ax[0].set_xlabel('TSNE component 1')
    ax[0].set_ylabel('TSNE component 2')
    ax[0].set_title('True Clustering')

    # predicted labels
    ax[1].scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=predLabels, cmap='inferno')
    ax[1].set_xlabel('TSNE component 1')
    ax[1].set_title('Predicted Clustering')

    return

# ---------------------------------- REGRESSION  ----------------------------------
class MN_Logistic_Regression_model:
    def __init__(self, 
                 num_classes: int,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame):
        
        my_model = LogisticRegression()

        my_model.fit(x_train, y_train)

        # define class vars for funcs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes

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
        return parent_get_entropy(self.y_test, self.y_test_probabilities, self.num_classes)


# ------------------------------ UNSUPERVISED MODELS ------------------------------
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
        return parent_get_entropy(self.trueLabels, self.pred_label_probabilities, self.num_components)


# ------------------------------ SUPERVISED MODELS ------------------------------
class Naive_Bayes_model:
    def __init__(self, 
                 num_classes: int,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame):
        
        my_model = GaussianNB()

        my_model.fit(x_train, y_train)

        # define class vars for funcs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes

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
        return parent_get_entropy(self.y_test, self.y_test_probabilities, self.num_classes)
