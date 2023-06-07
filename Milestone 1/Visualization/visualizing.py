from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAModelVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def fit_transform(self, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)
        self.dataset_pca = pca.transform(self.dataset)
    
    def plot(self, target, title):
        plt.scatter(self.dataset_pca[:, 0], self.dataset_pca[:, 1], c=target)
        plt.xlabel('First Principal Component')
        plt.ylabel('Average User Rating')
        plt.title(title)
        plt.colorbar()
        plt.show()
