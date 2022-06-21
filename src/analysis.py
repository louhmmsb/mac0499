import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
from lifelines import KaplanMeierFitter
import umap
import argparse
import seaborn as sns

# TODO: Preciso checar o que a normalização dos dados está fazendo para deixar o TSNE completamente aleatório. Fixando uma semente os resultados são sempre os mesmos, mas isso não parece ser necessário para os dados não normalizados.
# Acredito que isso se deve ao fato de que como estou usando init="pca" como parâmetro do tsne, quando os dados estão todos com a mesma variância, o pca escolhe aleatoriamente quais features vai usar.
# TODO: Acharl um jeito melhor de descobrir quantas componentes manter depois do PCA (link útil: https://towardsdatascience.com/how-to-select-the-best-number-of-principal-components-for-the-dataset-287e64b14c6d)

class Analysis:
    def __init__(self, featuresPath, clinicalPath, numberSep = ",", scaler = preprocessing.StandardScaler()):
        self.df = pd.read_csv(featuresPath, thousands=numberSep)
        self.clinical = pd.read_csv(clinicalPath, thousands=numberSep)
        self.noPacientIndex = self.df.iloc[:, 1:].to_numpy()
        self.noPacientIndexNormalized = (lambda x : scaler.fit(x).transform(x))(self.df.iloc[:, 1:])

    def runPCA(self, nFeatures):
        return PCA(n_components = nFeatures).fit_transform(self.noPacientIndexNormalized)
        
    def runReducer(self, reducer = umap.UMAP(), pca = None):
        data = self.noPacientIndexNormalized
        if pca != None:
            data = pca.fit_transform(self.noPacientIndexNormalized)
        return reducer.fit_transform(data)

    def plotReducedData(self, plotName, reducer = umap.UMAP(), pca = None):
        x, y = zip(*self.runReducer(reducer = reducer, pca = pca))
        plt.scatter(x, y)
        plt.savefig(plotName)
        plt.clf()

    def plotKaplanMeier(self, plotName):
        # deathObserved = self.clinical["vital_status"].iloc[2:].replace(['Dead', 'Alive'], [1, 0])
        # pacientSurvival = ((lambda x, y: np.where(x == '[Not Applicable]', y, x))(self.clinical["death_days_to"].iloc[2:], self.clinical["last_contact_days_to"].iloc[2:])).astype(int)
        pacientSurvival = ((lambda x: x[x != '[Not Applicable]'])(self.clinical["death_days_to"].iloc[2:])).astype(int)
        # deathObserved   = (lambda x, y: y[x != '[Not Applicable]'])(self.clinical["death_days_to"].iloc[2:], self.clinical["vital_status"].iloc[2:].replace(['Dead', 'Alive'], [1, 0]))

        # print(pacientSurvival)
        # print(deathObserved)

        kmf = KaplanMeierFitter()
        # kmf.fit(pacientSurvival, event_observed=deathObserved)
        kmf.fit(pacientSurvival, event_observed=None)

        kmf.plot_survival_function()

        plt.savefig(plotName)
        plt.clf()

    def clusterData(self, reducer = umap.UMAP(), pca = None):
        data = self.runReducer(reducer = reducer, pca = pca)
        x, y = zip(*self.runReducer(reducer = reducer, pca = pca))
        cluster_labels = AffinityPropagation().fit_predict(data)
        df = pd.DataFrame(data=[self.df.iloc[:, 0], data[:, 0], data[:, 1], cluster_labels]).T
        df.columns=['pacient_id', 'x', 'y', 'label']
        cluster_number = len(np.unique(cluster_labels))

        #sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=cluster_labels, palette=sns.color_palette("hls", 2), legend="full")
        sns.scatterplot(x='x', y='y', hue='label', data = df, palette=sns.color_palette("hls", cluster_number), legend="full")
        plt.savefig("plotzin.png")
        return df
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Processes data from TCGA.')
    parser.add_argument('--reducer', '-r', dest='reducer', default='umap')
    parser.add_argument('--plotName', '-p', dest='plotName', default = None)
    parser.add_argument('--variance', '-v', dest='variance', default=0.9, type=float)
    parser.add_argument('--data', '-d', dest='data')


    arguments = parser.parse_args()
    reducer = None
    
    match arguments.reducer:
        case "umap":
            reducer = umap.UMAP(n_neighbors = 5, n_components = 2)
        case "tsne":
            reducer = TSNE(perplexity=2.78*3.14, n_iter=10000, init="random", random_state=None)

    analysis = Analysis(featuresPath = arguments.data + "/features.csv", clinicalPath = arguments.data + "/clinical.csv", scaler = preprocessing.StandardScaler())
    analysis.clusterData(reducer = reducer, pca = PCA(n_components = arguments.variance))
    if arguments.plotName != None:
        analysis.plotReducedData(arguments.plotName + ".png", reducer = reducer, pca = PCA(n_components = arguments.variance))
    # analysis.plot("tsnePCANormalized3.png", tsne = TSNE(perplexity=2.78*3.14, n_iter=10000, init="pca", random_state=1))
    # x = analysis.runTSNE(tsne = TSNE(perplexity=2.78*3.14, n_iter=10000, init="pca")) # TODO: COLOCAR O CLUSTERING E TESTAR O RANDOM COM O INIT DO TSNE
    # analysis.plotKaplanMeier("kaplan.png")
