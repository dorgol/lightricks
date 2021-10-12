import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from get_data import df_cluster


class ClusterModel:
    def __init__(self, df: pd.DataFrame, model_type, **kwargs):
        self.df = df
        self.model = model_type(**kwargs)

    def fit_model(self):
        return self.model.fit(self.df)

    @staticmethod
    def obs_per_cluster(pred: np.ndarray):
        """
        count the number of observation in each cluster
        :param pred: prediction made by clustering model, have to be same length as data
        :return: dictionary of classes and numbers of observations in each class
        """
        unique, counts = np.unique(pred, return_counts=True)
        return dict(zip(unique, counts))

    def get_metrics(self, pred: np.ndarray, metric: sklearn.metrics = metrics.silhouette_score):
        """
        Return a metric of prediction for specific model.
        :param pred: prediction made by clustering algorithm, have to be same length as data
        :param metric: metric from sklearn.metrics module. Default to silhouette_score
        :return: chosen metric per prediction
        """
        return metric(self.df, pred)

    def get_elbow_plot(self, model, range_clusters):
        """
        Prints plot of number of clusters against the inner sum of squares (inertia)
        :param model: Fitted clusetring model
        :param range_clusters: range of different clustering choices.
        :return: plotly figure
        """
        sse = {}
        for k in range(1, range_clusters):
            model_to_run = model(n_clusters=k, max_iter=1000).fit(self.df)
            self.df["clusters"] = model_to_run.labels_
            sse[k] = model_to_run.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

        fig = px.line(x=list(sse.values()), y=list(sse.keys()))
        return fig

    def visualize_results(self, pred):
        """
        plotting the clustered data after performing PCA; colored by predicted class.
        :param pred: prediction made by clustering algorithm, have to be same length as data
        :return: see description above
        """
        pca = PCA(n_components=2)
        pca_df = pca.fit_transform(self.df)
        x_reduceddf = pd.DataFrame(pca_df, index=self.df.index, columns=['PC1', 'PC2'])
        x_reduceddf['cluster'] = pred
        fig = px.scatter(x_reduceddf, x='PC1', y='PC2', color='cluster')
        return fig

    @staticmethod
    def parallel_centroids(fitted_model, regex_names: str):
        """
        Plots centroid lines - every centroid is a line and every value is a feature from the data
        :param fitted_model: fitter clustering model
        :param regex_names: the regex allows to choose columns from the data to foucs on.
        :return: plotly plot
        """
        data = pd.DataFrame(fitted_model.cluster_centers_, columns=fitted_model.feature_names_in_)
        fig = px.parallel_coordinates(data.filter(regex=regex_names),
                                      color_continuous_scale=px.colors.diverging.Tealrose,
                                      color_continuous_midpoint=2)
        return fig


def running_kmeans():
    """
    function for instantiation of the ClusterModel class
    :return: instance model, elbow plot, fitted model, prediction from the fitted model
    """
    clus = ClusterModel(df_cluster.iloc[:, 3:], KMeans, n_clusters=6)
    elbow = clus.get_elbow_plot(KMeans, 10)
    fitted = clus.fit_model()
    pred = fitted.labels_
    return clus, elbow, fitted, pred


def vis_kmeans(clus, pred):
    """
    see running_kmeans documentation for details of arguments
    :param clus:
    :param pred:
    :return: plotly plot
    """
    return clus.visualize_results(pred)


def plot_result_kmeans(clus, fitted, regex):
    """
    see running_kmeans documentation for details of arguments
    :param clus:
    :param fitted:
    :param regex:
    :return: plotly plot
    """
    return clus.parallel_centroids(fitted, regex)
