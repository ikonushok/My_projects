import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import tqdm, trange
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def TimeSeriesKMeans_cluster_analisys(df, range_n_clusters, params, save_stat=False):

    data = df.to_numpy()
    scaler = RobustScaler()
    data_std = scaler.fit_transform(data)
    print((f'\nСтатистика scaler(X):\n{pd.DataFrame(data_std).describe()}\n'))
    pca = PCA(n_components=data.shape[1])  # 2 для отрисовки на двумерной поверхности
    data_PCA = pca.fit_transform(data_std)
    print((f'Статистика PCA(X):\n{pd.DataFrame(data_PCA).describe()}\n'))
    # print(data_PCA)
    silhouette_avg_list = []
    n_clasters_list = []
    best_cluster_labels = []
    best_silhouette_avg = 0

    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        TimeSeriesKMeans.set_params(params)
        clusterer = TimeSeriesKMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data_std, cluster_labels)

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data_PCA) + (n_clusters + 1) * 10])

        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data_std, cluster_labels)
        unique, counts = np.unique(cluster_labels, return_counts=True)

        if best_silhouette_avg < silhouette_avg:
            best_cluster_labels = cluster_labels
            best_silhouette_avg = silhouette_avg

        print(f"For n_clusters = {n_clusters} The average silhouette_score is :\t"
              f"{silhouette_avg}\tAlgorithm name - 'TimeSeriesKMeans'\t|\tNum objects in groups {counts}")

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            data_PCA[:, 0], data_PCA[:, 1], marker=".", s=150, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker="o", c="white", alpha=1, s=200, edgecolor="k",
                    )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the normalized clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for TimeSeriesKMeans clustering  with : n_clusters = {n_clusters}, "
            f"features = {data.shape[1]}",
            fontsize=14,
            fontweight="bold",
        )
        plt.savefig(f'outputs/Silhouette analysis for TimeSeriesKMeans for {n_clusters} clusters.png')
        plt.show()

        df['Groups'] = cluster_labels
        # plt.figure(figsize=(8, 8))
        # plt.scatter(df['X'], df['Y'], marker=".", s=100, lw=0, alpha=0.7, c=colors, edgecolor="k")
        # plt.title(f'Разбиение обьектов на оптимальное число групп: n_clusters = {n_clusters}')
        # plt.show()

    print(f'Best score: {max(silhouette_avg_list)} '
          f'for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}')

    if save_stat == True:
        return {'pattern_size': data.shape[1], 'Algorithm': 'TimeSeriesKMeans',
                'max_sil_score': max(silhouette_avg_list),
                'n_clusters': n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]
                }

    return best_cluster_labels


def make_transformations(data):
    # add magnitudes for velocity and acceleration
    data['velocity_magnitudes'] = np.linalg.norm(data[data.columns[0:3]], axis=1).transpose()
    data['acceleration_magnitudes'] = np.linalg.norm(data[data.columns[3:6]], axis=1).transpose()

    return data


from numpy import inf
# https://github.com/shunsukeaihara/pydtw
from pydtw import dtw1d, dtw2d
def distance_calculation(data, label):
    # Выбмраем то, что относится к определенному классу
    df_grouped = data.groupby(' label')
    df_group = df_grouped.get_group(label)
    columns = df_group.columns.tolist()
    df_x = df_group[columns[0:-1]]
    df_y = df_group[columns[-1:]]
    df_x = make_transformations(df_x)
    # print(f'\nDataset X:\n{df_x}')
    # print(f'\nDataset Y:\n{pd.unique(df_y[" label"])}')

    # print(f'Проверим, есть ли NaN в dataframe df_x: {df_x.isnull().values.any()}')
    df_log = np.log(df_x.values)
    # print(f'Проверим, есть ли NaN в dataframe df_log: {np.isnan(np.sum(df_log))}')
    df_log = np.nan_to_num(df_log, 0)
    # print(f'Проверим, есть ли NaN в dataframe df_log: {np.isnan(np.sum(df_log))}')
    df_log = np.absolute(df_log)

    # https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/
    df_distance = pd.DataFrame()
    finish = df_log.shape[0] - 1
    for window in trange(20, 101, desc=f'For label {label}'):
        arr_distance = []
        for start in range(0, finish, window):
            end = start + window
            a = df_log[start: end]
            a = a.copy(order='C')  # https://overcoder.net/q/778480/valueerror-ndarray-не-является-c-смежным-в-cython
            b = df_log[end: end + window]
            b = b.copy(order='C')
            # print(a.flags)
            # print(a.shape, b.shape, type(a))
            # print(b.flags)

            if end + window >= finish + 2: break
            cost_matrix, distance, alignmend_a, alignmend_b = dtw2d(a, b)
            arr_distance = np.append(arr_distance, distance)
        arr_distance[arr_distance == inf] = 0
        df_distance[f'{window}'] = pd.DataFrame(arr_distance)

    # Посчитаем среднее расстояние
    columns = df_distance.columns.tolist()
    distances = pd.DataFrame()
    for col in columns:
        df_temp = df_distance[col].replace(-np.inf, np.nan)
        df_temp.replace(np.inf, np.nan, inplace=True)
        df_temp.fillna(df_temp.mean())
        distances[col] = [df_temp.mean()]
    # print(f'\nDistances:\n{df_distances}')

    return distances