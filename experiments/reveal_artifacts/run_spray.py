import os
from distutils.util import strtobool

import click
import h5py
import numpy as np
import torch
import yaml
from corelay.base import Param
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.affinity import SparseKNN
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from corelay.processor.distance import SciPyPDist
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.flow import Sequential, Parallel
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

"""
This python script is based on the examples from https://github.com/virelay/corelay.
"""


class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
    def function(self, data):
        return data.sum(1)


class Absolute(Processor):
    def function(self, data):
        return np.absolute(data)


class Normalize(Processor):
    # axes = Param(tuple, (1, 2))
    axes = Param(tuple, (1,))

    def function(self, data):
        print(data.shape)
        data = data / data.sum(self.axes, keepdims=True)
        return data


def csints(string):
    return tuple(int(elem) for elem in string.split(','))


class Histogram(Processor):
    bins = Param(int, 256)

    def function(self, data):
        hists = np.stack([
            np.stack([
                np.histogram(
                    arr.reshape(arr.shape[0], np.prod(arr.shape[1:3])),
                    bins=self.bins,
                    density=True
                ) for arr in channel
            ]) for channel in data.transpose(3, 0, 1, 2)])
        return hists


class PCC(Processor):
    def function(self, data):
        return squareform(pdist(data, metric=lambda x, y: pearsonr(x, y)[0]))


class SSIM(Processor):
    def function(self, data):
        N, H, W = data.shape
        return squareform(pdist(
            data.reshape(N, H * W),
            metric=lambda x, y: structural_similarity(x.reshape(H, W), y.reshape(H, W), multichannel=False)
        ))


VARIANTS = {
    'absspectral': {
        'preprocessing': Sequential([
            Absolute(),
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'spectral': {
        'preprocessing': Sequential([
            # SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'fullspectral': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'histogram': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Histogram(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'ssim': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
        ]),
        'distance': SSIM(),
    },
    'pcc': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': PCC(),
    }
}


def str2bool(s):
    if isinstance(s, str):
        return strtobool(s)
    return bool(s)


@click.command()
@click.option('--variant', type=click.Choice(list(VARIANTS)), default='spectral')
@click.option('--analysis-file', type=click.Path(),
              default="results/spray/vgg16_Vanilla_features28.hdf5")
@click.option('--class-indices', type=csints, default="0,1")
@click.option('--n-eigval', type=int, default=32)
@click.option('--n-clusters', type=csints, default=','.join(str(elem) for elem in range(1, 2)))
@click.option('--n-neighbors', type=int, default=32)
@click.option('--layer_name', type=str, default="features.28")
@click.option('--config_file', default="config_files/correcting_isic/local/vgg16_Vanilla.yaml")
@click.option('--corrected_model', default=False, type=str2bool)
def main(variant, analysis_file, class_indices, n_eigval, n_clusters, n_neighbors, layer_name, config_file,
         corrected_model):
    preprocessing = VARIANTS[variant]['preprocessing']
    distance = VARIANTS[variant]['distance']

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    pipeline = SpectralClustering(
        preprocessing=preprocessing,
        pairwise_distance=distance,
        affinity=SparseKNN(n_neighbors=n_neighbors, symmetric=True),
        embedding=EigenDecomposition(n_eigval=n_eigval, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                DBSCAN(eps=k / 10.) for k in n_clusters
            ], broadcast=True),
            HDBSCAN(),
            Parallel([
                AgglomerativeClustering(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                UMAPEmbedding(),
                TSNEEmbedding(),
            ], broadcast=True)
        ], broadcast=True, is_output=True)
    )

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    if corrected_model:
        path = f"results/global_relevances_and_activations/{config['config_name']}"
    else:
        path = f"results/global_relevances_and_activations/{dataset_name}/{model_name}"

    layer = layer_name
    mode = "crvs"
    for class_index in class_indices:
        vecs = []
        print('Loading class {:03d}'.format(class_index))
        data = torch.load(f"{path}/{layer}_class_{class_index}_all.pth")
        if data['samples']:
            vecs.append(torch.stack(data[mode], 0))
            sample_ids = data['samples']
        train_flag = None

        data = torch.cat(vecs, 0)

        print('Computing class {:03d}'.format(class_index))
        (eigenvalues, embedding), (kmeans, dbscan, hdbscan, agglo, (umap, tsne)) = pipeline(data)

        print('Saving class {:03d}'.format(class_index))

        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)

        with h5py.File(analysis_file, 'a') as fp:
            analysis_name = f"{class_index}"
            g_analysis = fp.require_group(analysis_name)
            g_analysis['index'] = sample_ids

            g_embedding = g_analysis.require_group('embedding')
            g_embedding['spectral'] = embedding.astype('float32')
            g_embedding['spectral'].attrs['eigenvalue'] = eigenvalues.astype('float32')

            g_embedding['tsne'] = tsne.astype('float32')
            g_embedding['tsne'].attrs['embedding'] = 'spectral'
            g_embedding['tsne'].attrs['index'] = np.array([0, 1])

            g_embedding['umap'] = umap.astype('float32')
            g_embedding['umap'].attrs['embedding'] = 'spectral'
            g_embedding['umap'].attrs['index'] = np.array([0, 1])

            g_cluster = g_analysis.require_group('cluster')
            for n_cluster, clustering in zip(n_clusters, kmeans):
                s_cluster = 'kmeans-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, dbscan):
                s_cluster = 'dbscan-eps={:.1f}'.format(n_cluster / 10.)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            s_cluster = 'hdbscan'
            g_cluster[s_cluster] = hdbscan
            g_cluster[s_cluster].attrs['embedding'] = 'spectral'
            g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, agglo):
                s_cluster = 'agglomerative-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            if train_flag is not None:
                g_cluster['train_split'] = train_flag


if __name__ == '__main__':
    main()
