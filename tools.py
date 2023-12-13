import numpy as np
from pipeline_fuctions import graph_to_representations
from sklearn.decomposition import PCA
import pandas as pd
import os


def save_embeddings(embeddings, file_name):

    np.savez(file_name, **embeddings)

def load_representations(database, k_list):
    """
    Load graph representations for a list of k values.
    Input: database name and candidate k list
    Output: dict with k as keys and loaded embeddings as values
    """
    embeddings_dict = {}
    embeddings_path = f"Data/{database}/embeddings"
    
    for k in k_list:
        file_path = os.path.join(embeddings_path, f"{k}.npy")
        if os.path.exists(file_path):
            embeddings = np.load(file_path)
            key = f'{k}'
            embeddings_dict[key] = embeddings
        else:
            print(f"Embedding file for k={k} not found in {embeddings_path}.")
    
    return embeddings_dict


def load_embeddings(database):
    file_name = f"Data/{database}/embeddings.npz"
    data = np.load(file_name, allow_pickle=True)

    return {k: data[k] for k in data}

def generate_rep_list(k_list, graph, dataset):
    """
    Input: k_list and graph
    Output: rep dict:{k:node reps}
    """
    rep_dict = {}
    for k in k_list:
        rep_dict['k'] = graph_to_representations(graph, k)

    file_name = f"Data/{dataset}/embeddings.npz"
    save_embeddings(rep_dict, file_name)

def reduce_dimensions(rep, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_rep = pca.fit_transform(rep)
    return reduced_rep


def load_node_labels(dataset):
    if dataset == 'BlogCatalog':
        file_path = 'Data/BlogCatalog/data/group-edges.csv'

    df = pd.read_csv(file_path, header=None, names=['node_id', 'label'])

    # Remove duplicate node_ids, keeping the first occurrence
    df = df.drop_duplicates(subset='node_id', keep='first')

    labels_series = df.set_index('node_id')['label']
    max_node_id = labels_series.index.max()
    # labels_series = labels_series.reindex(range(max_node_id + 1), fill_value=-1)
    labels_list = labels_series.tolist()

    return labels_list

