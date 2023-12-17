import networkx as nx
import numpy as np
from scipy.stats import kendalltau
from tiny_grarep import GraRep as TinyGraRep
from legacy_grarep import GraRep
from dgl.data import CornellDataset,CoraGraphDataset, TexasDataset, KarateClubDataset

def data_to_graph(DATABASE):
    """
    Input: Dataset name
    Output: graph data
    """
    if DATABASE == 'BlogCatalog':
        G = nx.read_edgelist('Data/BlogCatalog/data/edges.csv', delimiter=",")
        G = nx.convert_node_labels_to_integers(G, first_label=0)

    elif DATABASE == 'Cornell':
        dataset = CornellDataset()
        G = dataset[0]
        G = G.to_networkx().to_undirected()

    elif DATABASE == "CoraGraph":
        dataset = CoraGraphDataset()
        G = dataset[0]
        G = G.to_networkx().to_undirected()

    elif DATABASE == "Texas":
        dataset = TexasDataset()
        G = dataset[0]
        G = G.to_networkx().to_undirected()
    elif DATABASE == "KarateClub":
        dataset = KarateClubDataset()
        G = dataset[0]
        G = G.to_networkx().to_undirected()
    else:
        raise ValueError("Invalid database name!")
    G.name = DATABASE
        
    return G


def graph_to_representations(graph, k_list, database, GraRep_type = 'GraRep'):
    """
    Generate representations(k) based on graphs and different hp: k
    Input: graph data and candidate k list
    Output: graph representations with corresponding k, dict={k:rep_list}
    """
    Embeddings = {}
    for k in k_list:
        print(f"----------Generating reps for k-step:{k}")
        if GraRep_type == 'GraRep':
            grarep = GraRep(order=k, dimensions=16)
        else:
            grarep = TinyGraRep(order=k, dimensions=16)
        grarep.fit(graph)
        embedding = grarep.get_embedding()
        Embeddings[f'{k}'] = embedding
        print(f"----------Finished reps for k-step:{k}")

    file_name = f"Data/{database}/embeddings"
    np.savez(file_name, **Embeddings)

    return

def udr(node_representations):
    """
    Calculates the Kendall's Tau Rank Correlation Coefficient for multiple models and returns the ranking results.

    Parameters:
    models_scores (dict): A dictionary containing scores_list for multiple models, where the key is the model name and the value is the scores_list.

    Returns:
    dict: A dictionary containing the name of each model and its corresponding rank.
    """
    print(f"The shape of rep_list is{node_representations.keys()}")
    # Retrieve the names of the models and their scores list
    k_value = list(node_representations.keys())
    rep_list = list(node_representations.values())
    for i in range(len(rep_list)):
        rep_list[i] = rep_list[i].flatten().tolist()
        print(f"The k is:{i} and value is:{len(rep_list[i])}")
    # Initialize a correlation matrix to store the correlation scores between models
    num_models = len(k_value)
    correlation_matrix = np.zeros((num_models, num_models))

    # Calculate the Kendall's Tau correlation score between each pair of models
    for i in range(num_models):
        for j in range(i + 1, num_models):
            print
            tau, _ = kendalltau(rep_list[i], rep_list[j])
            correlation_matrix[i, j] = tau
            correlation_matrix[j, i] = tau  # Symmetric matrix, needs to be set twice

    # Set the diagonal elements to 0, as they represent each model's correlation with itself
    np.fill_diagonal(correlation_matrix, 0)

    # Calculate the average correlation scores for each model
    average_correlation_scores = np.mean(correlation_matrix, axis=1)

    # Create a ranking dictionary based on the descending order of average correlation scores
    ranking_dict = {}
    ranked_indices = np.argsort(average_correlation_scores)[::-1]
    for rank, index in enumerate(ranked_indices):
        model_name = k_value[index]
        ranking_dict[model_name] = rank + 1  # Ranking starts from 1

    return ranking_dict