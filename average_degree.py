import networkx as nx
from pipeline_fuctions import data_to_graph

datasets = ['BlogCatalog','Cornell','CoraGraph', 'Texas', 'KarateClub']
# load each dataset
for dataset in datasets:
    graph = data_to_graph(dataset)
    # get the average degree of the graph
    average_degree = sum(dict(nx.degree(graph)).values()) / len(dict(nx.degree(graph)))
    average_degree = round(average_degree, 3)
    print(f"Average degree of {dataset} is {average_degree}")