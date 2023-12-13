from pipeline_fuctions import data_to_graph, graph_to_representations, udr
from node_classification import node_classification_eva
from tools import eva_udr

if __name__ == "__main__":
    datasets = ['BlogCatalog','Cornell','CoraGraph', 'Texas', 'KarateClub']
    eva_udr(datasets)