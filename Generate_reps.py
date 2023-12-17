from pipeline_fuctions import data_to_graph, graph_to_representations


if __name__ == "__main__":
    datasets = ['BlogCatalog','Cornell','CoraGraph', 'Texas', 'KarateClub']
    for dataset in datasets:
        G = data_to_graph(dataset)
        k_list = [1,2,5,6,7]
        graph_to_representations(G, k_list, dataset,GraRep_type='GraRep')