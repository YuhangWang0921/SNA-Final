from pipeline_fuctions import data_to_graph, graph_to_representations, udr
from node_classification import node_classification_eva
from tools import save_embeddings, load_embeddings, generate_rep_list, reduce_dimensions, load_node_labels, load_representations

if __name__ == "__main__":
    dataset = 'BlogCatalog'

    # ## Generate and save Reps
    # graph = data_to_graph(dataset)

    k_list = [1,2,5,6,7]

    # generate_rep_list(k_list, graph)

    ## Load and eva Reps
    BC_embeddings = load_embeddings(dataset)
    # BC_embeddings = load_representations(dataset, k_list)
    BC_embeddings = dict(BC_embeddings)
    ## Load labels
    node_labels = load_node_labels(dataset=dataset)

    ## Evaluate Reps by Labels
    eva_scores,avg_micro_f1,avg_macro_f1 = node_classification_eva(node_reps=BC_embeddings, node_labels=node_labels)

    ## Select HPs by UDR
    temp_embeddings = {}
    print(f"The keys in BC_embeddings{BC_embeddings.keys()}")
    for k, representations in BC_embeddings.items():
        print(f'The current k is:{k}')
        print(f'The shape of current representation is:{representations.shape}')
        representations = reduce_dimensions(representations, 1)
        temp_embeddings[f'{k}'] = representations
    BC_embeddings.update(temp_embeddings)
    print(f"The keys in BC_embeddings{BC_embeddings.keys()}")
    ranked_list = udr(BC_embeddings)

    print(f"ranked list is:{ranked_list}")
    print(f"Baseline Micro F1 is{avg_micro_f1}")
    print(f"Baseline Macro F1 is{avg_macro_f1}")
    print(f"Performaces for each k are:{eva_scores}")