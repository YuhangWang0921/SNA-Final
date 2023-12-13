from tools import load_embeddings, load_representations

if __name__ == "__main__":
    dataset = 'BlogCatalog'
    embeddings = load_embeddings(database=dataset)
    k_list = [1,2,5,6,7]
    node_reps = load_representations(database=dataset,k_list=k_list)
    print(f"Embeddings:{len(embeddings['1'][3])}")
    print(f"node_reps:{len(node_reps['1'][3])}")