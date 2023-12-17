import tracemalloc
import legacy_grarep
import tiny_grarep
import os
import time
import numpy as np
from pipeline_fuctions import data_to_graph

k_list = [1, 2, 5, 6, 7]
datasets = ['BlogCatalog','Cornell','CoraGraph', 'Texas', 'KarateClub']
datasets.reverse()
for dataset in datasets:
    print(f"----------Using {dataset}----------")
    G = data_to_graph(dataset)
    alg_list = [tiny_grarep, legacy_grarep]
    for grarep in alg_list:
        print(f"----------Using {grarep.__name__}----------")
        start = time.time()
        for k in k_list:
            print(f"----------Generating reps for k-step:{k}----------")
            tracemalloc.start()
            # Your code here
            alg = grarep.GraRep(order=k, dimensions=32)
            alg.fit(G)

            # print memory usage
            memory_usage = tracemalloc.get_traced_memory()
            # convert to MB
            memory_usage = np.round(memory_usage[1] / (1024 * 1024), 3)
            print(f"memory_usage: {memory_usage} MB")

            tracemalloc.stop()
            end = time.time()
            time_elapsed = np.round((end - start) / 60, 3)
            print(f"Time: {time_elapsed} minutes")
            # delete all variables
            del alg
            del end
            
            if grarep.__name__ == "tiny_grarep":
                folder = f"Data/{dataset}/tmp"
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    os.remove(file_path)