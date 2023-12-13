import tracemalloc
import legacy_grarep as grarep
import pipeline_fuctions
import os
import time
import numpy as np

# k_list = [1, 2, 5, 6, 7]
k_list = [1]
dataset = 'BlogCatalog'
graph = pipeline_fuctions.data_to_graph(dataset)

for k in k_list:
    tracemalloc.start()
    # Your code here
    start = time.time()
    alg = grarep.GraRep(order=k, dimensions=32)
    alg.fit(graph)

    # print memory usage
    memory_usage = tracemalloc.get_traced_memory()
    # convert to MB
    memory_usage = memory_usage[1] / (1024 * 1024)
    print(f"memory_usage: {memory_usage} MB")

    tracemalloc.stop()
    end = time.time()
    time_elapsed = np.round((end - start) / 60, 3)
    print(f"elapsed_time in minutes: {time_elapsed}")
    
    # delete all variables
    del alg
    del end
