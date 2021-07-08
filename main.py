import numpy as np
import GraphDataToGraphList as gc
from GraphSet import GraphSet
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def RWL(graphset, h_max, runs, cluster_type, k, cluster_iters):

    for i in range(1,h_max):
        graphset.compute_next_gen_labels_type1(compute_pairwise_dists=True)
    graphset.compute_next_gen_labels_type1(compute_pairwise_dists=False)
    
    for i in range(runs):
        for h in range(0,h_max+1):
            graphset.cluster(h, cluster_type, k, cluster_iters)
    
    accs, mean, std = graphset.perform_svm_type1()
    return accs


def RWL_star(graphset, h_max, runs, cluster_type, k, cluster_iters):
    
    for r in range(runs):
        for i in range(h_max):
            graphset.compute_next_gen_labels_type2(i<h_max-1, cluster_type, k, cluster_iters)
        graphset.flush()
    
    accs, mean, std = graphset.perform_svm_type2()
    return accs


def main():
    parser = argparse.ArgumentParser(description='Generalized Weisfeiler-Lehman Graph Kernel')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('--approx', default=False, action='store_true', help='Use approximation variant')
    parser.add_argument('--h', type=int, required=False, default=4, help='Max. unfolding tree depth')
    parser.add_argument('--c', type=int, required=False, default=3, help='Number of paritionings')
    args = parser.parse_args()
    
    dataset_path = args.dataset
    approx = args.approx
    h_max = args.h
    runs = args.c
    cluster_type = 'barycenter'
    k = 'sqrt'
    cluster_iters = 3
    
    graph_data = gc.graph_data_to_graph_list(dataset_path)
    graphset = GraphSet(graph_data)
    
    print('... Performing 10-fold cross-validation ...')
    
    if approx: 
        accs_rwl_star = RWL_star(graphset, h_max, runs, cluster_type, k, cluster_iters)
        print('Result (acc): ', np.mean(accs_rwl_star), '+-', np.std(accs_rwl_star))
    else:
        accs_rwl = RWL(graphset, h_max, runs, cluster_type, k, cluster_iters)
        print('Result (acc): ', np.mean(accs_rwl), ' +- ', np.std(accs_rwl))
    

if __name__ == "__main__":
    main()
