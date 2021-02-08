import numpy as np
from Graph import Graph
from WeisfeilerLehmann import WeisfeilerLehmann
from EarthMover import EarthMover
import SVM
import GlobalVariables

class GraphSet: 
    
    # constructor & attributes
    def __init__(self, graph_data):
        
        # setup a  WL dictionary object
        self.wl = WeisfeilerLehmann() 
        
        # WL dictionary for temporary purposes
        self.wl_temp = WeisfeilerLehmann() 
        
        # create set of graphs
        self.graph_set = []
        start_idx_counter = self.wl.idxc
        for nxgraph, glbl in zip(graph_data[0], graph_data[1]):
            g = Graph(nxgraph, glbl)
            self.graph_set.append(g)
            # set initial labels for nodes and save initial label list
            for node in g.get_nodes():
                lbl = self.wl.get_initial_label(nxgraph, node)
                g.add_node_label(node, lbl)
        end_idx_counter = self.wl.idxc
        initial_labels = list(range(start_idx_counter, end_idx_counter))
        
        # lists of all labels for each generation
        self.label_lists = []
        self.conc_label_lists = []
        self.label_lists.append(initial_labels)
        
        # 
        self.em = EarthMover(initial_labels, GlobalVariables.cost_a, GlobalVariables.cost_b)
        
        # 
        self.dist_mats = []
        self.dist_mats.append(self.em.initial_dist_mat)   
        
    
    # return all concatenated labels of the next WL generation
    def get_next_gen_conc_label_list(self):
        conc_lbls = set()
        for g in self.graph_set:
            for node in g.get_nodes():
                conc_lbl = g.get_next_gen_conc_label(node)
                conc_lbls.add(conc_lbl)
        conc_lbls = list(conc_lbls)
        conc_lbls.sort(key=None, reverse=False)
        return conc_lbls
    
    # 
    def compute_next_gen_labels_type1(self, compute_pairwise_dists):
        conc_lbl_list = self.get_next_gen_conc_label_list()
        child_dist_mat = self.dist_mats[-1]
        child_lbl_list = self.label_lists[-1]
        next_gen_dist_mat, label_clusters = self.em.get_next_generation(child_lbl_list, child_dist_mat, conc_lbl_list, compute_pairwise_dists, cluster_type='no_clustering', k=0, iters=0)
        next_gen_labels = self.wl.put_next_gen_labels(label_clusters)
        # add distance matrix and label lists
        self.label_lists.append(next_gen_labels)
        self.conc_label_lists.append(conc_lbl_list)
        self.dist_mats.append(next_gen_dist_mat)
        # adds the new labels to each graph
        for g in self.graph_set:
            for node in g.get_nodes():
                conc_lbl = g.get_next_gen_conc_label(node)
                next_gen_lbl = self.wl.get_next_gen_label(conc_lbl)
                g.add_node_label(node, next_gen_lbl)
    
    
    def compute_next_gen_labels_type2(self, compute_centers, cluster_type, k, iters=2):
        conc_lbl_list = self.get_next_gen_conc_label_list()
        child_dist_mat = self.dist_mats[-1]
        child_lbl_list = self.label_lists[-1]
        next_gen_dist_mat, label_clusters = self.em.get_next_generation(child_lbl_list, child_dist_mat, conc_lbl_list, compute_centers, cluster_type, k, iters)
        next_gen_labels = self.wl.put_next_gen_labels(label_clusters)
        # add distance matrix and label lists
        self.label_lists.append(next_gen_labels)
        self.dist_mats.append(next_gen_dist_mat)
        # adds the new labels to each graph
        for g in self.graph_set:
            for node in g.get_nodes():
                conc_lbl = g.get_next_gen_conc_label(node)
                next_gen_lbl = self.wl.get_next_gen_label(conc_lbl)
                g.add_node_label(node, next_gen_lbl)
    
    
    def cluster(self, h, cluster_type, k, iters):
        # 
        if h == 0:
            self.wl_temp.clear()
            for g in self.graph_set:
                for node in g.get_nodes():
                    orig_lbl = g.wl_node_labels[node][0]
                    nlbl = self.wl_temp.put_label(orig_lbl)
                    g.add_svm_label(nlbl)
        # 
        else:
        # 
            conc_lbl_list = self.conc_label_lists[h-1]
            child_dist_mat = self.dist_mats[h-1]
            child_lbl_list = self.label_lists[h-1]
            # 
            next_gen_dist_mat, label_clusters = self.em.get_next_generation(child_lbl_list, child_dist_mat, conc_lbl_list, False, cluster_type, k, iters)
            self.wl_temp.clear()
            self.wl_temp.put_next_gen_labels(label_clusters)
            # adds the new labels to each graph
            for g in self.graph_set:
                for node in g.get_nodes():
                    conc_lbl = g.get_conc_label_at_gen(node, h-1)
                    next_gen_lbl = self.wl_temp.get_next_gen_label(conc_lbl)
                    g.add_svm_label(next_gen_lbl)
    
    
    def flush(self):
        offset = self.wl.idxc
        self.wl.idxc += len(self.label_lists[0])
        for g in self.graph_set:
            g.flush(offset)
        self.wl.clear()
        self.label_lists = [self.label_lists[0]]
        self.dist_mats = [self.dist_mats[0]]
    
    
    # 
    def perform_svm_type1(self):
        #build vectors X, y
        X = []
        y = []
        vec_len = self.wl_temp.idxc
        for g in self.graph_set:
            vec = np.zeros(vec_len)
            g_lbl = g.get_graph_label()
            for lbl in g.svm_lbls:
                vec[lbl] += 1
            X.append(vec)
            y.append(g_lbl)
        # perform classification
        accs, mean, std = SVM.learn(X, y)
        #print(accs, mean, std)
        return accs, mean, std
    
    
    def perform_svm_type2(self):
        #build vectors X, y
        X = []
        y = []
        vec_len = self.wl.idxc
        for g in self.graph_set:
            vec = np.zeros(vec_len)
            g_lbl = g.get_graph_label()
            for lbl in g.svm_lbls:
                vec[lbl] += 1
            X.append(vec)
            y.append(g_lbl)
        # perform classification
        accs, mean, std = SVM.learn(X, y)
        #print(accs, mean, std)
        return accs, mean, std
        
