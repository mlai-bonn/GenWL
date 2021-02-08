class Graph:
    
    # constructor & attributes
    def __init__(self, graph, label):
        self.graph_label = label
        
        #INFO: Graph nodes need to start at 0 and be successive
        self.nodes = graph.nodes()
        self.neighbors = [[] for _ in range(len(self.nodes))]
        for e in graph.edges():
            self.neighbors[e[0]].append(e[1])
            self.neighbors[e[1]].append(e[0])
        self.wl_node_labels = [[] for _ in range(len(self.nodes))]
        self.svm_lbls = []
    
    # return the concatenated label with parent node having first generation label and child nodes most recent generation labels
    def get_next_gen_conc_label(self, node):
        gen = len(self.wl_node_labels[node])-1
        lbl = self.wl_node_labels[node][0]
        neighbor_lbls = []
        for n in self.neighbors[node]:
            nlbl = self.wl_node_labels[n][gen]
            neighbor_lbls.append(nlbl)
        neighbor_lbls.sort(key=None, reverse=False)
        conc_lbl = lbl, tuple(neighbor_lbls)
        return conc_lbl
    
    def get_conc_label_at_gen(self, node, gen):
        lbl = self.wl_node_labels[node][0]
        neighbor_lbls = []
        for n in self.neighbors[node]:
            nlbl = self.wl_node_labels[n][gen]
            neighbor_lbls.append(nlbl)
        neighbor_lbls.sort(key=None, reverse=False)
        conc_lbl = lbl, tuple(neighbor_lbls)
        return conc_lbl
    
    def flush(self, offset):
        for n in self.nodes:
            node_labels = (self.wl_node_labels[n]).copy()
            node_labels[0] += offset - 1
            self.svm_lbls += node_labels
            self.wl_node_labels[n] = [self.wl_node_labels[n][0]]
    
    def get_nodes(self):
        return self.nodes
    
    def get_graph_label(self):
        return self.graph_label
    
    def get_wl_labels(self, node):
        return self.wl_node_labels[node]
    
    def add_node_label(self, node, lbl):
        self.wl_node_labels[node].append(lbl)
        
    def add_svm_label(self, lbl):
        self.svm_lbls.append(lbl)
        
        
