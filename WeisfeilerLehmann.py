from Graph import Graph

class WeisfeilerLehmann:
    
    def __init__(self):
        self.conc2lbl = {}
        #self.lbl2conc = {}
        self.idxc = 1
    
    # returns a normed initial label of the original label in the networkx graph
    def get_initial_label(self, nxgraph, node):
        lbl = 'none'
        if 'label' in nxgraph.node[node]: 
            lbl = str(nxgraph.node[node]['label'])
        if not lbl in self.conc2lbl:
            self.conc2lbl[lbl] = self.idxc
            #self.lbl2conc[self.idxc] = lbl
            self.idxc += 1
        return self.conc2lbl[lbl]
    
    # returns the new label for a concatenation label
    def get_next_gen_label(self, conc_lbl):
        return self.conc2lbl[conc_lbl]
    
    # expects a list of lists of concatenation labels and relabels each list with a new label; returns the list of new labels
    def put_next_gen_labels(self, label_clusters):
        relabels = []
        for labels in label_clusters:
            relabels.append(self.idxc)
            for conc_lbl in labels:
                self.conc2lbl[conc_lbl] = self.idxc
            self.idxc += 1
        return relabels
    
    def put_label(self, lbl):
        if lbl in self.conc2lbl: return self.conc2lbl[lbl]
        self.conc2lbl[lbl] = self.idxc
        self.idxc += 1
        return self.conc2lbl[lbl]
    
    def clear(self):
        self.conc2lbl = {}

    
