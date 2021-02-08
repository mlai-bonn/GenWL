import networkx as nx
import numpy as np
from pathlib import Path
import io

def attributes_to_np_array(attr_str):
    return np.asfarray(np.array(attr_str.strip().split(",")), float)
    

def graph_data_to_graph_list(path):
    
    splits = path.split('/')
    db = splits[-1]
    
    #return variables
    graph_list = []
    graph_label_list = []
    graph_attribute_list = []
    
    #open the data files and read first line
    edge_file = open(path + "/" + db + "_A.txt", "r")
    edge = edge_file.readline().strip().split(",")
    
    #graph indicator
    graph_indicator = open(path + "/" + db + "_graph_indicator.txt", "r")
    graph = graph_indicator.readline()
    
    #graph labels
    graph_label_file = open(path + "/" + db + "_graph_labels.txt", "r")
    graph_label = graph_label_file.readline()   
    
    #node labels
    node_labels = False
    if Path(path + "/" + db + "_node_labels.txt").is_file():
        node_label_file = open(path + "/" + db + "_node_labels.txt", "r")
        node_labels = True
        node_label = node_label_file.readline()    
    
     
    #edge labels
    edge_labels = False
    if Path(path + "/" + db + "_edge_labels.txt").is_file():
        edge_label_file = open(path + "/" + db + "_edge_labels.txt", "r")
        edge_labels = True
        edge_label = edge_label_file.readline()
        
    #edge attribures
    edge_attributes = False
    if Path(path + "/" + db + "_edge_attributes.txt").is_file():
        edge_attribute_file = open(path + "/" + db + "_edge_attributes.txt", "r")
        edge_attributes = True
        edge_attribute = edge_attribute_file.readline()
    
    #node attribures
    node_attributes = False
    if Path(path + "/" + db + "_node_attributes.txt").is_file(): 
        node_attribute_file = open(path + "/" + db + "_node_attributes.txt", "r")    
        node_attributes = True
        node_attribute = node_attribute_file.readline()

    #graph attribures
    graph_attributes = False
    if Path(path + "/" + db + "_graph_attributes.txt").is_file():
        graph_attribute_file = open(path + "/" + db + "_graph_attributes.txt", "r")
        graph_attributes = True
        graph_attribute = graph_attribute_file.readline()

    
    #go through the data and read out the graphs
    node_counter = 1
    #all node_id will start with 0 for all graphs
    node_id_subtractor = 1
    while graph_label:
        G = nx.Graph()
        old_graph = graph
        new_graph = False
        
        #read out one complete graph
        while not new_graph and edge:
            #set all node labels with possibly node attributes    
            while max(int(edge[0]), int(edge[1])) >= node_counter and not new_graph:
                if graph == old_graph:
                    if node_attributes and node_labels:
                        G.add_node(node_counter - node_id_subtractor, label = attributes_to_np_array(node_label), attribute = attributes_to_np_array(node_attribute))
                        node_attribute = node_attribute_file.readline()
                        node_label = node_label_file.readline()
                    elif node_attributes:
                        G.add_node(node_counter - node_id_subtractor, attribute = attributes_to_np_array(node_attribute))
                        node_attribute = node_attribute_file.readline()
                    elif node_labels:
                        G.add_node(node_counter - node_id_subtractor, label = attributes_to_np_array(node_label))
                        node_label = node_label_file.readline()
                    else:
                        G.add_node(node_counter - node_id_subtractor)
                    node_counter += 1
                    graph = graph_indicator.readline() 
                else:
                    old_graph = graph 
                    new_graph = True
                    node_id_subtractor = node_counter

            if not new_graph:
                #set edge with possibly edge label and attributes and get next line
                if edge_labels and edge_attributes:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor, label = attributes_to_np_array(edge_label), attribute = attributes_to_np_array(edge_attribute))
                    edge_attribute = edge_attribute_file.readline()
                    edge_label = edge_label_file.readline()
                elif  edge_labels:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor, label = attributes_to_np_array(edge_label))
                    edge_label = edge_label_file.readline()
                elif edge_attributes:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor, attribute = attributes_to_np_array(edge_attribute))
                    edge_attribute = edge_attribute_file.readline()
                else:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor)
                    
                #get new edge
                edge = edge_file.readline()
                if edge:
                    edge = edge.strip().split(",")
    
        #add graph to list
        graph_list.append(G)
        
        #add graph label to list
        graph_label_list.append(int(graph_label))
        graph_label = graph_label_file.readline()
        
        #add graph attributes as numpy array
        if graph_attributes:        
            graph_attribute_list.append(attributes_to_np_array(graph_attributes))
            graph_attribute = graph_attribute_file.readline()
            
    #close all files
    edge_file.close()
    graph_indicator.close()
    graph_label_file.close()
    
    if node_labels:
        node_label_file.close()
    if edge_labels:
        edge_label_file.close()
    if edge_attributes:
        edge_attribute_file.close()
    if node_attributes:
        node_attribute_file.close()
    if graph_attributes:
        graph_attribute_file.close()
        
    #returns list of the graphs of the db, together with graph label list and possibly graph_attributes or an empty list of there are no attributes
    return (graph_list, graph_label_list, graph_attribute_list)

#node label from node_id
def node_label_vector(graph, node_id):
    if graph.has_node(node_id):
        node = graph.nodes(data = True)[node_id]
        if "label" in node.keys():
            label = node["label"]
            return label
        else:
            return []
    else:
        return []

#simple node label array from graph node labels
def nodes_label_matrix(graph):
    if "label" in graph.nodes(data = True)[0].keys():
        label_array = np.zeros((graph.number_of_nodes(), graph.nodes(data = True)[0]["label"].size))
        for i, node in enumerate(graph.nodes(data = True), 0):
            for j, entry in enumerate(node[1]["label"], 0):
                label_array[i][j] = entry 
        return label_array
    else:
        return []

#node label matrix with one hot coding, with a previous given size of coding, labels have to be of the form 0, 1, 2, 3, 4, 5, 6
def node_label_coding_matrix(graph, max_coding):
    if "label" in graph.nodes(data = True)[0].keys():
        label_mat = np.zeros((graph.number_of_nodes(), max_coding))
        for i, node in enumerate(graph.nodes(data = True), 0):
            num = int(node[1]["label"])
            if num >= 0 and num < max_coding:
                label_mat[i][num] = 1
        return label_mat
    else:
        return []


def node_attribute_vector(graph, node_id):
    node = graph.nodes(data = True)[node_id]
    if "attribute" in node.keys():
        label_mat = node["attribute"]
        return label_mat
    else:
        return []

#node attribute matrix 
def nodes_attribute_matrix(graph):
    if "attribute" in graph.nodes(data = True)[0].keys():
        label_mat = np.zeros((graph.number_of_nodes(), graph.nodes(data = True)[0]["attribute"].size))
        for i, node in enumerate(graph.nodes(data = True), 0):
            arr = node[1]["attribute"]
            for j in range(0, len(arr)):
                label_mat[i][j] = arr[j]
        return label_mat
    else:
        return []

#edge label from node_ids
def edge_label(graph, node_i, node_j):
    if graph.has_edge(node_i, node_j):
        edge = graph.get_edge_data(node_i, node_j)
        if "label" in edge.keys():
            label = edge["label"]
            return label
        else:
            return []
    else:
        return []

        
def edge_attribute_matrix(graph, node_i, node_j):
    if graph.has_edge(node_i, node_j) and "attribute" in graph.edges[node_i, node_j].keys():
        label_mat = graph.edges[node_i, node_j]["attribute"]
        return label_mat
    else:
        return []
