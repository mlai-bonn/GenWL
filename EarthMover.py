import numpy as np
import random
import multiprocessing as mp
import ot
import warnings
import GlobalVariables

conc_dist_mat = None

class EarthMover:
    
    # INFO: the last entry in matrices is the empty label/graph
    
    def __init__(self, initial_lbl_list, cost_a, cost_b):
        self.initial_lbl_list = initial_lbl_list
        self.initial_dist_mat = self.get_initial_distance_matrix(initial_lbl_list, cost_a, cost_b)
    
    # define initial pairwise distances in matrix with cost_a being cost for insertion and removal and cost_b cost for renaming
    def get_initial_distance_matrix(self, lbl_list, cost_a, cost_b):
        size = len(lbl_list)+1
        dist_mat = np.full((size,size), cost_a)
        for i in range(size):
            dist_mat[i,size-1] = cost_b
            dist_mat[size-1,i] = cost_b
            dist_mat[i,i] = 0
        return dist_mat
    
    # 
    def get_next_generation(self, child_lbl_list, child_dist_mat, conc_lbl_list, compute_pairwise_dists, cluster_type, k, iters):
        
        # 
        center_sparse_vectors = []
        label_clusters = []
        child_min_idx = min(child_lbl_list)
        vec_upper_size = len(self.initial_dist_mat)
        vec_lower_size = len(child_lbl_list)+1
        
        # create distance matrix made up of the parent element distance matrix M1 and the child label distance matrix M4 and inf matrices M2, M3
        m1_size = len(self.initial_dist_mat)
        m4_size = len(child_dist_mat)
        nmb_inf = 10 * np.max(child_dist_mat)
        M1 = self.initial_dist_mat
        M2 = np.full((m1_size,m4_size), nmb_inf)
        M3 = np.full((m4_size,m1_size), nmb_inf)
        M4 = child_dist_mat
        M1cM2 = np.concatenate((M1,M2), axis=1)
        M3cM4 = np.concatenate((M3,M4), axis=1)
        global conc_dist_mat 
        conc_dist_mat = np.concatenate((M1cM2,M3cM4), axis=0)
        
        # setup sparse vectors from concatenation labels
        sparse_vectors = []
        for conc_lbl in conc_lbl_list:
                lbl, children = conc_lbl
                sparse_vec = {}
                sparse_vec[self.initial_lbl_list.index(lbl)] = 1
                for c in children:
                    nidx = c - child_min_idx + vec_upper_size
                    sparse_vec.setdefault(nidx, 0) 
                    sparse_vec[nidx] += 1
                sparse_vec = list(sparse_vec.items())
                sparse_vectors.append(sparse_vec)
        
        # case: no clustering of labels
        if cluster_type == 'no_clustering':
            center_sparse_vectors = sparse_vectors
            label_clusters = []
            for conc_lbl in conc_lbl_list:
                label_clusters.append([conc_lbl])
        
        # case: cluster using barycenters
        elif cluster_type == 'barycenter':
            # determine number of centers
            if k == 'sqrt': k = int(np.sqrt(len(conc_lbl_list)))
            elif k == 'log2': k = int(np.log2(len(conc_lbl_list)))
            elif k == 'same': k = len(child_lbl_list)
            elif k == 'double': k = len(child_lbl_list) * 2
            elif isinstance(k, int): k = k
            else: raise ValueError('Invalid k value!')
            if k < 1: k = 1
            if k > len(conc_lbl_list): k = len(conc_lbl_list)
            # randomly choose k initial centers
            center_sparse_vectors = random.sample(sparse_vectors, k)
            # compute initial clusters
            point_to_center_dists = self.compute_pairwise_distances(sparse_vectors, center_sparse_vectors)
            sparse_vector_clusters = [[] for i in range(0,k)]
            for i,row in enumerate(point_to_center_dists):
                sparse_vec_i = sparse_vectors[i]
                nearest_center_idx = np.argmin(row)
                sparse_vector_clusters[nearest_center_idx].append(sparse_vec_i)
            # recalculate centers and compute new clusters 
            for _ in range(iters):
                center_sparse_vectors = self.compute_barycenters(sparse_vector_clusters)
                point_to_center_dists = self.compute_pairwise_distances(sparse_vectors, center_sparse_vectors)
                sparse_vector_clusters = [[] for i in range(0,k)]
                for i,row in enumerate(point_to_center_dists):
                    sparse_vec_i = sparse_vectors[i]
                    nearest_center_idx = np.argmin(row)
                    sparse_vector_clusters[nearest_center_idx].append(sparse_vec_i)
            # build final concatenation label clusters
            label_clusters = [[] for i in range(0,k)]
            for i,row in enumerate(point_to_center_dists):
                conc_lbl_i = conc_lbl_list[i]
                nearest_center_idx = np.argmin(row)
                label_clusters[nearest_center_idx].append(conc_lbl_i)
                
        # case: cluster using medoids
        elif cluster_type == 'medoid':
            # determine number of medoids
            if k == 'sqrt': k = int(np.sqrt(len(conc_lbl_list)))
            elif k == 'log2': k = int(np.log2(len(conc_lbl_list)))
            elif k == 'same': k = len(child_lbl_list)
            elif k == 'double': k = len(child_lbl_list) * 2
            elif isinstance(k, int): k = k
            else: raise ValueError('Invalid k value!')
            if k < 1: k = 1
            if k > len(conc_lbl_list): k = len(conc_lbl_list)
            # randomly choose k initial centers
            medoid_sparse_vectors = random.sample(sparse_vectors, k)
            # compute initial clusters
            point_to_medoid_dists = self.compute_pairwise_distances(sparse_vectors, medoid_sparse_vectors)
            sparse_vector_clusters = [[] for i in range(0,k)]
            for i,row in enumerate(point_to_medoid_dists):
                sparse_vec_i = sparse_vectors[i]
                nearest_medoid_idx = np.argmin(row)
                sparse_vector_clusters[nearest_medoid_idx].append(sparse_vec_i)
            # recalculate medoids and compute new clusters 
            for _ in range(iters):
                # 
                medoid_sparse_vectors = []
                # calculate pairwise distances within each cluster and find new cluster medoid
                for cluster_vecs in sparse_vector_clusters:
                    within_cluster_dists = self.compute_pairwise_distances(cluster_vecs, cluster_vecs)
                    cluster_medoid_idx = np.argmin(within_cluster_dists.sum(axis=1))
                    cluster_medoid = cluster_vecs[cluster_medoid_idx]
                    medoid_sparse_vectors.append(cluster_medoid)
                # assign points to new medoids
                point_to_medoid_dists = self.compute_pairwise_distances(sparse_vectors, medoid_sparse_vectors)
                sparse_vector_clusters = [[] for i in range(0,k)]
                for i,row in enumerate(point_to_medoid_dists):
                    sparse_vec_i = sparse_vectors[i]
                    nearest_medoid_idx = np.argmin(row)
                    sparse_vector_clusters[nearest_medoid_idx].append(sparse_vec_i)
            # build final concatenation label clusters
            label_clusters = [[] for i in range(0,k)]
            for i,row in enumerate(point_to_medoid_dists):
                conc_lbl_i = conc_lbl_list[i]
                nearest_medoid_idx = np.argmin(row)
                label_clusters[nearest_medoid_idx].append(conc_lbl_i)
            center_sparse_vectors = medoid_sparse_vectors
            
        # case: cluster using maximal inter cluster distance
        elif cluster_type == 'maxdist':
            label_clusters = []
            medoid_sparse_vectors = []
            conc_lbl_list_copy = conc_lbl_list.copy()
            # sample a vector and find all points within its radius k
            while sparse_vectors:
                #medoid = sparse_vectors.pop(random.randint(0, len(sparse_vectors)-1))
                medoid = random.choice(sparse_vectors)
                medoid_sparse_vectors.append(medoid)
                medoid_to_point_dists = self.compute_pairwise_distances([medoid], sparse_vectors)
                point_idcs_in_radius = [i for i,e in enumerate(medoid_to_point_dists[0]) if e <= k]
                new_label_cluster = [conc_lbl_list_copy[i] for i in point_idcs_in_radius]
                label_clusters.append(new_label_cluster)
                sparse_vectors = [sparse_vectors[i] for i,e in enumerate(sparse_vectors) if i not in point_idcs_in_radius]
                conc_lbl_list_copy = [conc_lbl_list_copy[i] for i,e in enumerate(conc_lbl_list_copy) if i not in point_idcs_in_radius]
            # build final concatenation label clusters
            center_sparse_vectors = medoid_sparse_vectors
                        
        # add empty label vector
        vec_upper_size = len(self.initial_dist_mat)
        empty_sparse_vec = [(vec_upper_size-1, 1)]
        center_sparse_vectors.append(empty_sparse_vec)
        
        # compute and return the distance matrix for center vectors
        if not compute_pairwise_dists: return None, label_clusters
        next_gen_dist_mat = self.compute_pairwise_distances(center_sparse_vectors, center_sparse_vectors)
        return next_gen_dist_mat, label_clusters
                
            
    def compute_pairwise_distances(self, sparse_vectors_a, sparse_vectors_b):
        
        # multiprocessing instances
        job_args = []
        
        # check whether distance matrix is symmetric
        symmetric = (sparse_vectors_a == sparse_vectors_b)
        
        # compute pairwise distances between vectors
        for i,vec in enumerate(sparse_vectors_a):
            job_args.append([i, vec, sparse_vectors_b, symmetric])
        with mp.Pool(processes=GlobalVariables.threads) as pool:
                dist_list = pool.starmap(calc_dist, job_args)
        
        # put together final distance matrix
        size_a = len(sparse_vectors_a)
        size_b = len(sparse_vectors_b)
        dist_mat = np.zeros((size_a, size_b))
        for e in dist_list:
            i,row = e
            dist_mat[i,:] = row
        if symmetric: 
            dist_mat += np.transpose(dist_mat)
        
        return dist_mat
    
    
    def compute_barycenters(self, sparse_vector_clusters):
        
        # multiprocessing instances
        job_args = []
        
        # compute barycenters for each cluster
        for cluster in sparse_vector_clusters:
            job_args.append([cluster])
        with mp.Pool(processes=GlobalVariables.threads) as pool:
            sparse_barycenters = pool.starmap(calc_barycenter, job_args)
        
        return sparse_barycenters

#   
def calc_dist(i, sparse_vec_i, center_sparse_vectors, symmetric):
    
    dist_row = (i, [0] * len(center_sparse_vectors))
    
    for j,sparse_vec_j in enumerate(center_sparse_vectors):
        if (symmetric and i<j) or (not symmetric): 
            
            # get only the relevant indices
            relevant_idcs_vec_i = set(k[0] for k in sparse_vec_i)
            relevant_idcs_vec_j = set(k[0] for k in sparse_vec_j)
            relevant_idcs = list(relevant_idcs_vec_i.union(relevant_idcs_vec_j))
            relevant_idcs.append(len(conc_dist_mat)-1)
            relevant_idcs.sort(key=None, reverse=False)
            
            # setup vectors
            vec_i = np.zeros(len(relevant_idcs))
            for p in sparse_vec_i: 
                idx = relevant_idcs.index(p[0])
                vec_i[idx] = p[1]
            vec_j = np.zeros(len(relevant_idcs))
            for p in sparse_vec_j: 
                idx = relevant_idcs.index(p[0])
                vec_j[idx] = p[1]
            
            # pad vectors such that they have equal l1 norm
            vec_i_norm = np.sum(vec_i)
            vec_j_norm = np.sum(vec_j)
            if vec_i_norm < vec_j_norm: vec_i[-1] = vec_j_norm - vec_i_norm
            else: vec_j[-1] = vec_i_norm - vec_j_norm
            
            # project concatenation distance matrix     
            proj_conc_dist_mat = conc_dist_mat[np.ix_(relevant_idcs, relevant_idcs)] 
            
            # calculate distance
            dist = ot.emd2(vec_i, vec_j, proj_conc_dist_mat)
            dist_row[1][j] = dist
        
    return dist_row
            
#
def calc_barycenter(sparse_vectors, reg=0.05):
    
    # get only the relevant indices
    relevant_idcs = set()
    max_norm = 0
    for sparse_vec in sparse_vectors:
        vec_idcs = set(k[0] for k in sparse_vec)
        vec_norm = sum(k[1] for k in sparse_vec)
        max_norm = max(max_norm, vec_norm)
        relevant_idcs.update(vec_idcs)
    relevant_idcs.add(len(conc_dist_mat)-1)
    relevant_idcs = list(relevant_idcs)
    relevant_idcs.sort(key=None, reverse=False)
    
    # setup vectors
    vectors = np.zeros((len(relevant_idcs), len(sparse_vectors)))
    for j,sparse_vec in enumerate(sparse_vectors):
        vec = np.zeros(len(relevant_idcs))
        vec_norm = sum(k[1] for k in sparse_vec)
        for p in sparse_vec: 
            idx = relevant_idcs.index(p[0])
            vec[idx] = p[1]
        vec[-1] = max_norm - vec_norm
        vectors[:,j] = vec
        
    # project concatenation distance matrix           
    proj_conc_dist_mat = conc_dist_mat[np.ix_(relevant_idcs, relevant_idcs)]
    
    # calculate barycenter
    warnings.filterwarnings("error")
    barycenter = None
    barycenter_ok = False
    while not barycenter_ok:
        try: 
            barycenter = ot.bregman.barycenter(vectors, proj_conc_dist_mat, reg, numItermax=1000, stopThr=0.001)
            barycenter_ok = True
        except RuntimeWarning:
            barycenter_ok = False
            reg *= 2
    
    # sparsify barycenter vector
    sparse_barycenter = []
    for j,val in enumerate(barycenter):
        idx = relevant_idcs[j]
        if val != 0: sparse_barycenter.append((idx, val))
    
    return sparse_barycenter
        
    
    
    
    
    
    
    
            
