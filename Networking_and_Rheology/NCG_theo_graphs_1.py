# Functions for generation and evaluation of graph representations of networked materials

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readdy
import math
import scipy
import pandas as pd
import sklearn
import random

from NCG_readdy_sim_eval_3 import *


    
#generate a graph laplacian with N nodes, with up to m edges to other nodes (excluding itself) p of these nodes are randmoly formed
def generate_rd_graph_laplacian(N, m, p,rs):
    # Total possible edges (each node can have up to m edges to other nodes)

    np.random.seed(rs)
    max_possible_edges = N * m
    total_edges = p

    # Create a set of all possible edges (excluding self-loops)
    possible_edges = set()
    for i in range(N):
        for j in range(N):
            if i != j:
                possible_edges.add((i, j))

    # Randomly select edges to form
    #print(possible_edges,total_edges)
    selected_edges = random.sample(list(possible_edges), total_edges)

    # Initialize adjacency matrix
    A = np.zeros((N, N), dtype=int)
    for i, j in selected_edges:
        A[i][j] = 1
        A[j][i] = 1  # Ensure the graph is undirected

    # Degree matrix
    D = np.diag(np.sum(A, axis=1))

    # Laplacian matrix
    L = D - A
    return L

def generate_rd_graph_laplacian2(N, m, p,rs):
    #N:number of nodes
    #m: max number of edges for each node
    #p: total number of edges to be added randomly
    #rs: rd seed
    # Initialize adjacency matrix
    A = np.zeros((N, N), dtype=int)

    # Track number of edges added
    edges_added = 0

    # Keep trying until p edges are added
    while edges_added < p:
        np.random.seed(int((rs+1)*1000000)+edges_added )
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        # Skip self-loops and existing edges
        if i == j or A[i, j] == 1:
            continue

        # Check degree constraints
        if np.sum(A[i]) >= m or np.sum(A[j]) >= m:
            continue

        # Add edge
        A[i, j] = 1
        A[j, i] = 1
        edges_added += 1

    # Degree matrix
    D = np.diag(np.sum(A, axis=1))

    # Laplacian matrix
    L = D - A

    return L


#start with all possible edges and select p
#because of the greedy sampling, this algorithm sometimes fails to generate graphs where all nodes have the full degree
def generate_rd_graph_laplacian2_b(N, m, p,rs):
    #N:number of nodes
    #m: max number of edges for each node
    #p: total number of edges to be added randomly
    #rs: random seed
    # Generate all possible undirected edges without self-loops
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    np.random.seed(rs)
    np.random.shuffle(all_edges)

    # Initialize adjacency matrix and degree count
    A = np.zeros((N, N), dtype=int)
    degree = np.zeros(N, dtype=int)
    edges_added = 0

    for i, j in all_edges:
        if edges_added >= p:
            break
        if degree[i] < m and degree[j] < m:
            A[i, j] = 1
            A[j, i] = 1
            degree[i] += 1
            degree[j] += 1
            edges_added += 1

    # Degree matrix
    D = np.diag(degree)

    # Laplacian matrix
    L = D - A

    return L


def _k_regular_edges_circulant(N, m):
    """
    Deterministic construction of a simple undirected m-regular graph on N nodes.
    Returns a list of undirected edges (u, v) with u < v.
    Requires: 0 <= m <= N-1 and (N even if m is odd).
    """
    assert 0 <= m <= N - 1, "m must satisfy 0 <= m <= N-1"
    if m % 2 == 1:
        assert N % 2 == 0, "For odd m, N must be even"

    edges = set()
    # Even part
    even_m = m if m % 2 == 0 else m - 1
    half = even_m // 2
    for i in range(N):
        for s in range(1, half + 1):
            u, v = i, (i + s) % N
            if u > v: u, v = v, u
            edges.add((u, v))

    # If m is odd, add a perfect matching
    if m % 2 == 1:
        for i in range(N // 2):
            u, v = i, i + N // 2
            edges.add((u, v))

    return list(edges)


#this code can always generate a graph where all nodes have the max degree
def generate_rd_graph_laplacian3(N, m, p, rs=None):
    """
    Guaranteed: returns a simple undirected graph with N nodes and at most degree m,
    containing exactly p edges (0 <= p <= floor(N*m/2)), if feasible.
    If p == N*m//2, the output is m-regular.
    """
    if not (0 <= m <= N - 1):
        raise ValueError("m must satisfy 0 <= m <= N-1")
    if m % 2 == 1 and N % 2 == 1:
        raise ValueError("Odd m requires even N (graphicality).")

    max_edges = (N * m) // 2
    if p > max_edges:
        raise ValueError(f"p cannot exceed N*m/2 = {max_edges}")
    if p < 0:
        raise ValueError("p must be nonnegative")

    # Build an m-regular supergraph (deterministic)
    all_m_edges = _k_regular_edges_circulant(N, m)

    # Randomly sample p edges from the m-regular supergraph
    rng = np.random.default_rng(rs)
    if p < len(all_m_edges):
        idx = rng.choice(len(all_m_edges), size=p, replace=False)
        chosen = [all_m_edges[i] for i in idx]
    else:
        chosen = all_m_edges  # p == max_edges

    chosen=randomize_by_2switch(edges=chosen, N=N, steps=5_000, rs=rs)
    # Build adjacency and Laplacian
    A = np.zeros((N, N), dtype=int)
    deg = np.zeros(N, dtype=int)
    for u, v in chosen:
        A[u, v] = 1
        A[v, u] = 1
        deg[u] += 1
        deg[v] += 1

    D = np.diag(deg)
    L = D - A
    return L
def randomize_by_2switch(edges, N, steps=5_000, rs=None):
    """
    edges: iterable of (u,v), u < v
    Performs 'steps' random 2-switches to randomize while preserving degrees.
    Returns a list of edges (u,v) with u < v.
    """
    rng = np.random.default_rng(rs)
    E = set(edges)

    def add_edge(u, v):
        if u > v: u, v = v, u
        E.add((u, v))

    def has_edge(u, v):
        if u > v: u, v = v, u
        return (u, v) in E

    for _ in range(steps):
        if len(E) < 2:
            break
        (a, b), (c, d) = rng.choice(list(E), size=2, replace=False)

        # Distinct endpoints
        if len({a, b, c, d}) < 4:
            continue

        # Candidate new edges
        x1, y1 = a, c
        x2, y2 = b, d

        # Avoid parallel edges and keep graph simple
        if has_edge(x1, y1) or has_edge(x2, y2):
            # Try the other pairing
            x1, y1 = a, d
            x2, y2 = b, c
            if has_edge(x1, y1) or has_edge(x2, y2):
                continue

        # Perform the switch
        if a > b: a, b = b, a
        if c > d: c, d = d, c
        E.remove((a, b))
        E.remove((c, d))
        add_edge(x1, y1)
        add_edge(x2, y2)

    return list(E)

def get_moduli_from_graph_laplacian3(laplacian_matrix,omega_test,t_test):

    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    
    eigv_1_nz=np.array([e for e in eigenvalues if e>10**-9])
    tau_1_nz=relax_times(zeta=1,b=1,k=1,T=1,eigv_lambda=eigv_1_nz)
    #print(tau_1_nz)
    #N_c=6
    N_c=len(laplacian_matrix)
    storage_mod_1=storage_mod(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],omega=omega_test)
    loss_mod_1=loss_mod_deref(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],omega=omega_test)
    relaxation_mod_1=relaxation_mod(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],t=t_test)

  

    return(storage_mod_1,loss_mod_1,relaxation_mod_1)

def get_moduli_from_graph_laplacian4(laplacian_matrix,omega_test,t_test):

    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    
    eigv_1_nz=np.array([e for e in eigenvalues if e>10**-9])
    tau_1_nz=relax_times(zeta=1,b=1,k=1,T=1,eigv_lambda=eigv_1_nz)
    #print(tau_1_nz)
    #N_c=6
    N_c=len(laplacian_matrix)
    storage_mod_1=storage_mod(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],omega=omega_test)
    loss_mod_1=loss_mod_deref(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],omega=omega_test)
    relaxation_mod_1=relaxation_mod(phi=1,k=1,T=1,N=N_c,b=1,tau=tau_1_nz[:],t=t_test)

  

    return(storage_mod_1,loss_mod_1,relaxation_mod_1,tau_1_nz)

#get region between intersection between of G' and G'' and maximum of G''
def rd_inters_max2(num_samples,nodes,n_elem_max,n_elem_min,max_edges_per_node,omega_test,t_test):

    rd_elem=np.linspace(n_elem_min,n_elem_max,num_samples)

    f_max_loss_mod_list=[]
    f_intersection_list=[]
    len_f_range_list=[]
    storage_mod_list=[]
    loss_mod_list=[]
    
    for i in range(len(rd_elem)):

        #gl_i=generate_graph_laplacian_sub_comp_add_nodes_rd(N=size_grap_laplacian, m=1, m_fraction=frac,rs=i, random_sizes=True)
        gl_i=generate_rd_graph_laplacian2_b(N=nodes, m=max_edges_per_node, p=int(rd_elem[i]),rs=i)
        
        storage_mod,loss_mod,relaxation_mod=get_moduli_from_graph_laplacian3(gl_i,omega_test,t_test)
        storage_mod_list.append(storage_mod)  
        loss_mod_list.append(loss_mod)

        f_intersection_list.append(find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][0])

        f_max_loss_mod_list.append(omega_test[np.argmax(loss_mod)])
        
        len_f_range_e=np.abs(f_intersection_list[-1]-f_max_loss_mod_list[-1])/np.abs(omega_test[-1]-omega_test[0])
        len_f_range_list.append(len_f_range_e)
        #print()
    return np.array(f_intersection_list),np.array(f_max_loss_mod_list),np.array(len_f_range_list), np.array(storage_mod_list),np.array(loss_mod_list),rd_elem

#extract region between intersection between of G' and G'' and maximum of G'' for given G' and G''
def rd_inters_max_examples2(storage_mod,loss_mod,omega_test,t_test):

    f_intersection=find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][0]

    f_max_loss_mod=omega_test[np.argmax(loss_mod)]
    
    #len_f_range=np.abs(f_intersection_list[-1]-f_max_loss_mod_list[-1])/np.abs(omega_test[-1]-omega_test[0])
    len_f_range=np.abs(f_intersection-f_max_loss_mod)/np.abs(omega_test[-1]-omega_test[0])

        #print()
    return f_intersection,f_max_loss_mod,len_f_range



def generate_laplacian_from_edges(edges):
    # Extract unique nodes
    nodes = sorted(set(node for edge in edges for node in edge))
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Initialize adjacency and degree matrices
    adjacency = [[0] * n for _ in range(n)]
    degree = [0] * n

    # Fill adjacency matrix and degree list
    for u, v in edges:
        i, j = node_index[u], node_index[v]
        adjacency[i][j] = 1
        adjacency[j][i] = 1
        degree[i] += 1
        degree[j] += 1

    # Compute Laplacian: L = D - A
    laplacian = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                laplacian[i][j] = degree[i]
            else:
                laplacian[i][j] = -adjacency[i][j]

    return laplacian



#get characteristic region between intersection between of G' and G'' XX

def rd_inters_max3(num_samples,nodes,n_elem_max,n_elem_min,max_edges_per_node,omega_test,t_test,sl_loss,sl_sto):

    rd_elem=np.linspace(n_elem_min,n_elem_max,num_samples)

    f_intersection_list=[]
    f_sl_list=[]
    len_f_range_list=[]
    storage_mod_list=[]
    loss_mod_list=[]
    
    for i in range(len(rd_elem)):
        omega_test_1=np.copy(omega_test)
        #gl_i=generate_graph_laplacian_sub_comp_add_nodes_rd(N=size_grap_laplacian, m=1, m_fraction=frac,rs=i, random_sizes=True)
        gl_i=generate_rd_graph_laplacian2_b(N=nodes, m=max_edges_per_node, p=int(rd_elem[i]),rs=i)
        
        storage_mod,loss_mod,relaxation_mod=get_moduli_from_graph_laplacian3(gl_i,omega_test_1,t_test)
        storage_mod_list.append(storage_mod)  
        loss_mod_list.append(loss_mod)

        f_mod_int=find_intersection(x1=omega_test_1, y1=storage_mod, x2=omega_test_1, y2=loss_mod)[0][1]

        f_intersection_list.append(find_intersection(x1=omega_test_1, y1=storage_mod, x2=omega_test_1, y2=loss_mod)[0][0])
        
           
        #dy_dx_s = np.gradient(storage_mod, omega_test_1)/storage_mod  *omega_test_1 #*intersection_list[-1][0][0]
        #dy_dx_l = np.gradient(loss_mod, omega_test_1)/loss_mod   *omega_test_1#*intersection_list[-1][0][0]

        
        #slope_mask= (dy_dx_s >= sl_sto) & (dy_dx_s <= sl_sto) & (dy_dx_l >= sl_loss) & (dy_dx_l <= sl_loss)
        #slope_mask=  (dy_dx_l <= sl_loss)
        #print(dy_dx_s[0],dy_dx_l[0])
        mask=(loss_mod>f_mod_int*3/4)
        
        omega_test_mask=omega_test_1[mask]
        f_sl_list.append(omega_test_mask[0])

        len_char_region=np.abs(f_intersection_list[-1]-omega_test_mask[0])/f_intersection_list[-1]
        len_f_range_list.append(len_char_region)
        
    
    return np.array(f_intersection_list),np.array(f_sl_list),np.array(len_f_range_list), np.array(storage_mod_list),np.array(loss_mod_list),rd_elem

#extract characteristic region between intersection between of G' and G'' XX from given moduli

def rd_inters_max_examples3(storage_mod,loss_mod,omega_test,t_test,sl_loss,sl_sto):
    f_mod_int=find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][1]
    f_intersection=find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][0]

    omega_test_1=np.copy(omega_test)
    #dy_dx_s = np.gradient(storage_mod, omega_test_1)/storage_mod  *omega_test_1 #*intersection_list[-1][0][0]
    #dy_dx_l = np.gradient(loss_mod, omega_test_1)/loss_mod   *omega_test_1#*intersection_list[-1][0][0]

    
    #slope_mask= (dy_dx_s >= sl_sto) & (dy_dx_s <= sl_sto) & (dy_dx_l >= sl_loss) & (dy_dx_l <= sl_loss)
    #slope_mask=  (dy_dx_l <= sl_loss)
    #print(dy_dx_s[0],dy_dx_l[0])
    mask=(loss_mod>f_mod_int*3/4)
    print(f_mod_int*3/4)
    print(mask)
    omega_test_mask=omega_test_1[mask]
    f_sl=omega_test_mask[0]
    print(f_sl)
    len_char_region=np.abs(f_intersection-f_sl)/f_intersection

    return f_intersection,f_sl,len_char_region


#get region between intersection between of G' and G'' and maximum of G'' return relax times
def rd_inters_max4(num_samples,nodes,n_elem_max,n_elem_min,max_edges_per_node,omega_test,t_test):

    rd_elem=np.linspace(n_elem_min,n_elem_max,num_samples)

    f_max_loss_mod_list=[]
    f_intersection_list=[]
    len_f_range_list=[]
    storage_mod_list=[]
    loss_mod_list=[]
    relax_times_list=[]
    
    for i in range(len(rd_elem)):

        #gl_i=generate_graph_laplacian_sub_comp_add_nodes_rd(N=size_grap_laplacian, m=1, m_fraction=frac,rs=i, random_sizes=True)
        #gl_i=generate_rd_graph_laplacian2_b(N=nodes, m=max_edges_per_node, p=int(rd_elem[i]),rs=i)
        gl_i=generate_rd_graph_laplacian3(N=nodes, m=max_edges_per_node, p=int(rd_elem[i]),rs=i)
        
        storage_mod,loss_mod,relaxation_mod, relax_times=get_moduli_from_graph_laplacian4(gl_i,omega_test,t_test)
        storage_mod_list.append(storage_mod)  
        loss_mod_list.append(loss_mod)
        relax_times_list.append(relax_times)

        f_intersection_list.append(find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][0])

        f_max_loss_mod_list.append(omega_test[np.argmax(loss_mod)])
        
        len_f_range_e=np.abs(f_intersection_list[-1]-f_max_loss_mod_list[-1])/np.abs(omega_test[-1]-omega_test[0])
        len_f_range_list.append(len_f_range_e)
        #print()
    return np.array(f_intersection_list),np.array(f_max_loss_mod_list),np.array(len_f_range_list), np.array(storage_mod_list),np.array(loss_mod_list),rd_elem,relax_times_list

#extract region between intersection between of G' and G'' and maximum of G'' for given G' and G'' return relax times
def rd_inters_max_examples4(storage_mod,loss_mod,omega_test,t_test):

    f_intersection=find_intersection(x1=omega_test, y1=storage_mod, x2=omega_test, y2=loss_mod)[0][0]

    f_max_loss_mod=omega_test[np.argmax(loss_mod)]
    
    len_f_range=np.abs(f_intersection_list[-1]-f_max_loss_mod_list[-1])/np.abs(omega_test[-1]-omega_test[0])
        #print()
    return f_intersection,f_max_loss_mod,len_f_range


#generate theoretical graphs and extract rheo. prop. based on parameters in input lists
def theo_graph_gen(num_nodes_list,m_list,p_list,rs_list,omega_t_sim1,t_sim1,lim_to_one_relax_time=None,lower_t_lim=0,upper_t_lim=9999):
    storage_mod_list=[]
    loss_mod_list=[]
    relaxation_mod_list=[]
    relax_times_list=[]
    
    for i in range(len(num_nodes_list)):
            
        num_nodes_rd_1=num_nodes_list[i]
        m=m_list[i]
        p=p_list[i]
        rs=rs_list[i]
        conn_matrix_gen_rd_1=generate_rd_graph_laplacian3(N=num_nodes_rd_1, m=m, p=p,rs=rs)
        print(0.5*np.trace(conn_matrix_gen_rd_1))
        fig1=plt.figure(figsize=(3,3))
        ax1 = fig1.add_subplot(111)
        cax1=ax1.matshow(conn_matrix_gen_rd_1, cmap='viridis')
        fig1.colorbar(cax1,shrink=0.8)
        
        eigenvalues_gen_rd_1, eigenvectors_gen_rd_1 = np.linalg.eig(conn_matrix_gen_rd_1)
        
        
        
        #get relaxation times for non-zero eigenvalues
        #!! sometimes very small eigenvalues are calculated that should probably be zero, for instance for graph laplacian for N monomers, only N-1 non-zero eigenvalues exist
        eigenvalues_gen_rd_1_nz=np.array([e for e in eigenvalues_gen_rd_1 if abs(e)>10**-9])
        tau_gen_rd_1=relax_times(zeta=1,b=1,k=1,T=1,eigv_lambda=eigenvalues_gen_rd_1_nz)[0:lim_to_one_relax_time]
        
        tau_gen_rd_1=np.array(sorted(tau_gen_rd_1,reverse=True))[:]
        #print(tau_gen_rd_1)
        #print(len(tau_gen_rd_1))
        relax_times_list.append(tau_gen_rd_1)

        tau_gen_rd_1=tau_gen_rd_1[tau_gen_rd_1<upper_t_lim]
        tau_gen_rd_1=tau_gen_rd_1[tau_gen_rd_1>lower_t_lim]

        
        #get storage mod and loss mod
        #omega_gen_rd_1=np.geomspace(10**-2,10**4,10**4)
        #t_gen_rd_1=np.geomspace(10**-2,10**3,10**4)
        
        storage_mod_gen_rd_1=storage_mod(phi=1,k=1,T=1,N=num_nodes_rd_1,b=1,tau=tau_gen_rd_1[:],omega=omega_t_sim1)
        loss_mod_gen_rd_1=loss_mod_deref(phi=1,k=1,T=1,N=num_nodes_rd_1,b=1,tau=tau_gen_rd_1[:],omega=omega_t_sim1)
        relaxation_mod_gen_rd_1=relaxation_mod(phi=1,k=1,T=1,N=num_nodes_rd_1,b=1,tau=tau_gen_rd_1[:],t=t_sim1)
        storage_mod_gen_rd_1=[x.real if isinstance(x, complex) else x for x in storage_mod_gen_rd_1]
        loss_mod_gen_rd_1=[x.real if isinstance(x, complex) else x for x in loss_mod_gen_rd_1]
        relaxation_mod_gen_rd_1=[x.real if isinstance(x, complex) else x for x in relaxation_mod_gen_rd_1]

        storage_mod_list.append(storage_mod_gen_rd_1)
        loss_mod_list.append(loss_mod_gen_rd_1)
        relaxation_mod_list.append(relaxation_mod_gen_rd_1)
        
        fig1=plt.figure(figsize=(3,3))
        ax1 = fig1.add_subplot(111)
        
        ax1.errorbar(omega_t_sim1,storage_mod_gen_rd_1,fmt="-",label="G'")
        ax1.errorbar(omega_t_sim1,loss_mod_gen_rd_1,fmt="-",label="G''")
        
        omega_gen_rd_2=np.geomspace(10**0,10**2,10**3)
        ax1.errorbar(omega_t_sim1,omega_t_sim1**2,label=r"$\omega^2$",fmt="--")
        ax1.errorbar(omega_t_sim1,omega_t_sim1**0.5,label=r"$\omega^{1/2}$",fmt="--")
        ax1.errorbar(omega_t_sim1,omega_t_sim1**1,label=r"$\omega$",fmt="--")
        ax1.errorbar(omega_t_sim1,omega_t_sim1**-1,label=r"$\omega^{-1}$",fmt="--")
        
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.xticks(fontname = "Arial",fontsize=11)
        plt.yticks(fontname = "Arial",fontsize=11)
        ax1.set_ylabel("Dynamic modulus",fontname = "Arial",fontsize=11)
        ax1.set_xlabel("Frequency",fontname = "Arial",fontsize=11)
        ax1.legend(loc="upper left",prop={'family': 'Arial','size': 11})
        
    return storage_mod_list, loss_mod_list, relaxation_mod_list,relax_times_list

#shift sim. data from lists to match cross over freq. from exp data in list
def shift_mod(storage_mod_list,loss_mod_list,omega_t_list,  storage_mod_exp_list,loss_mod_exp_list,freq_list):
    storage_mod_list_shift=[]
    loss_mod_list_shift=[]
    omega_t_list_shift=[]
    cross_over_freq_exp=[]
    cross_over_mod_exp=[]
    mult_factor_freq=[]
    for i in range(len(storage_mod_list)):
        freq=freq_list[i]
        G_storage_exp=storage_mod_exp_list[i]
        G_loss_exp=loss_mod_exp_list[i]
        omega_inters_target_1,dyn_mod_inters_target_1=find_intersection(x1=freq, y1=G_storage_exp, x2=freq, y2=G_loss_exp)[0]
        #print("Intersection exp dataset 1:","omega:",omega_inters_target_1,"dynamic mod.", dyn_mod_inters_target_1)


        omega_t_sim1=omega_t_list[i]

        storage_mod_gen_rd_1=storage_mod_list[i]
        loss_mod_gen_rd_1=loss_mod_list[i]
        
        omega_inters_4,dyn_mod_inters_4=find_intersection(x1=omega_t_sim1, y1=storage_mod_gen_rd_1, x2=omega_t_sim1, y2=loss_mod_gen_rd_1)[0]
        cross_over_freq_exp.append(omega_inters_target_1)
        cross_over_mod_exp.append(dyn_mod_inters_target_1)
        
        storage_mod_gen_rd_1_shift=np.array(storage_mod_gen_rd_1)*dyn_mod_inters_target_1/dyn_mod_inters_4
        loss_mod_gen_rd_1_shift=np.array(loss_mod_gen_rd_1)*dyn_mod_inters_target_1/dyn_mod_inters_4
        
        #print("Intersection sim dataset 1:","omega:",omega_inters_4,"dynamic mod.", dyn_mod_inters_4)
        omega_t_sim1_shift_rd_1=np.array(omega_t_sim1)*omega_inters_target_1/omega_inters_4
        mult_factor_freq.append(omega_inters_target_1/omega_inters_4)

        print("R^2 for storage mod")
        calc_r2_for_dyn_mod(x_ref=freq,y_ref=G_storage_exp, x2=omega_t_sim1_shift_rd_1,y2=storage_mod_gen_rd_1_shift,X_low=5* 10**-10,X_high=5* 10**10)
        print("R^2 for loss mod")
        calc_r2_for_dyn_mod(x_ref=freq,y_ref=G_loss_exp, x2=omega_t_sim1_shift_rd_1,y2=loss_mod_gen_rd_1_shift,X_low=5* 10**-10,X_high=5* 10**10)

        storage_mod_list_shift.append(storage_mod_gen_rd_1_shift)
        loss_mod_list_shift.append(loss_mod_gen_rd_1_shift)
        omega_t_list_shift.append(omega_t_sim1_shift_rd_1)
    return storage_mod_list_shift, loss_mod_list_shift, omega_t_list_shift,cross_over_freq_exp,cross_over_mod_exp,mult_factor_freq