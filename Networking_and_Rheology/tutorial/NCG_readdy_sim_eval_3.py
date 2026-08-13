# Functions for evaluation of readdy simulations of nanomotifs
#Version 2: based on Version 1, with updated methods to calculate complex moduli, based on using connected components in graph for separate G' and G'' curves (not used for final figures)
#Version 3: additional code to generate plots 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readdy
import math
import scipy
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import contextlib
import io
import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import multiprocessing

import threading

import sys
import os
from collections import defaultdict, deque

import contextlib
import io
import pickle
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error,mean_squared_log_error,mean_absolute_error



#####################
# General data
#####################

#get number of particles as a function of time from simulations
def get_particle_number_from_sims(folder,name,add_num,index_particle_tpye):
    part_number=[]
    for i in range(len(add_num)):
        #part_number_e=[]
        name_load_in=folder+name+str(add_num[i])+".h5"
        
        traj=readdy.Trajectory(name_load_in)

        times,types,ids,part_positions=traj.read_observable_particles()
        time_pt, counts_pt = traj.read_observable_number_of_particles()
        
        #for t in range(len(time_pt)):
            #count=0
        count=[]
        for j in range(len(index_particle_tpye)):
            count.append(counts_pt[:,index_particle_tpye[j]] )
            
        part_number.append(np.sum(count,axis=0) )
        #part_number.append(part_number_e)
    return np.asarray(part_number)
    
#get number of particles as a function of time from simulations
#Version 2: use full name (folder+sim name) for load in  
def get_particle_number_from_sims2(folder_name,index_particle_tpye):
    part_number=[]
    for i in range(len(folder_name)):
        #part_number_e=[]
        name_load_in=folder_name[i]
        
        traj=readdy.Trajectory(name_load_in)

        times,types,ids,part_positions=traj.read_observable_particles()
        time_pt, counts_pt = traj.read_observable_number_of_particles()
        
        #for t in range(len(time_pt)):
            #count=0
        count=[]
        for j in range(len(index_particle_tpye)):
            count.append(counts_pt[:,index_particle_tpye[j]] )
            
        part_number.append(np.sum(count,axis=0) )
        #part_number.append(part_number_e)
    return np.asarray(part_number)

#get average of largest component from several sim repeats, in each sim average [t_skip:] time steps
def avg_largest_component_size(traj_names,t_skip):
    largest_comp_s_mean=[]
    for i in range(len(traj_names)):
        #read in traj
        traj_mech=readdy.Trajectory(traj_names[i])
    
        
        time_obs_top_mech,obs_top_mech=traj_mech.read_observable_topologies()
        obs_top_mech=obs_top_mech[t_skip:]
        #get component sizes
        particles_per_component_read_in=[]
        for i in range(len(obs_top_mech)):
            comp_lens=[]
            for j in range(len(obs_top_mech[i])):
                comp_lens.append(len(obs_top_mech[i][j].particles)/13)
            particles_per_component_read_in.append(comp_lens)
        #get largest component
        largest_comp_s=[]
        for i in range(len(particles_per_component_read_in)):
            largest_comp_s.append(np.max(particles_per_component_read_in[i]))
        
        largest_comp_s_mean.append(np.mean(largest_comp_s))
    return np.mean(largest_comp_s_mean), np.std(largest_comp_s_mean)/np.sqrt(len(traj_names))
    
######################    
#### Rheological data
######################

#Theory described in Cohen et al 2024, Direct computation of viscoelastic moduli of biomolecular condensates



#get list of edges between all particles
#replace indices of edges ranging from 0 to topology length with ids of particles in topology for certain time step
def get_list_of_all_edges(time_sel,tops):
    edges_list_1=[]
    for top in tops[time_sel]:
        particles=top.particles
        edges=top.edges

        edges_with_ids=[]
        for k in range(len(edges)):
            #get indices of edges from 0 to length of top
            index_a_edge=edges[k][0]
            index_b_edge=edges[k][1]
            #convert to indicies of edges using particle id
            index_id_a_edge=particles[index_a_edge]
            index_id_b_edge=particles[index_b_edge]
            edges_with_ids.append((index_id_a_edge,index_id_b_edge))

        edges_list_1.append(edges_with_ids)
    edges_list_1 = [e for sublist in edges_list_1 for e in sublist]
    return edges_list_1
    
    
#Get all indices of vertices with four edges= centre particle
#Starting from the index of each centre particle get all indices of vertices/particles three edges away=linker particles
#get the non-surface linkers
def vertices_with_n_edges(edges,count_edges):
    # Create a dictionary to count the number of edges for each vertex
    edge_count = defaultdict(int)
    
    # Iterate over each edge and update the count for both vertices
    for edge in edges:
        edge_count[edge[0]] += 1
        edge_count[edge[1]] += 1
    
    # Get the list of vertices that have exactly four edges
    result = [vertex for vertex, count in edge_count.items() if count == count_edges]
    
    return result

def vertices_n_edges_away(edges, start_vertex,number_edges):
    # Create an adjacency list for the graph
    adjacency_list = defaultdict(list)
    
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    
    # Perform BFS to find vertices that are exactly three edges away
    queue = deque([(start_vertex, 0)])
    visited = set([start_vertex])
    result = []
    
    while queue:
        current_vertex, distance = queue.popleft()
        
        if distance == number_edges:
            result.append(current_vertex)
        
        if distance < number_edges:
            for neighbor in adjacency_list[current_vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    return result
    
#create adjacency matrix
#start from list of lists with linker vertices [[id_linker_1_motif_1,id_linker_2_motif_1,id_linker_3_motif_1,id_linker_4_motif_1],]
def create_adjacency_matrix(edges, sublists):
    # Create an adjacency list for the graph
    adjacency_list = defaultdict(list)
    
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    
    # Initialize the adjacency matrix with zeros
    n = len(sublists)
    matrix = [[0] * n for _ in range(n)]
    
    # Check for edges between vertices of different sublists
    for i in range(n):
        for j in range(i + 1, n):
            for vertex_i in sublists[i]:
                for vertex_j in sublists[j]:
                    if vertex_j in adjacency_list[vertex_i]:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
                        break
                if matrix[i][j] == 1:
                    break
    
    return matrix
    
#create adjacency matrix
#start from list of lists with linker vertices [[id_linker_1_motif_1,id_linker_2_motif_1,id_linker_3_motif_1,id_linker_4_motif_1],]
#V2: allow each vertice to have mutliuple links with other vertices (i.e. two nanomotifs can be bound by two or more arms)    
def create_adjacency_matrix_2(edges, sublists):
    # Create an adjacency list for the graph
    adjacency_list = defaultdict(list)
    
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    
    # Initialize the adjacency matrix with zeros
    n = len(sublists)
    matrix = [[0] * n for _ in range(n)]
    
    # Check for edges between vertices of different sublists
    for i in range(n):
        for j in range(i + 1, n):
            for vertex_i in sublists[i]:
                for vertex_j in sublists[j]:
                    if vertex_j in adjacency_list[vertex_i]:
                        matrix[i][j] = matrix[i][j]+1
                        matrix[j][i] = matrix[j][i]+1
                        break
                #if matrix[i][j] == 1:
                 #   break
    
    return matrix
    
    
#function that converts edges at time point t into connectivity matrix and eigenvalues from observed topology

def get_conn_mat_eigv_1(time_sel,obs_top,edges_centre,edges_linker):
    #time_sel: time point to evaluate
    #obs_top: observed topologies for all time steps
    #edges_centre: get vertices with this number of edges
    #edges_linker:get linkers this number of edges away from centre particles

    
    #all edges at time point time_sel
    edges_ids_t=get_list_of_all_edges(time_sel=time_sel,tops=obs_top)

    #get all vertices with n edges (i.e. all centre particles)
    vertices_with_n_edges_t=vertices_with_n_edges(edges=edges_ids_t,count_edges=edges_centre)

    #get all vertices m edges away from centre particles (i.e. get all linker particles)
    vertices_m_edges_away_t=[]
    for i in range(len(vertices_with_n_edges_t)):
        check_v=vertices_with_n_edges_t[i]
        vertices_m_edges_away_t.append( vertices_n_edges_away(edges=edges_ids_t, start_vertex=check_v,number_edges=edges_linker) )
        
        

    #create adjacency matrix 
    adjacency_matrix_t=create_adjacency_matrix(edges=edges_ids_t, sublists=vertices_m_edges_away_t)
    
    #create connectivity matrix 
    connectivity_matrix_t=np.zeros( (len(adjacency_matrix_t),len(adjacency_matrix_t))  )
    #fill main diagonal with number of edges (i.e. sum of each column)
    for i in range(len(connectivity_matrix_t)):
            connectivity_matrix_t[i][i]=np.sum(adjacency_matrix_t[i])
    #fill off diagonal entries with entries of connectivity matrix    
    for i in range(len(connectivity_matrix_t)):
        for j in range(i): #dont go to main diagonal 
            connectivity_matrix_t[i][j]=adjacency_matrix_t[i][j]*-1
            connectivity_matrix_t[j][i]=adjacency_matrix_t[i][j]*-1


    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(connectivity_matrix_t)

    return connectivity_matrix_t, eigenvalues
    
    
#get eigenvalues of graph laplacian split by connected sub-components (i.e. for each topology) ordered by decending component size 
def get_subgraph_eigenvalues(laplacian):
    #laplacian:graph laplacian with arbitrary labelling of nodes
    laplacian_sparse = csr_matrix(laplacian)
    n_components, labels = connected_components(csgraph=laplacian_sparse, directed=False, return_labels=True)

    components = []

    for component in range(n_components):
        component_indices = np.where(labels == component)[0]
        sub_laplacian = laplacian[np.ix_(component_indices, component_indices)]
        sub_eigenvalues = eigh(sub_laplacian, eigvals_only=True)
        components.append((len(component_indices), sub_eigenvalues))

    # Sort by component size (descending)
    components.sort(reverse=True, key=lambda x: x[0]) # components=[[size of component 0, [array of eigenvalues 0]],[size of component 1, [array of eigenvalues 1],...]

    return components
    
    
#function that converts edges at time point t into connectivity matrix and eigenvalues from observed topology
#Version 2: get eigenvalues of each connected sub component i.e. each topology 
def get_conn_mat_eigv_2(time_sel,obs_top,edges_centre,edges_linker):
    #time_sel: time point to evaluate
    #obs_top: observed topologies for all time steps
    #edges_centre: get vertices with this number of edges
    #edges_linker:get linkers this number of edges away from centre particles

    
    #all edges at time point time_sel
    edges_ids_t=get_list_of_all_edges(time_sel=time_sel,tops=obs_top)

    #get all vertices with n edges (i.e. all centre particles)
    vertices_with_n_edges_t=vertices_with_n_edges(edges=edges_ids_t,count_edges=edges_centre)

    #get all vertices m edges away from centre particles (i.e. get all linker particles)
    vertices_m_edges_away_t=[]
    for i in range(len(vertices_with_n_edges_t)):
        check_v=vertices_with_n_edges_t[i]
        vertices_m_edges_away_t.append( vertices_n_edges_away(edges=edges_ids_t, start_vertex=check_v,number_edges=edges_linker) )
        
        

    #create adjacency matrix 
    adjacency_matrix_t=create_adjacency_matrix(edges=edges_ids_t, sublists=vertices_m_edges_away_t)
    
    #create connectivity matrix  (i.e. graph laplacian)
    connectivity_matrix_t=np.zeros( (len(adjacency_matrix_t),len(adjacency_matrix_t))  )
    #fill main diagonal with number of edges (i.e. sum of each column)
    for i in range(len(connectivity_matrix_t)):
            connectivity_matrix_t[i][i]=np.sum(adjacency_matrix_t[i])
    #fill off diagonal entries with entries of connectivity matrix    
    for i in range(len(connectivity_matrix_t)):
        for j in range(i): #dont go to main diagonal 
            connectivity_matrix_t[i][j]=adjacency_matrix_t[i][j]*-1
            connectivity_matrix_t[j][i]=adjacency_matrix_t[i][j]*-1


    # Calculate the eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = np.linalg.eig(connectivity_matrix_t)
    components=get_subgraph_eigenvalues(connectivity_matrix_t)

    return connectivity_matrix_t, components
    
    
   
#function that converts edges at time point t into connectivity matrix and eigenvalues from observed topology
#Version 2: get eigenvalues of each connected sub component i.e. each topology 
#Version 3: allow each vertice to have mutliuple links with other vertices (i.e. two nanomotifs can be bound by two or more arms)
def get_conn_mat_eigv_3(time_sel,obs_top,edges_centre,edges_linker):
    #time_sel: time point to evaluate
    #obs_top: observed topologies for all time steps
    #edges_centre: get vertices with this number of edges
    #edges_linker:get linkers this number of edges away from centre particles

    
    #all edges at time point time_sel
    edges_ids_t=get_list_of_all_edges(time_sel=time_sel,tops=obs_top)

    #get all vertices with n edges (i.e. all centre particles)
    vertices_with_n_edges_t=vertices_with_n_edges(edges=edges_ids_t,count_edges=edges_centre)

    #get all vertices m edges away from centre particles (i.e. get all linker particles)
    vertices_m_edges_away_t=[]
    for i in range(len(vertices_with_n_edges_t)):
        check_v=vertices_with_n_edges_t[i]
        vertices_m_edges_away_t.append( vertices_n_edges_away(edges=edges_ids_t, start_vertex=check_v,number_edges=edges_linker) )
        
        

    #create adjacency matrix 
    adjacency_matrix_t=create_adjacency_matrix_2(edges=edges_ids_t, sublists=vertices_m_edges_away_t)
    
    #create connectivity matrix  (i.e. graph laplacian)
    connectivity_matrix_t=np.zeros( (len(adjacency_matrix_t),len(adjacency_matrix_t))  )
    #fill main diagonal with number of edges (i.e. sum of each column)
    for i in range(len(connectivity_matrix_t)):
            connectivity_matrix_t[i][i]=np.sum(adjacency_matrix_t[i])
    #fill off diagonal entries with entries of connectivity matrix    
    for i in range(len(connectivity_matrix_t)):
        for j in range(i): #dont go to main diagonal 
            connectivity_matrix_t[i][j]=adjacency_matrix_t[i][j]*-1
            connectivity_matrix_t[j][i]=adjacency_matrix_t[i][j]*-1


    # Calculate the eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = np.linalg.eig(connectivity_matrix_t)
    components=get_subgraph_eigenvalues(connectivity_matrix_t)

    return connectivity_matrix_t, components

#relaxation time from eigenvalue eigv_lambda
def relax_times(zeta,b,k,T,eigv_lambda):
    tau=(zeta * b**2)/(6 * k * T * eigv_lambda)
    return tau


#storage modulus G' for list of freq omega, from list of N-1 relaxation times tau
def storage_mod(phi,k,T,N,b,tau,omega):
    G_p=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(omega)):
        G_p_e= np.sum( (omega[i]**2 * tau**2)/(1 + omega[i]**2 * tau**2) )
        G_p.append(G_p_e)
    G_p=np.array(G_p)
    G_p=G_p*pre_factor
    return G_p


#loss modulus G'', dereferenced against solvent viscosity mu_s
#i.e. G'' - omega*mu_s
def loss_mod_deref(phi,k,T,N,b,tau,omega): 
    G_pp_dr=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(omega)):
        G_pp_dr_e= np.sum( (omega[i] * tau)/(1 + omega[i]**2 * tau**2) )
        G_pp_dr.append(G_pp_dr_e)
    G_pp_dr=np.array(G_pp_dr)
    G_pp_dr=G_pp_dr*pre_factor
    return G_pp_dr


#relaxation modulus G
def relaxation_mod(phi,k,T,N,b,tau,t):
    G=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(t)):
        G_e= np.sum( np.exp(-t[i]/tau) )
        G.append(G_e)
    G=np.array(G)
    G=G*pre_factor
    return G



#time averaged storage and loss moduli
def get_time_avg_storage_loss_mod(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,eigv_t=get_conn_mat_eigv_1(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #get relaxation times for non-zero eigenvalues
        eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
        tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)

        #collect all conn. matrices, eigenvalues and relaxation times
        conn_mat_all_t.append(conn_mat_t)
        eigv_all_t.append(eigv_t_nz)
        tau_all_t.append(tau_t)
        
        #get loss, storage and relaxation modulus
        #N_non_zero=len(eigv_t_nz)+1
        N_monomers=len(conn_mat_t)
        #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
        storage_mod_t=storage_mod(phi=phi,k=k,T=T,N=N_monomers,b=b,tau=tau_t[:],omega=omega_t)
        loss_mod_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_monomers,b=b,tau=tau_t[:],omega=omega_t)
        relaxation_mod_t=relaxation_mod(phi=phi,k=k,T=T,N=N_monomers,b=b,tau=tau_t[:],t=t)
    
        storage_mod_all_t.append(storage_mod_t)
        loss_mod_all_t.append(loss_mod_t)
        relaxation_mod_all_t.append(relaxation_mod_t)

    
    #convert to arrays   
    conn_mat_all_t=np.array(conn_mat_all_t)
    eigv_all_t=np.array(eigv_all_t)
    tau_all_t=np.array(tau_all_t)
    
    storage_mod_all_t=np.array(storage_mod_all_t)
    loss_mod_all_t=np.array(loss_mod_all_t)
    relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/np.sqrt(len(storage_mod_all_t))
    loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/np.sqrt(len(loss_mod_all_t))
    relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/np.sqrt(len(relaxation_mod_all_t))
    
    #remove imaginary part
    storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_avg_t_nc,storage_mod_stde_t_nc,loss_mod_avg_t_nc,loss_mod_stde_t_nc,relaxation_mod_avg_t_nc,relaxation_mod_stde_t_nc,conn_mat_all_t,eigv_all_t,tau_all_t
       
    
#time averaged storage and loss moduli
#Version 2: get viscoelastic properties for each sub-component of the graph laplacian
def get_time_avg_storage_loss_mod_2(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    #print("here",zeta, b,k,T)

    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,components_t=get_conn_mat_eigv_2(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #iterate over all components in the graph lapacian at current time step
        
        storage_mod_comp=[]
        loss_mod_comp=[]
        relaxation_comp=[]
        eigv_comp=[]
        tau_comp=[]
        
        #graph laplacian for time step i
        conn_mat_all_t.append(conn_mat_t)
        #print("here",zeta, b,k,T)
        for mm, (size, eigv_t) in enumerate(components_t):
    
            #get relaxation times for non-zero eigenvalues
            
            eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
            if len(eigv_t_nz)>0:
                #print("here",zeta, b,k,T)
                tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)
                #tau_t=1/eigv_t_nz

                #collect igenvalues and relaxation times for time step t and component k
                eigv_comp.append(eigv_t_nz)
                tau_comp.append(tau_t)
                #print(tau_t)
                #get loss, storage and relaxation modulus
                #N_nodes=len(eigv_t) #number of nodes in graph/objects in topology
                N_nodes=size
                #print(size)
                #N_non_zero=len(eigv_t_nz)+1
                #N_nodes=len(eigv_t_nz)+1
                #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
                storage_mod_comp_t=storage_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                loss_mod_comp_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                relaxation_mod_comp_t=relaxation_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],t=t)
            
                storage_mod_comp.append(storage_mod_comp_t)
                loss_mod_comp.append(loss_mod_comp_t)
                relaxation_comp.append(relaxation_mod_comp_t)
            

        storage_mod_all_t.append(storage_mod_comp)
        loss_mod_all_t.append(loss_mod_comp)
        relaxation_mod_all_t.append(relaxation_comp)
        eigv_all_t.append(eigv_comp)
        tau_all_t.append(tau_comp)
    #convert to arrays   
    #conn_mat_all_t=np.array(conn_mat_all_t)
    #eigv_all_t=np.array(eigv_all_t)
    #tau_all_t=np.array(tau_all_t)
    
    #storage_mod_all_t=np.array(storage_mod_all_t)
    #loss_mod_all_t=np.array(loss_mod_all_t)
    #relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    #storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    #loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    #relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    #storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    #loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    #relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    #storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    #loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    #relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    #storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    #loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    #relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_all_t,loss_mod_all_t,relaxation_mod_all_t,eigv_all_t,tau_all_t,conn_mat_all_t
    
    
#time averaged storage and loss moduli
#Version 2: get viscoelastic properties for each sub-component of the graph laplacian
#Version 3: normalize relaxation times
def get_time_avg_storage_loss_mod_3(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    #print("here",zeta, b,k,T)

    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,components_t=get_conn_mat_eigv_2(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #iterate over all components in the graph lapacian at current time step
        
        storage_mod_comp=[]
        loss_mod_comp=[]
        relaxation_comp=[]
        eigv_comp=[]
        tau_comp=[]
        
        #graph laplacian for time step i
        conn_mat_all_t.append(conn_mat_t)
        #print("here",zeta, b,k,T)
        for mm, (size, eigv_t) in enumerate(components_t):
    
            #get relaxation times for non-zero eigenvalues
            
            eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
            if len(eigv_t_nz)>0:
                #print("here",zeta, b,k,T)
                tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)
                tau_t=tau_t/np.max(tau_t)
                #tau_t=tau_t/np.max(len(eigv_t))
                #collect igenvalues and relaxation times for time step t and component k
                eigv_comp.append(eigv_t_nz)
                tau_comp.append(tau_t)
                #print(tau_t)
                #get loss, storage and relaxation modulus
                N_nodes=len(eigv_t) #number of nodes in graph/objects in topology
                #N_non_zero=len(eigv_t_nz)+1 
                #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
                storage_mod_comp_t=storage_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                loss_mod_comp_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                relaxation_mod_comp_t=relaxation_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],t=t)
            
                storage_mod_comp.append(storage_mod_comp_t)
                loss_mod_comp.append(loss_mod_comp_t)
                relaxation_comp.append(relaxation_mod_comp_t)
            

        storage_mod_all_t.append(storage_mod_comp)
        loss_mod_all_t.append(loss_mod_comp)
        relaxation_mod_all_t.append(relaxation_comp)
        eigv_all_t.append(eigv_comp)
        tau_all_t.append(tau_comp)
    #convert to arrays   
    #conn_mat_all_t=np.array(conn_mat_all_t)
    #eigv_all_t=np.array(eigv_all_t)
    #tau_all_t=np.array(tau_all_t)
    
    #storage_mod_all_t=np.array(storage_mod_all_t)
    #loss_mod_all_t=np.array(loss_mod_all_t)
    #relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    #storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    #loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    #relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    #storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    #loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    #relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    #storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    #loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    #relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    #storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    #loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    #relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_all_t,loss_mod_all_t,relaxation_mod_all_t,eigv_all_t,tau_all_t,conn_mat_all_t
    
    
    
#time averaged storage and loss moduli
#Version 2: get viscoelastic properties for each sub-component of the graph laplacian
#Version 4: allow each vertice to have mutliuple links with other vertices (i.e. two nanomotifs can be bound by two or more arms)  
def get_time_avg_storage_loss_mod_4(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    #print("here",zeta, b,k,T)

    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,components_t=get_conn_mat_eigv_3(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #iterate over all components in the graph lapacian at current time step
        
        storage_mod_comp=[]
        loss_mod_comp=[]
        relaxation_comp=[]
        eigv_comp=[]
        tau_comp=[]
        
        #graph laplacian for time step i
        conn_mat_all_t.append(conn_mat_t)
        #print("here",zeta, b,k,T)
        for mm, (size, eigv_t) in enumerate(components_t):
    
            #get relaxation times for non-zero eigenvalues
            
            eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
            if len(eigv_t_nz)>0:
                #print("here",zeta, b,k,T)
                tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)
                #tau_t=1/eigv_t_nz

                #collect igenvalues and relaxation times for time step t and component k
                eigv_comp.append(eigv_t_nz)
                tau_comp.append(tau_t)
                #print(tau_t)
                #get loss, storage and relaxation modulus
                N_nodes=len(eigv_t) #number of nodes in graph/objects in topology
                #N_non_zero=len(eigv_t_nz)+1 
                #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
                storage_mod_comp_t=storage_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                loss_mod_comp_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                relaxation_mod_comp_t=relaxation_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],t=t)
            
                storage_mod_comp.append(storage_mod_comp_t)
                loss_mod_comp.append(loss_mod_comp_t)
                relaxation_comp.append(relaxation_mod_comp_t)
            

        storage_mod_all_t.append(storage_mod_comp)
        loss_mod_all_t.append(loss_mod_comp)
        relaxation_mod_all_t.append(relaxation_comp)
        eigv_all_t.append(eigv_comp)
        tau_all_t.append(tau_comp)
    #convert to arrays   
    #conn_mat_all_t=np.array(conn_mat_all_t)
    #eigv_all_t=np.array(eigv_all_t)
    #tau_all_t=np.array(tau_all_t)
    
    #storage_mod_all_t=np.array(storage_mod_all_t)
    #loss_mod_all_t=np.array(loss_mod_all_t)
    #relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    #storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    #loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    #relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    #storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    #loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    #relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    #storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    #loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    #relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    #storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    #loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    #relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_all_t,loss_mod_all_t,relaxation_mod_all_t,eigv_all_t,tau_all_t,conn_mat_all_t
  

#get averaged moduli from full graph laplacian
#this function takes get_time_avg_storage_loss_mod_1/2/3/4 and gives time and repeat averaged moduli
def avg_moduli_3(folder_names,omega_t_sim1,t_sim1,b_u,zeta_u,k_u,T_u,phi_u,time_step,time_skip):
    storage_mod_all_sims=[]
    loss_mod_all_sims=[]
    #conn_mat_all_sims=[]
    cross_links_all_sims=[]
    for p in range(len(folder_names)):
        name_load_in=folder_names[p]
        traj_mech=readdy.Trajectory(name_load_in)

        time_obs_top_mech,obs_top_mech=traj_mech.read_observable_topologies()
    
        storage_mod_avg_t_nc,storage_mod_stde_t_nc,loss_mod_avg_t_nc,loss_mod_stde_t_nc,relaxation_mod_avg_t_nc,relaxation_mod_stde_t_nc,conn_mat_all_t,eigv_all_t,tau_all_t=get_time_avg_storage_loss_mod(obs_top=obs_top_mech,time_step=time_step,time_skip=time_skip,omega_t=omega_t_sim1,t=t_sim1,edges_centre=4,edges_linker=3,cutoff_zero_eigv=10**-9,zeta=zeta_u,b=b_u,k=k_u,T=T_u,phi=phi_u)

        storage_mod_all_sims.append(storage_mod_avg_t_nc)
        loss_mod_all_sims.append(loss_mod_avg_t_nc)
        
        cross_links_all_sims_e=[]
        for j in range(len(conn_mat_all_t)):
            cross_links_all_sims_e.append(0.5*np.trace(conn_mat_all_t[j]))
            
        cross_links_all_sims.append(np.mean(cross_links_all_sims_e))

    

    return storage_mod_all_sims,loss_mod_all_sims,cross_links_all_sims
    
#get laplacian matrix
def calculate_graph_laplacian(edges, num_nodes):
    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    # Fill the adjacency matrix based on the edges
    for edge in edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    
    # Degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    
    # Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    return laplacian_matrix
    
    
    
    
#function to shift G' and G'' to new intersection point:
def find_intersection(x1, y1, x2, y2):
    # Interpolate the data points to find the intersection
    interp1 = np.interp(x1, x2, y2)
    diff = y1 - interp1
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]
    
    intersections = []
    for index in sign_change_indices:
        x_intersect = (x1[index] + x1[index + 1]) / 2
        y_intersect = (y1[index] + y1[index + 1]) / 2
        intersections.append((x_intersect, y_intersect))
    
    return intersections


def shift_graphs(x,y,x_shift,y_shift):
    return x+x_shift, y+y_shift
    
    
    
#####################
# General data
#####################

from typing import Tuple, Literal, Optional



def _clean_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove NaNs, sort by x, and average duplicate x-values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Average duplicates in x
    if x.size == 0:
        return x, y
    # Find run boundaries of unique x
    uniq_x, idx_start = np.unique(x, return_index=True)
    # Compute means over runs
    means = np.empty_like(uniq_x, dtype=float)
    for i, start in enumerate(idx_start):
        end = idx_start[i+1] if i+1 < len(idx_start) else len(x)
        means[i] = np.mean(y[start:end])
    return uniq_x, means


def _common_grid_and_interp(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    grid: Literal["union", "linspace"] = "union",
    num: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a common x-grid over the overlapping domain and interpolate y1,y2 onto it.
    - grid="union": use the union of the available sample points within overlap.
    - grid="linspace": use an evenly spaced grid (num points) over the overlap.
    """
    x1, y1 = _clean_xy(x1, y1)
    x2, y2 = _clean_xy(x2, y2)
    if x1.size == 0 or x2.size == 0:
        raise ValueError("After cleaning, one of the curves has no valid points.")

    # Overlap domain
    lo = max(x1[0], x2[0])
    hi = min(x1[-1], x2[-1])
    if not (hi > lo):
        raise ValueError("The two curves have no overlapping x-interval to compare.")

    if grid == "union":
        x1_in = x1[(x1 >= lo) & (x1 <= hi)]
        x2_in = x2[(x2 >= lo) & (x2 <= hi)]
        x = np.unique(np.concatenate([x1_in, x2_in]))
    elif grid == "linspace":
        if num is None:
            # A reasonable default: match the denser of the two
            num = max(len(x1), len(x2))
        x = np.linspace(lo, hi, int(num))
    else:
        raise ValueError("grid must be 'union' or 'linspace'.")

    # Interpolate
    y1i = np.interp(x, x1, y1)
    y2i = np.interp(x, x2, y2)
    return x, y1i, y2i


def area_between_curves(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    grid: Literal["union", "linspace"] = "union",
    num: Optional[int] = None,
    normalize_by_length: bool = False
) -> float:
    """
    Compute the area between two curves: ∫ |f(x) - g(x)| dx over the overlapping domain.
    Uses exact piecewise-linear integration by splitting at zero-crossings of the difference.
    
    Parameters:
      - grid: 'union' uses the union of sample points (good accuracy);
              'linspace' uses an evenly spaced grid of length 'num'.
      - normalize_by_length: if True, returns the mean absolute error (MAE),
                             i.e., area divided by domain length.
    """
    x, y1i, y2i = _common_grid_and_interp(x1, y1, x2, y2, grid=grid, num=num)
    d = y1i - y2i
    dx = np.diff(x)
    d0 = d[:-1]
    d1 = d[1:]

    # Exact integral of |linear| over each segment by splitting at zero-crossings
    area = 0.0
    for i in range(len(dx)):
        a, b = d0[i], d1[i]
        h = dx[i]
        if a == 0 and b == 0:
            continue
        if a == 0 or b == 0 or np.sign(a) == np.sign(b):
            # No sign change (or touch zero at an endpoint): trapezoid on |d|
            area += 0.5 * (abs(a) + abs(b)) * h
        else:
            # Sign change inside: find zero-crossing x* where linear diff crosses 0
            # linear interpolation: d(t) = a + (b-a)*t, t in [0,1]; solve for d(t)=0
            t_zero = abs(a) / (abs(a) + abs(b))  # fraction from left where |a| decays to 0
            left = 0.5 * abs(a) * (t_zero * h)
            right = 0.5 * abs(b) * ((1 - t_zero) * h)
            area += left + right

    if normalize_by_length:
        L = x[-1] - x[0]
        return area / L if L > 0 else np.nan
    return area


#return metrics quantifying similarity between sim and exp data
def calc_r2_for_dyn_mod(x_ref,y_ref, x2,y2,X_low,X_high):
    # Example data: each curve has its own x and y arrays
    x1 = x_ref 
    y1 = y_ref

    
    # Filter x1 within the interval
    mask1 = (x1 >= X_low) & (x1 <= X_high)
    x1_filtered = x1[mask1]
    y1_filtered = y1[mask1]
    
    # Interpolate y2 to x1_filtered
    interp_func = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value="extrapolate")
    y2_interpolated = interp_func(x1_filtered)
    #fig=plt.figure(figsize=(3.5/2.54,3.5/2.54))
    plt.errorbar(x1_filtered,y2_interpolated,fmt=".")
    plt.errorbar(x1_filtered,y1_filtered,fmt=".")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    # Calculate R² value between the two curves within the interval
    r2 = r2_score(y1_filtered, y2_interpolated)
    r2_log=r2_score(np.log(y1_filtered), np.log(y2_interpolated))
    
    ##############
    ##############
    ##############
    ##############
    ##############
    ##############
    #use same x interval with log spacing
    interp_func_ref = interp1d(x_ref, y_ref, kind='linear', bounds_error=False, fill_value="extrapolate")
    x_log_space=np.geomspace(x1_filtered[0],x1_filtered[-1],num=10000)

    y2_interpolated_x_space=interp_func(x_log_space)
    y1_interpolated_x_space=interp_func_ref(x_log_space)
    # Calculate R² value between the two curves within the interval
    r2_x_space = r2_score(y1_interpolated_x_space, y2_interpolated_x_space)
    r2_log_x_space=r2_score(np.log(y1_interpolated_x_space), np.log(y2_interpolated_x_space))

    #fig=plt.figure(figsize=(3.5/2.54,3.5/2.54))
    plt.errorbar(x_log_space,y2_interpolated_x_space,fmt=".")
    plt.errorbar(x_log_space,y1_interpolated_x_space,fmt=".")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()

    
    print(f"R² value between the two curves in the interval [{X_low}, {X_high}] is: {r2:}")
    print(f"R² value for log(y) values between the two curves in the interval [{X_low}, {X_high}] is: {r2_log:}")
    

  
    print(f"With same log spaced x range, R² value between the two curves in the interval [{X_low}, {X_high}] is: {r2_x_space:}")
    print(f"With same log spaced x range, R² value for log(y) values between the two curves in the interval [{X_low}, {X_high}] is: {r2_log_x_space:}")

    #print("Mean absolute percentage error:",mean_squared_log_error(y1_filtered, y2_interpolated))
    print("With same log spaced x range, mean absolute percentage error:",mean_absolute_percentage_error(y1_interpolated_x_space, y2_interpolated_x_space))
    print("With same log spaced x range, mean absolute log error:",mean_absolute_error(np.log(y1_interpolated_x_space), np.log(y2_interpolated_x_space)))

    mae = area_between_curves(x1=np.log(x1_filtered), y1=np.log(y1_filtered), x2=np.log(x2), y2=np.log(y2), grid="union", normalize_by_length=True)
    print("Integral mean absolute log error:", mae)


def find_segments_allowing_consecutive_false(valid_mask, n_false_allowed=0):
    """
    Returns (start, end) index pairs (inclusive) of segments where the mask is mostly True,
    allowing up to n_false_allowed consecutive False values within a segment.
    """
    segments = []
    start = None
    false_count = 0

    for i, val in enumerate(valid_mask):
        if val:
            if start is None:
                start = i
            # seeing True resets the consecutive False counter
            false_count = 0
        else:
            if start is not None:
                false_count += 1
                if false_count > n_false_allowed:
                    # close segment before the run of False that exceeded the allowance
                    end = i - false_count  # last True before the False run
                    segments.append((start, end))
                    start = None
                    false_count = 0
            # if start is None and val is False, we are still outside a segment

    if start is not None:
        # include any trailing allowed False values
        segments.append((start, len(valid_mask) - 1))

    return segments
#find the freq region in which theory and exp deviate less than X % Range has to contain x_t
#up to n_false_allowed consecutive values may exceed the deviation, to account for outliers
def region_of_conf(x_ref,y_ref, x2,y2,X_low,X_high,dev_p,x_t,n_false_allowed=0):
    # Example data: each curve has its own x and y arrays
    x1 = x_ref
    y1 = y_ref

    
    # Filter x1 within the interval
    mask1 = (x1 >= X_low) & (x1 <= X_high)
    x1_filtered = x1[mask1]
    y1_filtered = y1[mask1]

    x=np.geomspace(x1_filtered[0],x1_filtered[-1],num=10000)
    
    # Interpolate y2 to x1_filtered
    interp_func = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value="extrapolate")
    y2_interpolated = interp_func(x)

    interp_func2 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value="extrapolate")
    y1_interpolated = interp_func2(x)

    #dev_all=np.abs(y2_interpolated-y1)/y1
    #mask_dev_all=dev_all<=dev_p
    #x1_filtered_dev_p=x1_filtered[mask_dev_all]

    #conf_reg=x1_filtered_dev_p[-1]-x1_filtered_dev_p[0]
    #conf_reg_log=np.log(x1_filtered_dev_p[-1])-np.log(x1_filtered_dev_p[0])

    #x=x1_filtered
    
    # Calculate deviation
    y_dev = np.abs(y2_interpolated-y1_interpolated)/y1_interpolated *100

    plt.plot(x,y2_interpolated)
    plt.plot(x,y1_interpolated)
    plt.xscale("log")
    plt.yscale("log")


    # Calculate threshold for each point
    #threshold = dev_p * np.abs(y1)

    # Create boolean mask where deviation is within threshold
    valid_mask = y_dev <= dev_p
    #print(y_dev)
        
    # Find contiguous segments where valid_mask is True
    '''segments = []
    start = None
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(valid_mask) - 1))'''
    segments=find_segments_allowing_consecutive_false(valid_mask, n_false_allowed)
    # Find the longest segment that includes x_t
    longest_segment = None
    max_length = 0
    for seg_start, seg_end in segments:
        if x[seg_start] <= x_t <= x[seg_end]:
            length = seg_end - seg_start + 1
            if length > max_length:
                max_length = length
                longest_segment = (seg_start, seg_end)
    
    # Output the result
    if longest_segment:
        x_range = (x[longest_segment[0]], x[longest_segment[1]])
        plt.axvline(x_range[0])
        plt.axvline(x_range[1])
        plt.show()
        conf_x_range=x[longest_segment[1]]-x[longest_segment[0]]
        conf_log_x_range=np.log(x[longest_segment[1]])-np.log(x[longest_segment[0]])
        conf_perc_log_x_range=conf_log_x_range/(np.log(x[-1])-np.log(x[0]))*100

        
        print(f"Longest contiguous x-range where deviation is less than {dev_p}% and includes {x_t}: {x_range}")
    else:
        print(f"No contiguous x-range found where deviation is less than {dev_p}% and includes {x_t}.")
    

        conf_x_range=None
        conf_log_x_range=None
        x_range=None
        conf_perc_log_x_range=None

    return  x_range,conf_perc_log_x_range



#####################
# Radial distribution functions
#####################

#select particles with index type_particle
def select_coords(part_positions_load,types_load,type_particle,t_skip,t_step,t_stop=None):
    #list to fill with coordinates:
    coords=[]
    #iterate over all particle positions at time point dt:
    part_positions_load=part_positions_load[t_skip:t_stop:t_step]
    types_load=types_load[t_skip:t_stop:t_step]
    print(len(types_load))
    for i in range(len(part_positions_load)):
        coords_e=[]
        for k in range(len(part_positions_load[i])):
            #print(types_load[i][k])
            if type_particle==types_load[i][k]:
                
                coords_e.append(part_positions_load[i][k])
        coords.append(coords_e)

    return np.asarray(coords)
    
#rdf function
def compute_rdf(coordinates, box_length, r_max, dr):
    N = len(coordinates)
    num_bins = int(r_max / dr)
    rdf = np.zeros(num_bins)
    r = np.linspace(dr/2, r_max - dr/2, num_bins)

    for i in range(N):
        for j in range(i+1, N):
            delta = coordinates[i] - coordinates[j]
            delta -= box_length * np.round(delta / box_length)  # Minimum image convention
            dist = np.linalg.norm(delta)
            if dist < r_max:
                bin_index = int(dist / dr)
                rdf[bin_index] += 2  # Count both i-j and j-i

    rho = N / box_length**3
    for i in range(num_bins):
        shell_volume = 4/3 * np.pi * ((r[i]+dr/2)**3 - (r[i]-dr/2)**3)
        ideal_gas_count = shell_volume *rho * N
        rdf[i] /= ideal_gas_count
    return r, rdf

#mean rdf for time steps
def mean_rdf_for_all_times(part_positions,box_length,r_max,dr):
    rdf1_list=[]
    for i in range(len(part_positions)):
        
        r1, rdf1=compute_rdf(coordinates=part_positions[i], box_length=box_length, r_max=r_max, dr=dr)
        rdf1_list.append(rdf1)
    return r1,np.mean(np.array(rdf1_list),axis=0)

#mean rdf for repeeats
def mean_rdf_repeats(file_names,box_length,r_max,dr,type_particle,t_skip,t_step,t_stop=None):
    #load saved config
    rdf_repeats=[]
    for i in range(len(file_names)):
        traj_rdf=readdy.Trajectory(file_names[i])
    
        times_rdf1,types_rdf1,ids_rdf1,part_positions_rdf1=traj_rdf.read_observable_particles()
        print(traj_rdf.species_name(type_particle))

        sel_part_positions_rdf1=select_coords(part_positions_rdf1,types_rdf1,type_particle,t_skip,t_step,t_stop)

        #print(sel_part_positions_rdf1)
        r1,rdf1_t=mean_rdf_for_all_times(sel_part_positions_rdf1,box_length,r_max,dr)
        rdf_repeats.append(rdf1_t)
    return r1, np.array(rdf_repeats)
#r1, rdf1=compute_rdf(coordinates=sel_part_positions_rdf1[-1], box_length=69.718, r_max=80, dr=2)