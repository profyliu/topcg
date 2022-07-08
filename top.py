# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:17:52 2021

@author: gn0061
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from gams import *
import time
import gc
from numba import njit, prange
from numba.typed import List
import multiprocess

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2

def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

# path is a list [depot, demand, demand, ..., depot] where depot is in range(p) and demand is in range(n)
def calc_path_len(path):
    if path[0] >= p or path[-1] >= p or len(path) < 3:
        print("invalid path")
        return(0)
    pathlen = len(path)
    total_length = cij[path[0], path[1]+p]
    if pathlen == 3:
        total_length = total_length + cij[path[1]+p, path[2]]
    else:
        for i in range(1,pathlen-2):
            total_length = total_length + cij[path[i]+p, path[i+1]+p]
        total_length = total_length + cij[path[-2]+p, path[-1]]
    return(total_length)

def calc_path_val(path):
    if path[0] >= p or path[-1] >= p or len(path) < 3:
        print("invalid path")
        return(0)
    total_val = sum(revenue[path[1:len(path)-1]]) - pathcost
    return(total_val)




def find_a_path_2(rb, msm, look_at_most=100, return_at_most = 100, seed=2021):
    np.random.seed(seed)
    n_paths = 0
    paths = []
    pathlen = []
    pathrb = []
    candidate_demands = []
    for i in range(n):
        if rb[i] > 0:
            candidate_demands.append(i)
    candidate_start_depots = [j for j in range(p) if ndrones[j] > 0]
    if len(candidate_start_depots) == 0:
        return([])
    for h in range(look_at_most):
        done = 0
        cur_path = []
        cur_dist = 0
        cur_rb = -pathcost 
        while done == 0:
            if len(cur_path) == 0:
                start_depot = w[h % K]
                cur_path.append(start_depot)
                cur_rb = cur_rb - msm[start_depot]
                feasible_next_i = [i for i in candidate_demands if cur_dist + distji[start_depot,i] + trip_dist[i] + dist_to_nearest_depot[i] < B]
                if len(feasible_next_i) > 0:
                    #chosen_i = feasible_next_i[int(np.argmax(rb[feasible_next_i]))]  # too greedy less variance
                    chosen_i = np.random.choice(feasible_next_i, 1)[0]  # choose one randomly
                    cur_dist = cur_dist + distji[start_depot, chosen_i] + trip_dist[chosen_i]
                    cur_path.append(chosen_i)
                    cur_rb = cur_rb + rb[chosen_i]
                else:
                    done = -1
            else:
                feasible_next_i = [i for i in candidate_demands if cur_dist + distii[cur_path[-1],i] + trip_dist[i] + dist_to_nearest_depot[i] < B and not np.isin(i,cur_path[1:])]
                if len(feasible_next_i) > 0:
                    #chosen_i = feasible_next_i[int(np.argmax(rb[feasible_next_i]))]
                    chosen_i = np.random.choice(feasible_next_i, 1)[0]  # choose one randomly
                    cur_dist = cur_dist + distii[cur_path[-1],chosen_i] + trip_dist[chosen_i]
                    cur_path.append(chosen_i)
                    cur_rb = cur_rb + rb[chosen_i]
                else:
                    if cur_rb <= 1e-5:
                        done = -1
                    else:
                        # complete the path by adding the nearest depot
                        chosen_depot = index_of_nearest_depot[cur_path[-1]]
                        cur_dist = cur_dist + dist_to_nearest_depot[cur_path[-1]]
                        cur_path.append(chosen_depot)
                        done = 1
        if done == -1:
            continue
        # the first element and the last element should be the starting and ending depot indexes, middle elements are demand indexes. 
        if cur_rb > 1e-5:
            paths.append(cur_path)
            pathlen.append(cur_dist)
            pathrb.append(cur_rb)
            n_paths = n_paths + 1
    unique_pathrb = list(set(pathrb))
    unique_pathrb.sort(reverse=True)
    n_unique_pathrb = len(unique_pathrb)
    n_keep = min(n_unique_pathrb, return_at_most)
    good_paths = []
    for k in range(n_keep):
        this_pathrb = unique_pathrb[k]
        keep_index = np.argwhere(pathrb == this_pathrb)[0]
        good_paths.append(paths[keep_index[0]])
    print("{:d} paths have been generated".format(len(good_paths)), end='')
    if len(good_paths) > 0:
        print(" max rb: {:.5f}".format(np.max(unique_pathrb)))  
    else:
        print("")
    return(good_paths)

# Find many collections of paths. Each collection of paths is constructed using a randomized greedy method. 
# Specifically, first find a maximal length path randomly, then exclude the demand nodes that have been covered, and find a maximal length
# path using the remaining demand nodes, until no positive profit path can be found after several trials
# This function is written so that it can be called as a separate process for parallel computing
def find_good_collections_of_paths(dictin):
    import numpy as np
    revenue = dictin['revenue']
    trip_dist = dictin['trip_dist']
    distii = dictin['distii']
    distji = dictin['distji']
    dist_to_nearest_depot = dictin['dist_to_nearest_depot']
    index_of_nearest_depot = dictin['index_of_nearest_depot']
    ndrones = dictin['ndrones']
    p = dictin['p']
    n = dictin['n']
    B = dictin['B']
    pathcost = dictin['pathcost']
    max_collections = dictin['max_collections']
    np.random.seed(dictin['seed'])
    paths = []
    pathprofit = []
    candidate_demands = []
    for i in range(n):
        if revenue[i] > 0:
            candidate_demands.append(i)
    for collection in range(max_collections):
        collection_done = 0
        remaining_demands = np.copy(candidate_demands)
        remaining_ndrones = np.copy(ndrones)
        while collection_done == 0:
            done = 0
            cur_path = []
            cur_dist = 0
            cur_profit = -pathcost 
            while done == 0:
                if len(cur_path) == 0:
                    candidate_start_depots = [j for j in range(p) if remaining_ndrones[j] > 0]
                    if len(candidate_start_depots) == 0:
                        done = -1
                        break
                    start_depot = np.random.choice(candidate_start_depots, 1)[0]  # choose a start depot randomly
                    cur_path.append(start_depot)
                    feasible_next_i = [i for i in remaining_demands if cur_dist + distji[start_depot,i] + trip_dist[i] + dist_to_nearest_depot[i] < B]
                    if len(feasible_next_i) > 0:
                        #chosen_i = feasible_next_i[int(np.argmax(rb[feasible_next_i]))]  # too greedy less variance
                        chosen_i = np.random.choice(feasible_next_i, 1)[0]  # choose one randomly
                        cur_dist = cur_dist + distji[start_depot, chosen_i] + trip_dist[chosen_i]
                        cur_path.append(chosen_i)
                        cur_profit = cur_profit + revenue[chosen_i]
                    else:
                        done = -1
                else:
                    feasible_next_i = [i for i in remaining_demands if cur_dist + distii[cur_path[-1],i] + trip_dist[i] + dist_to_nearest_depot[i] < B and not np.isin(i,cur_path[1:])]
                    if len(feasible_next_i) > 0:
                        #chosen_i = feasible_next_i[int(np.argmax(rb[feasible_next_i]))]
                        chosen_i = np.random.choice(feasible_next_i, 1)[0]  # choose one randomly
                        cur_dist = cur_dist + distii[cur_path[-1],chosen_i] + trip_dist[chosen_i]
                        cur_path.append(chosen_i)
                        cur_profit = cur_profit + revenue[chosen_i]
                    else:
                        if cur_profit <= 1e-5:
                            done = -1
                        else:
                            # complete the path by adding the nearest depot
                            chosen_depot = index_of_nearest_depot[cur_path[-1]]
                            cur_dist = cur_dist + dist_to_nearest_depot[cur_path[-1]]
                            cur_path.append(chosen_depot)
                            done = 1
            if done == 1:
                paths.append(cur_path)
                pathprofit.append(cur_profit)
                # update the collection info
                remaining_demands = [i for i in remaining_demands if i not in cur_path[1:(len(cur_path)-1)]]
                remaining_ndrones[cur_path[0]] -= 1
            elif done == -1:
                # collection is done, start a new collection
                collection_done = 1
    # post-processing the paths: if two paths has the same starting depot and the same profit, one is kept
    kept_paths = []
    unique_keys = set()
    for k in range(len(paths)):
        this_key = (paths[k][0], pathprofit[k])
        if this_key not in unique_keys:
            unique_keys.add(this_key)
            kept_paths.append(paths[k])
    return(kept_paths)

'''
n_total_collections = 20000
ncores = 10
dictin = {'revenue':revenue,'ndrones':ndrones, 'max_collections': int(n_total_collections/ncores),
          'trip_dist':trip_dist, 'distii':distii, 'distji':distji,
          'dist_to_nearest_depot':dist_to_nearest_depot,
          'index_of_nearest_depot':index_of_nearest_depot,
          'p':p,'n':n,'B':B,'pathcost':pathcost,'seed':2021}
'''

# caller function for find_good_collections_of_paths for multiprocess
def run_find_good_collections_of_paths_in_parallel(n_total_collections, ncores, seed=2021):
    find_start_time = time.time()
    # try to partition the search space to reduce possibility of duplicate paths from different parallel processes
    
    pool_obj = multiprocess.Pool()
    arglist = []
    for i in range(ncores):
        arglist.append({'revenue':revenue,'ndrones':ndrones, 'max_collections': int(n_total_collections/ncores),
                        'trip_dist':trip_dist, 'distii':distii, 'distji':distji,
                        'dist_to_nearest_depot':dist_to_nearest_depot,
                        'index_of_nearest_depot':index_of_nearest_depot,
                        'p':p,'n':n,'B':B,'pathcost':pathcost,'seed':(seed + 2021*i)})
    paths_all = pool_obj.map(find_good_collections_of_paths, arglist)
    paths_list = [item for sublist in paths_all for item in sublist]
    print("parallel find time: {:.1f}".format(time.time() - find_start_time))
    # deduplicate 
    dedup_start_time = time.time()
    kept_paths = []
    unique_keys = set()
    for k in range(len(paths_list)):
        this_key = (paths_list[k][0], frozenset(paths_list[k][1:len(paths_list[k])-1]))
        if this_key not in unique_keys:
            unique_keys.add(this_key)
            kept_paths.append(paths_list[k])
    print("paths after final dedup: {:d} dedup time: {:.1f}".format(len(kept_paths), time.time() - dedup_start_time))
    return(kept_paths)


# only returns the maximum length paths
def find_max_paths_dedup(rb, msm):
    # first get all eligible demand nodes
    candidate_demands = np.squeeze(np.argwhere(rb>0), axis=1)  # subset of index (1..n) with positive rb  
    candidate_start_depots = np.squeeze(np.argwhere(ndrones), axis=1)  # only the depots having at least a drone can be the starting depot
    pp = {}  # partial paths
    pplen = {}  # partial path lengths
    ppval = {}  # partial path values
    lastind = 0  # last node index
    gp = List()  # global path pool
    gpval = {}  # values of paths in the global pool 
    gpn = 0  # number of paths in the global pool
    for start_depot in candidate_start_depots:
        # evaluate the partial path from start_depot to each candidate demand; if feasible, add to list
        for j in candidate_demands:
            if cij[start_depot, j+p] + dist_to_nearest_depot[j] > B:
                continue
            pp[lastind] = [start_depot, j]
            pplen[lastind] = cij[start_depot, j+p]
            ppval[lastind] = rb[j] - msm[start_depot] - pathcost
            lastind = lastind + 1
    nlevels = 1  # how many demand nodes in the next level of partial paths (nextpp)
    while lastind > 0:
        nlevels = nlevels + 1
        nextpp = {}
        nextpplen = {}
        nextppval = {}
        nextlastind = 0
        set_of_sets_of_demands = set()
        setsignext = set()  # set of signatures added
        setsignext_len = {}
        setsignext_pos = {}  # position in the nextpp list of best partial path covering a demand set
        for k in range(lastind):
            cur_pp = pp[k]
            cur_pplen = pplen[k]
            cur_ppval = ppval[k]
            cur_i = cur_pp[-1]  # current demand node is the last element in the partial path
            max_length = True 
            for j in Diff(candidate_demands, cur_pp[1:]):
                if cur_pplen + cij[cur_i+p, j+p] + dist_to_nearest_depot[j] > B:
                    continue
                max_length = False
                this_pp = cur_pp + [j]
                this_pplen = cur_pplen + cij[cur_i+p, j+p]
                this_ppval = cur_ppval + rb[j]
                # check if the unordered set of demands covered by the same starting depot and ending demand node has been entered or not
                # if one exists: if this_pp has a smaller len, replace the existing; otherwise, do not add this_pp
                # if none exisits: add this_pp, take notes of its set signature, value and index in nextpp
                this_sig = frozenset(set([this_pp[0]-p] + this_pp[1:nlevels] + [this_pp[-1]+n+p]))
                if this_sig not in setsignext:
                    setsignext.add(this_sig)
                    setsignext_len[this_sig] = this_pplen
                    setsignext_pos[this_sig] = nextlastind
                    nextpp[nextlastind] = this_pp
                    nextpplen[nextlastind] = this_pplen
                    nextppval[nextlastind] = this_ppval 
                    nextlastind = nextlastind + 1
                else:
                    best_sig_len = setsignext_len[this_sig]
                    if this_pplen < best_sig_len:
                        setsignext_len[this_sig] = this_pplen  # update 
                        best_sig_pos = setsignext_pos[this_sig]
                        nextpp[best_sig_pos] = this_pp
                        nextpplen[best_sig_pos] = this_pplen         
                        nextppval[best_sig_pos] = this_ppval
            # close and save this partial path (if its value is positive) and add it to the global pool
            if cur_ppval > 0 and max_length == True:
                # check if the unordered set of demands visited in this partial path has been added before; if yes, ignore; if no, add it and update the bookkeepting list
                cur_demand_set = frozenset(set([cur_pp[0]-p] + cur_pp[1:]))
                if cur_demand_set not in set_of_sets_of_demands:
                    set_of_sets_of_demands.add(cur_demand_set)
                    last_demand_node = cur_pp[-1]
                    gp.append(List(cur_pp + [index_of_nearest_depot[last_demand_node]]))
                    gpval[gpn] = cur_ppval
                    gpn = gpn + 1                
        '''
        del set_of_sets_of_demands 
        del setsignext 
        del setsignext_len
        del setsignext_pos
        del pp
        del pplen
        del ppval
        gc.collect()
        '''
        pp = nextpp
        pplen = nextpplen
        ppval = nextppval
        lastind = nextlastind        
    return(gp)


# only returns the maximum length paths
# returns the max paths that originates from a given depot
def find_max_paths_dedup_from_a_depot(dictin):
    import numpy as np
    rb = dictin['rb']
    msm = dictin['msm']
    start_depot = dictin['start_depot']
    cij = dictin['cij']
    dist_to_nearest_depot = dictin['dist_to_nearest_depot']
    index_of_nearest_depot = dictin['index_of_nearest_depot']
    p = dictin['p']
    n = dictin['n']
    B = dictin['B']
    pathcost = dictin['pathcost']
    candidate_demands = np.squeeze(np.argwhere(rb>0), axis=1)  # subset of index (1..n) with positive rb  
    pp = {}  # partial paths
    pplen = {}  # partial path lengths
    ppval = {}  # partial path values
    lastind = 0  # last node index
    gp = []  # global path pool
    gpval = {}  # values of paths in the global pool 
    gpn = 0  # number of paths in the global pool
    for j in candidate_demands:
        if cij[start_depot, j+p] + dist_to_nearest_depot[j] > B:
            continue
        pp[lastind] = [start_depot, j]
        pplen[lastind] = cij[start_depot, j+p]
        ppval[lastind] = rb[j] - msm[start_depot] - pathcost
        lastind = lastind + 1
    nlevels = 1  # how many demand nodes in the next level of partial paths (nextpp)
    while lastind > 0:
        nlevels = nlevels + 1
        nextpp = {}
        nextpplen = {}
        nextppval = {}
        nextlastind = 0
        set_of_sets_of_demands = set()
        setsignext = set()  # set of signatures added
        setsignext_len = {}
        setsignext_pos = {}  # position in the nextpp list of best partial path covering a demand set
        for k in range(lastind):
            cur_pp = pp[k]
            cur_pplen = pplen[k]
            cur_ppval = ppval[k]
            cur_i = cur_pp[-1]  # current demand node is the last element in the partial path
            max_length = True 
            #unvisited_demands = Diff(candidate_demands, cur_pp[1:])
            unvisited_demands = list(set(candidate_demands) - set(cur_pp[1:])) + list(set(cur_pp[1:]) - set(candidate_demands))
            for j in unvisited_demands:
                if cur_pplen + cij[cur_i+p, j+p] + dist_to_nearest_depot[j] > B:
                    continue
                max_length = False
                this_pp = cur_pp + [j]
                this_pplen = cur_pplen + cij[cur_i+p, j+p]
                this_ppval = cur_ppval + rb[j]
                # check if the unordered set of demands covered by the same starting depot and ending demand node has been entered or not
                # if one exists: if this_pp has a smaller len, replace the existing; otherwise, do not add this_pp
                # if none exisits: add this_pp, take notes of its set signature, value and index in nextpp
                this_sig = frozenset(set([this_pp[0]-p] + this_pp[1:nlevels] + [this_pp[-1]+n+p]))
                if this_sig not in setsignext:
                    setsignext.add(this_sig)
                    setsignext_len[this_sig] = this_pplen
                    setsignext_pos[this_sig] = nextlastind
                    nextpp[nextlastind] = this_pp
                    nextpplen[nextlastind] = this_pplen
                    nextppval[nextlastind] = this_ppval 
                    nextlastind = nextlastind + 1
                else:
                    best_sig_len = setsignext_len[this_sig]
                    if this_pplen < best_sig_len:
                        setsignext_len[this_sig] = this_pplen  # update 
                        best_sig_pos = setsignext_pos[this_sig]
                        nextpp[best_sig_pos] = this_pp
                        nextpplen[best_sig_pos] = this_pplen         
                        nextppval[best_sig_pos] = this_ppval
            # close and save this partial path (if its value is positive) and add it to the global pool
            if cur_ppval > 0 and max_length == True:
                # check if the unordered set of demands visited in this partial path has been added before; if yes, ignore; if no, add it and update the bookkeepting list
                cur_demand_set = frozenset(set([cur_pp[0]-p] + cur_pp[1:]))
                if cur_demand_set not in set_of_sets_of_demands:
                    set_of_sets_of_demands.add(cur_demand_set)
                    last_demand_node = cur_pp[-1]
                    gp.append(cur_pp + [index_of_nearest_depot[last_demand_node]])
                    gpval[gpn] = cur_ppval
                    gpn = gpn + 1                
        '''
        del set_of_sets_of_demands 
        del setsignext 
        del setsignext_len
        del setsignext_pos
        del pp
        del pplen
        del ppval
        gc.collect()
        '''
        pp = nextpp
        pplen = nextpplen
        ppval = nextppval
        lastind = nextlastind        
    return(gp)

def find_max_paths_dedup_multiprocessing(rb, msm):
    pool_obj = multiprocess.Pool()
    arglist = []
    for i in range(p):
        arglist.append({'rb':rb,'msm':msm,'start_depot':i,
                        'cij':cij,'dist_to_nearest_depot':dist_to_nearest_depot,
                        'index_of_nearest_depot':index_of_nearest_depot,
                        'p':p,'n':n,'B':B,'pathcost':pathcost})
    paths_all = pool_obj.map(find_max_paths_dedup_from_a_depot, arglist)
    paths_list = List([List(item) for sublist in paths_all for item in sublist])
    return(paths_list)

def find_all_paths_dedup(rb, msm):
    # first get all eligible demand nodes
    candidate_demands = np.squeeze(np.argwhere(rb>0), axis=1)  # subset of index (1..n) with positive rb  
    candidate_start_depots = np.squeeze(np.argwhere(ndrones), axis=1)  # only the depots having at least a drone can be the starting depot
    pp = {}  # partial paths
    pplen = {}  # partial path lengths
    ppval = {}  # partial path values
    lastind = 0  # last node index
    gp = []  # global path pool
    gpn = 0  # number of paths in the global pool
    for start_depot in candidate_start_depots:
        # evaluate the partial path from start_depot to each candidate demand; if feasible, add to list
        for j in candidate_demands:
            if cij[start_depot, j+p] + dist_to_nearest_depot[j] > B:
                continue
            pp[lastind] = [start_depot, j]
            pplen[lastind] = cij[start_depot, j+p]
            ppval[lastind] = rb[j] - msm[start_depot] - pathcost
            lastind = lastind + 1
    nlevels = 1  # how many demand nodes in the next level of partial paths (nextpp)
    while lastind > 0:
        nlevels = nlevels + 1
        nextpp = {}
        nextpplen = {}
        nextppval = {}
        nextlastind = 0
        set_of_sets_of_demands = set()
        setsignext = set()  # set of signatures added
        setsignext_len = {}
        setsignext_pos = {}  # position in the nextpp list of best partial path covering a demand set
        for k in range(lastind):
            cur_pp = pp[k]
            cur_pplen = pplen[k]
            cur_ppval = ppval[k]
            cur_i = cur_pp[-1]  # current demand node is the last element in the partial path
            for j in Diff(candidate_demands, cur_pp[1:]):
                if cur_pplen + cij[cur_i+p, j+p] + dist_to_nearest_depot[j] > B:
                    continue
                this_pp = cur_pp + [j]
                this_pplen = cur_pplen + cij[cur_i+p, j+p]
                this_ppval = cur_ppval + rb[j]
                # check if the unordered set of demands covered by the same starting depot and ending demand node has been entered or not
                # if one exists: if this_pp has a smaller len, replace the existing; otherwise, do not add this_pp
                # if none exisits: add this_pp, take notes of its set signature, value and index in nextpp
                this_sig = frozenset(set([this_pp[0]-p] + this_pp[1:nlevels] + [this_pp[-1]+n+p]))
                if this_sig not in setsignext:
                    setsignext.add(this_sig)
                    setsignext_len[this_sig] = this_pplen
                    setsignext_pos[this_sig] = nextlastind
                    nextpp[nextlastind] = this_pp
                    nextpplen[nextlastind] = this_pplen
                    nextppval[nextlastind] = this_ppval 
                    nextlastind = nextlastind + 1
                else:
                    best_sig_len = setsignext_len[this_sig]
                    if this_pplen < best_sig_len:
                        setsignext_len[this_sig] = this_pplen  # update 
                        best_sig_pos = setsignext_pos[this_sig]
                        nextpp[best_sig_pos] = this_pp
                        nextpplen[best_sig_pos] = this_pplen         
                        nextppval[best_sig_pos] = this_ppval
            # close and save this partial path (if its value is positive) and add it to the global pool
            if cur_ppval > 0:
                # check if the unordered set of demands visited in this partial path has been added before; if yes, ignore; if no, add it and update the bookkeepting list
                cur_demand_set = frozenset(set([cur_pp[0]-p] + cur_pp[1:]))
                if cur_demand_set not in set_of_sets_of_demands:
                    set_of_sets_of_demands.add(cur_demand_set)
                    last_demand_node = cur_pp[-1]
                    gp.append(cur_pp + [index_of_nearest_depot[last_demand_node]])
                    gpn = gpn + 1                
        pp = nextpp
        pplen = nextpplen
        ppval = nextppval
        lastind = nextlastind        
    return(gp)


def find_all_paths_dedup_from_a_depot(dictin):
    import numpy as np
    # first get all eligible demand nodes
    rb = dictin['rb']
    msm = dictin['msm']
    start_depot = dictin['start_depot']
    cij = dictin['cij']
    dist_to_nearest_depot = dictin['dist_to_nearest_depot']
    index_of_nearest_depot = dictin['index_of_nearest_depot']
    p = dictin['p']
    n = dictin['n']
    B = dictin['B']
    pathcost = dictin['pathcost']
    candidate_demands = np.squeeze(np.argwhere(rb>0), axis=1)  # subset of index (1..n) with positive rb  
    pp = {}  # partial paths
    pplen = {}  # partial path lengths
    ppval = {}  # partial path values
    lastind = 0  # last node index
    gp = []  # global path pool
    gpn = 0  # number of paths in the global pool
    for j in candidate_demands:
        if cij[start_depot, j+p] + dist_to_nearest_depot[j] > B:
            continue
        pp[lastind] = [start_depot, j]
        pplen[lastind] = cij[start_depot, j+p]
        ppval[lastind] = rb[j] - msm[start_depot] - pathcost
        lastind = lastind + 1
    nlevels = 1  # how many demand nodes in the next level of partial paths (nextpp)
    while lastind > 0:
        nlevels = nlevels + 1
        nextpp = {}
        nextpplen = {}
        nextppval = {}
        nextlastind = 0
        set_of_sets_of_demands = set()
        setsignext = set()  # set of signatures added
        setsignext_len = {}
        setsignext_pos = {}  # position in the nextpp list of best partial path covering a demand set
        for k in range(lastind):
            cur_pp = pp[k]
            cur_pplen = pplen[k]
            cur_ppval = ppval[k]
            cur_i = cur_pp[-1]  # current demand node is the last element in the partial path
            unvisited_demands = list(set(candidate_demands) - set(cur_pp[1:])) + list(set(cur_pp[1:]) - set(candidate_demands))
            for j in unvisited_demands:
                if cur_pplen + cij[cur_i+p, j+p] + dist_to_nearest_depot[j] > B:
                    continue
                this_pp = cur_pp + [j]
                this_pplen = cur_pplen + cij[cur_i+p, j+p]
                this_ppval = cur_ppval + rb[j]
                # check if the unordered set of demands covered by the same starting depot and ending demand node has been entered or not
                # if one exists: if this_pp has a smaller len, replace the existing; otherwise, do not add this_pp
                # if none exisits: add this_pp, take notes of its set signature, value and index in nextpp
                this_sig = frozenset(set([this_pp[0]-p] + this_pp[1:nlevels] + [this_pp[-1]+n+p]))
                if this_sig not in setsignext:
                    setsignext.add(this_sig)
                    setsignext_len[this_sig] = this_pplen
                    setsignext_pos[this_sig] = nextlastind
                    nextpp[nextlastind] = this_pp
                    nextpplen[nextlastind] = this_pplen
                    nextppval[nextlastind] = this_ppval 
                    nextlastind = nextlastind + 1
                else:
                    best_sig_len = setsignext_len[this_sig]
                    if this_pplen < best_sig_len:
                        setsignext_len[this_sig] = this_pplen  # update 
                        best_sig_pos = setsignext_pos[this_sig]
                        nextpp[best_sig_pos] = this_pp
                        nextpplen[best_sig_pos] = this_pplen         
                        nextppval[best_sig_pos] = this_ppval
            # close and save this partial path (if its value is positive) and add it to the global pool
            if cur_ppval > 0:
                # check if the unordered set of demands visited in this partial path has been added before; if yes, ignore; if no, add it and update the bookkeepting list
                cur_demand_set = frozenset(set([cur_pp[0]-p] + cur_pp[1:]))
                if cur_demand_set not in set_of_sets_of_demands:
                    set_of_sets_of_demands.add(cur_demand_set)
                    last_demand_node = cur_pp[-1]
                    gp.append(cur_pp + [index_of_nearest_depot[last_demand_node]])
                    gpn = gpn + 1                
        pp = nextpp
        pplen = nextpplen
        ppval = nextppval
        lastind = nextlastind        
    return(gp)

def find_all_paths_dedup_multiprocessing(rb, msm):
    pool_obj = multiprocess.Pool()
    arglist = []
    for i in range(p):
        arglist.append({'rb':rb,'msm':msm,'start_depot':i,
                        'cij':cij,'dist_to_nearest_depot':dist_to_nearest_depot,
                        'index_of_nearest_depot':index_of_nearest_depot,
                        'p':p,'n':n,'B':B,'pathcost':pathcost})
    paths_all = pool_obj.map(find_all_paths_dedup_from_a_depot, arglist)
    paths_list = [item for sublist in paths_all for item in sublist]
    return(paths_list)


def solve_setcover(timelimit=600, solverlog=False, parallel=True, plot_sol=False):  
    start_time = time.time()
    if parallel==True:
        paths = find_all_paths_dedup_multiprocessing(revenue, np.zeros(p))
    else:
        paths = find_all_paths_dedup(revenue, np.zeros(p))
    enum_time = time.time() - start_time 
    n_paths = len(paths)
    #print("Total paths: {:d}, enum time: {:.1f}".format(n_paths,enum_time))
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.all_model_types = 'cplex'
    file = open(os.path.join(ws.working_directory, "cplex.opt"), "w")
    file.write("names=no")
    file.close()
    opt.optfile = 1
    
    i_g = gdb.add_set("i", 1, "set of demand points")
    k_g = gdb.add_set("k", 1, "set of batteries")
    h_g = gdb.add_set("h", 1, "set of candidate paths")
    hs_g = gdb.add_set_dc("hs",['h','i'], "starting depot of path")
    he_g = gdb.add_set_dc("he",['h','i'], "ending depot of path")
    hi_g = gdb.add_set_dc("hi",['h','i'], "depot order incidence")
    p_g = gdb.add_parameter("p", 0, "number of depots")
    w_g = gdb.add_parameter_dc("w",['i','k'], "whereabout of drone k")
    vh_g = gdb.add_parameter_dc("vh",['h'], "value of the path")
    pathcost_g = gdb.add_parameter("pathcost",0,"fixed cost per path")
    
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
        else:
            i_g.add_record(str(i-p))
    for k in range(K):
        k_g.add_record('k'+str(k))
    for h in range(n_paths):
        h_g.add_record('h'+str(h))
    
    for h in range(n_paths):
        this_path = paths[h]
        start_depot = this_path[0]
        end_depot = this_path[-1]
        hs_g.add_record(['h'+str(h),'depot'+str(start_depot)])
        he_g.add_record(['h'+str(h),'depot'+str(end_depot)])
        this_path_v = 0
        for k in range(1,len(this_path)-1):
            i = this_path[k]
            hi_g.add_record(['h'+str(h), str(i)])
            this_path_v = this_path_v + revenue[i]
        vh_g.add_record('h'+str(h)).value = float(this_path_v)   
    p_g.add_record().value = p
    pathcost_g.add_record().value = pathcost
    for k in range(K):
        w_g.add_record(['depot'+str(w[k]),'k'+str(k)]).value = 1
    setup_time = time.time() - start_time
    opt.reslim = timelimit - setup_time
    t1 = ws.add_job_from_file(os.getcwd()+"\\"+"setcover_all.gms")
    if solverlog:
        t1.run(gams_options = opt, databases = gdb, output=sys.stdout)
    else:
        t1.run(gams_options = opt, databases = gdb)
    total_time = time.time() - start_time
    
    # print out the paths selected in solution, record the demands served
    demands_served = set()
    for h in range(n_paths):
        if t1.out_db['y'].find_record('h'+str(h)).level == 1:
            demands_served = demands_served.union(set(paths[h][1:len(paths[h])-1]))
            #print(paths[h])
    n_demands_served = len(demands_served)
    objval = t1.out_db['objval'].find_record().level
    UB = t1.out_db['ub'].find_record().value
    gap = t1.out_db['gap'].find_record().value 
    selected_paths = t1.out_db['n_path_used'].find_record().value
    modelstat = t1.out_db['ms'].find_record().value
    run_stat = {'enum_time': enum_time, 'setup_time': setup_time, 'mip_time': total_time - setup_time, 'total_time':total_time,
                'total_path':n_paths, 'selected_path': selected_paths, 'n_demands_served':n_demands_served, 'objval':objval, 'UB':UB, 'gap':gap, 'modelstat':modelstat}
    
    if plot_sol:
        # plot solution
        pmar=10
        maxx = np.max(demands[:,:,0])
        minx = np.min(demands[:,:,0])
        maxy = np.max(demands[:,:,1])
        miny = np.min(demands[:,:,1])
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.autoscale(tight=False)
        plt.axis('equal')
        plt.xlim((minx-pmar, maxx+pmar))
        plt.ylim((miny-pmar, maxy+pmar))
        fig = plt.gcf()
        ax = fig.gca()
        for j in range(p):
            plt.plot(depots[j,0], depots[j,1], 'bs')
            plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j+1), fontsize = 'medium', color='b')
            plt.text(depots[j,0]+2, depots[j,1]-2, '(' + str(ndrones[j]) + ')', fontsize='medium', color ='b')
        for i in range(n):
            #plt.plot(demands[i,:,0], demands[i,:,1], 'r-')
            plt.arrow(demands[i,0,0], demands[i,0,1], demands[i,1,0]-demands[i,0,0], demands[i,1,1]-demands[i,0,1], head_width=1.5, color='k', length_includes_head=True)
            #plt.text(demands[i,0,0], demands[i,0,1], str(i), fontsize = 'large') 
            #plt.text((demands[i,0,0] + demands[i,1,0])/2, (demands[i,0,1] + demands[i,1,1])/2, str(np.round(v[i], decimals=1)), fontsize = 'medium')
    
        selected_paths_index = [0 for i in range(int(selected_paths))]
        counter = 0
        for h in range(n_paths):
            if t1.out_db['y'].find_record('h'+str(h)).level == 1:
                selected_paths_index[counter] = h
                counter += 1
        for h in selected_paths_index:
            this_path = paths[h]
            # starting depot to first demand
            j = this_path[0]
            i = this_path[1]
            plt.arrow(depots[j,0], depots[j,1], demands[i,0,0] - depots[j,0], demands[i,0,1] - depots[j,1], color='g', ls='-', head_width=1.5, length_includes_head=True)
            if len(this_path) > 3:
                # from demand i to demand j
                for this_node in range(1,len(this_path)-2):
                    i = this_path[this_node]
                    j = this_path[this_node + 1]
                    plt.arrow(demands[i,1,0], demands[i,1,1], demands[j,0,0] - demands[i,1,0], demands[j,0,1] - demands[i,1,1], color='r', ls='-', head_width=1.5, length_includes_head=True)        
            # last demand i to ending depot j
            i = this_path[len(this_path)-2]
            j = this_path[len(this_path)-1]
            plt.arrow(demands[i,1,0], demands[i,1,1], depots[j,0] - demands[i,1,0], depots[j,1] - demands[i,1,1], color='orange', ls='-', head_width=1.5, length_includes_head=True)
        plt.savefig("setcover_sol.pdf")
    return(run_stat)

# Do not enumerate all paths, randomly generate a large number of candidate paths
def solve_setcover_h(timelimit=600, solverlog=False, total_collections = 40000, ncores = 16, parallel=True):  
    start_time = time.time()
    preload_start_time = time.time()
    paths = run_find_good_collections_of_paths_in_parallel(total_collections, ncores)
    preload_time = time.time() - preload_start_time
    n_paths = len(paths)
    print("Preload {:d} paths, preload_time: {:.1f}".format(n_paths, preload_time))
    n_paths = len(paths)
    #print("Total paths: {:d}, enum time: {:.1f}".format(n_paths,enum_time))
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.all_model_types = 'cplex'
    file = open(os.path.join(ws.working_directory, "cplex.opt"), "w")
    file.write("names=no")
    file.close()
    opt.optfile = 1
    
    i_g = gdb.add_set("i", 1, "set of demand points")
    k_g = gdb.add_set("k", 1, "set of batteries")
    h_g = gdb.add_set("h", 1, "set of candidate paths")
    hs_g = gdb.add_set_dc("hs",['h','i'], "starting depot of path")
    he_g = gdb.add_set_dc("he",['h','i'], "ending depot of path")
    hi_g = gdb.add_set_dc("hi",['h','i'], "depot order incidence")
    p_g = gdb.add_parameter("p", 0, "number of depots")
    w_g = gdb.add_parameter_dc("w",['i','k'], "whereabout of drone k")
    vh_g = gdb.add_parameter_dc("vh",['h'], "value of the path")
    pathcost_g = gdb.add_parameter("pathcost",0,"fixed cost per path")
    
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
        else:
            i_g.add_record(str(i-p))
    for k in range(K):
        k_g.add_record('k'+str(k))
    for h in range(n_paths):
        h_g.add_record('h'+str(h))
    
    for h in range(n_paths):
        this_path = paths[h]
        start_depot = this_path[0]
        end_depot = this_path[-1]
        hs_g.add_record(['h'+str(h),'depot'+str(start_depot)])
        he_g.add_record(['h'+str(h),'depot'+str(end_depot)])
        this_path_v = 0
        for k in range(1,len(this_path)-1):
            i = this_path[k]
            hi_g.add_record(['h'+str(h), str(i)])
            this_path_v = this_path_v + revenue[i]
        vh_g.add_record('h'+str(h)).value = float(this_path_v)   
    p_g.add_record().value = p
    pathcost_g.add_record().value = pathcost
    for k in range(K):
        w_g.add_record(['depot'+str(w[k]),'k'+str(k)]).value = 1
    setup_time = time.time() - start_time
    opt.reslim = timelimit - setup_time
    t1 = ws.add_job_from_file(os.getcwd()+"\\"+"setcover_all.gms")
    if solverlog:
        t1.run(gams_options = opt, databases = gdb, output=sys.stdout)
    else:
        t1.run(gams_options = opt, databases = gdb)
    total_time = time.time() - start_time
    
    # print out the paths selected in solution
    demands_served = set()
    for h in range(n_paths):
        if t1.out_db['y'].find_record('h'+str(h)).level == 1:
            demands_served = demands_served.union(set(paths[h][1:len(paths[h])-1]))
            #print(paths[h])
    n_demands_served = len(demands_served)
    objval = t1.out_db['objval'].find_record().level
    UB = t1.out_db['ub'].find_record().value
    gap = t1.out_db['gap'].find_record().value 
    selected_paths = t1.out_db['n_path_used'].find_record().value
    modelstat = t1.out_db['ms'].find_record().value
    run_stat = {'preload_time': preload_time, 'setup_time': setup_time, 'mip_time': total_time - setup_time, 'total_time':total_time,
                'total_path':n_paths, 'selected_path': selected_paths, 'n_demands_served':n_demands_served, 'objval':objval, 'UB':UB, 'gap':gap, 'modelstat':modelstat}
    return(run_stat)
   

# all_max_paths = find_max_paths_dedup(rb, msm)
# select a path from the pool that gives the maximimum marginal profit
# the demand that has a negative rb should be removed from the path
# the path's marginal profit = sum of positive demand rb - msm of the starting depot - pathcost
@njit(parallel=True)
def select_a_path(all_max_paths, rb, msm):
    negative_rb_indx = np.argwhere(rb < 0)
    npaths = len(all_max_paths)
    profit = np.zeros(npaths, dtype=np.float32)
    for k in prange(npaths):
        this_path = all_max_paths[k]
        this_path_len = len(this_path)
        profit[k] = - msm[this_path[0]] - pathcost
        for i in range(1,this_path_len-1):
            profit[k] = profit[k] + max(0, rb[this_path[i]])
    best_k = np.argmax(profit)    
    best_rb = np.max(profit)
    best_max_path = all_max_paths[best_k]
    best_max_path_demands = best_max_path[1:len(best_max_path)-1]
    nng_demands = [i for i in best_max_path_demands if i not in negative_rb_indx]
    ending_depot = index_of_nearest_depot[nng_demands[-1]]
    best_path = [best_max_path[0]] + nng_demands + [ending_depot]
    return(best_path)


'''
timelimit=300
optcr=0
maxpaths=200
subtype='sort'
plot_sol=True
'''

# Use GAMS model instances to alternately solving master and sub problems
def top_cg_0(timelimit=300, optcr=0, maxpaths=2000, look_at_most=1000, return_at_most=50, subtype='multi', preload=True, total_collections = 4000, ncores = 8, parallel = True, plot_sol=True):
    start_time = time.time()
    preload_start_time = time.time()
    preload_time = 0
    n_preload_paths = 0
    if preload:
        preload_paths = run_find_good_collections_of_paths_in_parallel(total_collections, ncores)
        preload_time = time.time() - preload_start_time
        n_preload_paths = len(preload_paths)
        print("Preload {:d} paths, preload_time: {:.1f}".format(n_preload_paths, preload_time))
    ws_master = GamsWorkspace(working_directory = os.getcwd() + "/tmp_top_master")
    gdb_master = ws_master.add_database()
    opt_master = ws_master.add_options()
    opt_master.defines["gdxfile"] = gdb_master.name
    opt_master.defines["maxpaths"] = str(maxpaths + n_preload_paths)
    opt_master.all_model_types = 'cplex'
    opt_master.reslim = timelimit
    opt_master.optcr = optcr   
    file = open(os.path.join(ws_master.working_directory, "cplex.opt"), "w")
    file.write("names=no")
    file.close()
    opt_master.optfile = 1
    i_g = gdb_master.add_set("i", 1, "set of demand points")
    p_g = gdb_master.add_parameter("p", 0, "number of depots")
    wj_g = gdb_master.add_parameter_dc("wj",['i'], "number of drones at depot j")
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
            wj_g.add_record('depot'+str(i)).value = ndrones[i]
        else:
            i_g.add_record(str(i-p))
    p_g.add_record().value = p
    
    t1_master = ws_master.add_job_from_file(os.getcwd()+"/"+"top_cg_0.gms")
    cp_master = ws_master.add_checkpoint()
    t1_master.run(checkpoint=cp_master, gams_options = opt_master, databases = gdb_master)
    mi_master = cp_master.add_modelinstance()
    hs_g = mi_master.sync_db.add_parameter_dc("hs", ['h','i'])
    hi_g = mi_master.sync_db.add_parameter_dc("hi", ['h','i'])
    vh_g = mi_master.sync_db.add_parameter_dc("vh", ['h'])
    
    mi_master.instantiate("master use lp max master_objval", 
           modifiers=[GamsModifier(hs_g), GamsModifier(hi_g), GamsModifier(vh_g)], 
           options = opt_master)
    enum_time = 0
    enum_paths = 0
    if subtype in ['mip','multi']:        
        # Setup subproblem
        ws_sub = GamsWorkspace(working_directory = os.getcwd() + "/tmp_top_sub")
        gdb_sub = ws_sub.add_database()
        opt_sub = ws_sub.add_options()
        opt_sub.defines["gdxfile"] = gdb_sub.name
        opt_sub.all_model_types = 'cplex'
        opt_sub.reslim = timelimit
        opt_sub.optcr = optcr 
        file = open(os.path.join(ws_sub.working_directory, "cplex.opt"), "w")
        file.write("names=no")
        file.close()
        opt_sub.optfile = 1
        i_g = gdb_sub.add_set("i", 1, "set of demand points")
        p_g = gdb_sub.add_parameter("p", 0, "number of depots")
        B_g = gdb_sub.add_parameter("B", 0, "battery capacity")
        pathcost_g = gdb_sub.add_parameter("pathcost", 0, "pathcost")
        v_g = gdb_sub.add_parameter_dc("v",['i'], "value of order i")
        c_g = gdb_sub.add_parameter_dc("c", ['i','i'], "distii")
        for i in range(N):
            if i < p:
                i_g.add_record('depot'+str(i))
            else:
                i_g.add_record(str(i-p))
        for i in range(N):
            for j in range(N):
                if i < p and j < p:
                    c_g.add_record(['depot'+str(i), 'depot'+str(j)]).value = cij[i,j]
                elif i < p and j >= p:
                    c_g.add_record(['depot'+str(i), str(j-p)]).value = cij[i,j]
                elif i >= p and j < p:
                    c_g.add_record([str(i-p), 'depot'+str(j)]).value = cij[i,j]
                else:
                    c_g.add_record([str(i-p), str(j-p)]).value = cij[i,j]
        for i in range(p,N):
            v_g.add_record(str(i-p)).value = float(revenue[i-p])
        B_g.add_record().value = B
        p_g.add_record().value = p
        pathcost_g.add_record().value = pathcost
        
        t1_sub = ws_sub.add_job_from_file(os.getcwd()+"/"+"top_cg_sub.gms")
        cp_sub = ws_sub.add_checkpoint()
        t1_sub.run(checkpoint=cp_sub, gams_options = opt_sub, databases = gdb_sub)
        mi_sub = cp_sub.add_modelinstance()
        rb_g = mi_sub.sync_db.add_parameter_dc("rb",['i'], "reduced benefit")
        msm_g = mi_sub.sync_db.add_parameter_dc("msm",['i'], "master start multiplier")
        mi_sub.instantiate("sub use mip max sub_objval", 
                       modifiers=[GamsModifier(rb_g), GamsModifier(msm_g)], 
                       options = opt_sub)    
        if subtype == 'multi':
            rb = np.copy(revenue)
            msm = np.zeros(p)
    elif subtype == 'sort':
        # find all maximal feasible paths for use later
        rb = np.copy(revenue)
        msm = np.zeros(p)
        enum_start_time = time.time()
        if parallel == True:
            all_max_paths = find_max_paths_dedup_multiprocessing(rb, msm)
        else:
            all_max_paths = find_max_paths_dedup(rb, msm)
        enum_time = time.time() - enum_start_time
        enum_paths = len(all_max_paths)
        #print("Total max paths: {:d} enumeration time: {:.2f}".format(enum_paths, enum_time))

            
    initialize_time = time.time() - start_time
    master_time = 0
    sub_time = 0
    n_sub_solve = 0
    find_time = 0
    hlink = np.empty((0,3), dtype=int)
    npaths = 1  # initialize
    
    # preload the initial set of paths to master

    if preload:       
        # all all paths_to_add to the master problem
        for k in range(n_preload_paths):
            this_path = preload_paths[k]
            # get new path info and update parameters for master
            this_path_start_depot = this_path[0]
            this_path_demands = this_path[1:(len(this_path)-1)]
            this_path_profit = sum(revenue[this_path_demands]) - pathcost
            for i in this_path_demands:
                hi_g.add_record(['h'+str(npaths),str(i)]).value = 1
            vh_g.add_record('h'+str(npaths)).value = this_path_profit
            hs_g.add_record(['h'+str(npaths),'depot'+str(this_path_start_depot)]).value = 1
            # Save path information for plotting
            hlink = np.append(hlink, [[npaths, this_path_start_depot, this_path[1]+p]], axis=0)
            if len(this_path_demands) > 1:
                for i in range(1,len(this_path)-2):
                    hlink = np.append(hlink, [[npaths, this_path[i] + p, this_path[i+1] + p]], axis=0)
            hlink = np.append(hlink, [[npaths, this_path[-2] + p, this_path[-1]]], axis = 0)
            npaths = npaths + 1
        # solve master and update rb and msm for next round
        tim1 = time.time()
        mi_master.solve()
        master_time += time.time() - tim1
        #print("iter: {:d} master objval: {:.3f}".format(npaths, mi_master.sync_db['master_objval'].find_record().level))
        # update the rb and msm value for the first iteration
        if subtype in ['mip','multi']:
            rb_g.clear()
            msm_g.clear()
            for i in range(n):
                rb_g.add_record(str(i)).value = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
            for j in range(p):
                msm_g.add_record('depot'+str(j)).value = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal
        if subtype in ['multi','sort']:
            for i in range(n):
                rb[i] = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
            for j in range(p):
                msm[j] = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal
        

    while npaths < maxpaths + n_preload_paths and time.time() - start_time < timelimit:
        if subtype in ['mip','switch_to_mip']:
            tim1 = time.time()
            mi_sub.solve()
            n_sub_solve += 1
            sub_time += time.time() - tim1
            subobjval = mi_sub.sync_db['sub_objval'].find_record().level
            if subobjval < 1e-5:
                break
            # get sub solution and update parameters for master
            this_vh = -pathcost
            for i in range(n):
                if mi_sub.sync_db['xi'].find_record(str(i)).level == 1:
                    hi_g.add_record(['h'+str(npaths),str(i)]).value = 1
                    this_vh = this_vh + revenue[i]
            vh_g.add_record('h'+str(npaths)).value = this_vh       
            for j in range(p):
                if mi_sub.sync_db['zj'].find_record('depot'+str(j)).level == 1:
                    hs_g.add_record(['h'+str(npaths),'depot'+str(j)]).value = 1
            # save the extra path information for plotting
            for i in range(N):
                for j in range(N):
                    if i < p and j < p:
                        continue
                    elif i < p and j >= p:
                        try:
                            val = mi_sub.sync_db['xij'].find_record(['depot'+str(i), str(j-p)]).level
                        except:
                            val = 0
                    elif i >= p and j < p:
                        try:
                            val = mi_sub.sync_db['xij'].find_record([str(i-p),'depot'+str(j)]).level
                        except:
                            val = 0
                    elif i >= p and j >= p:
                        try:
                            val = mi_sub.sync_db['xij'].find_record([str(i-p),str(j-p)]).level
                        except:
                            val = 0
                    if val == 1:
                        hlink = np.append(hlink, [[npaths, i, j]], axis=0)  

            # solve master and update parameter for sub
            tim1 = time.time()
            mi_master.solve()
            master_time += time.time() - tim1
            #print("iter: {:d} master objval: {:.3f}".format(npaths, mi_master.sync_db['master_objval'].find_record().level))
            rb_g.clear()
            msm_g.clear()
            for i in range(n):
                rb_g.add_record(str(i)).value = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
            for j in range(p):
                msm_g.add_record('depot'+str(j)).value = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal
            npaths = npaths + 1
            if subtype == 'switch_to_mip':
                print("MIP sub found a new path with rb: {:.5f}".format(subobjval))
                subtype = 'multi'  # switch back for the next iteration
                for i in range(n):
                    rb[i] = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
                for j in range(p):
                    msm[j] = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal               
        elif subtype in ['sort']:
            tim1 = time.time()
            this_path = select_a_path(all_max_paths = all_max_paths, rb = rb, msm = msm)
            sub_time += time.time() - tim1
            # get new path info and update parameters for master
            this_path_start_depot = this_path[0]
            this_path_demands = this_path[1:(len(this_path)-1)]
            this_path_profit = sum(revenue[this_path_demands]) - pathcost
            if sum(rb[this_path_demands]) - pathcost - msm[this_path_start_depot] < 1e-5:
                break
            for i in this_path_demands:
                hi_g.add_record(['h'+str(npaths),str(i)]).value = 1
            vh_g.add_record('h'+str(npaths)).value = this_path_profit
            hs_g.add_record(['h'+str(npaths),'depot'+str(this_path_start_depot)]).value = 1
            # Save path information for plotting
            hlink = np.append(hlink, [[npaths, this_path_start_depot, this_path[1]+p]], axis=0)
            if len(this_path_demands) > 1:
                for i in range(1,len(this_path)-2):
                    hlink = np.append(hlink, [[npaths, this_path[i] + p, this_path[i+1] + p]], axis=0)
            hlink = np.append(hlink, [[npaths, this_path[-2] + p, this_path[-1]]], axis = 0)
            
            # solve master and update rb and msm for next round
            tim1 = time.time()
            mi_master.solve()
            master_time += time.time() - tim1
            #print("iter: {:d} master objval: {:.3f}".format(npaths, mi_master.sync_db['master_objval'].find_record().level))
            for i in range(n):
                rb[i] = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
            for j in range(p):
                msm[j] = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal
            npaths = npaths + 1
        elif subtype == 'multi':
            find_start_time = time.time()
            paths_to_add = find_a_path_2(rb = rb, msm = msm, look_at_most=look_at_most, return_at_most = return_at_most, seed=npaths)
            find_time += time.time() - find_start_time
            n_paths_to_add = len(paths_to_add)
            # if heuristics failed to find a new path with positive rb, then run MIP subproblem
            if n_paths_to_add == 0: 
                subtype = 'switch_to_mip'
                rb_g.clear()
                msm_g.clear()
                for i in range(n):
                    rb_g.add_record(str(i)).value = rb[i]
                for j in range(p):
                    msm_g.add_record('depot'+str(j)).value = msm[j]
                continue
            # all all paths_to_add to the master problem
            for k in range(n_paths_to_add):
                this_path = paths_to_add[k]
                # get new path info and update parameters for master
                this_path_start_depot = this_path[0]
                this_path_demands = this_path[1:(len(this_path)-1)]
                this_path_profit = sum(revenue[this_path_demands]) - pathcost
                for i in this_path_demands:
                    hi_g.add_record(['h'+str(npaths),str(i)]).value = 1
                vh_g.add_record('h'+str(npaths)).value = this_path_profit
                hs_g.add_record(['h'+str(npaths),'depot'+str(this_path_start_depot)]).value = 1
                # Save path information for plotting
                hlink = np.append(hlink, [[npaths, this_path_start_depot, this_path[1]+p]], axis=0)
                if len(this_path_demands) > 1:
                    for i in range(1,len(this_path)-2):
                        hlink = np.append(hlink, [[npaths, this_path[i] + p, this_path[i+1] + p]], axis=0)
                hlink = np.append(hlink, [[npaths, this_path[-2] + p, this_path[-1]]], axis = 0)
                npaths = npaths + 1
            # solve master and update rb and msm for next round
            tim1 = time.time()
            mi_master.solve()
            master_time += time.time() - tim1
            #print("iter: {:d} master objval: {:.3f}".format(npaths, mi_master.sync_db['master_objval'].find_record().level))
            for i in range(n):
                rb[i] = revenue[i] - mi_master.sync_db['master_once'].find_record(str(i)).marginal
            for j in range(p):
                msm[j] = mi_master.sync_db['master_start'].find_record('depot'+str(j)).marginal
    
    print("npaths = {:d}".format(npaths))
    # solve the set cover MIP to get the final solution
    ws_final = GamsWorkspace(working_directory = os.getcwd() + "/tmp_top_final")
    gdb_final = ws_final.add_database()
    opt_final = ws_final.add_options()
    opt_final.defines["gdxfile"] = gdb_final.name
    opt_final.all_model_types = 'cplex'
    opt_final.reslim = timelimit
    opt_final.optcr = optcr   
    h_g = gdb_final.add_set('h',1, "set of paths")
    i_g = gdb_final.add_set("i", 1, "set of demand points")
    p_g = gdb_final.add_parameter("p", 0, "number of depots")
    wj_g = gdb_final.add_parameter_dc("wj",['i'], "number of drones at depot j")
    hs_final_g = gdb_final.add_parameter_dc("hs",['h','i'])
    hi_final_g = gdb_final.add_parameter_dc("hi",['h','i'])
    vh_final_g = gdb_final.add_parameter_dc("vh",['h'])
    for h in range(1,npaths):
        h_g.add_record('h'+str(h))
        vh_final_g.add_record('h'+str(h)).value = vh_g.find_record('h'+str(h)).value
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
            wj_g.add_record('depot'+str(i)).value = ndrones[i]
        else:
            i_g.add_record(str(i-p))
    p_g.add_record().value = p        
    for h in range(1,npaths):
        for i in range(n):
            try:
                val = hi_g.find_record(['h'+str(h),str(i)]).value
            except:
                val = 0
            hi_final_g.add_record(['h'+str(h),str(i)]).value = val
        for j in range(p):
            try:
                val = hs_g.find_record(['h'+str(h),'depot'+str(j)]).value
            except:
                val = 0
            hs_final_g.add_record(['h'+str(h),'depot'+str(j)]).value = val
    t1_final = ws_final.add_job_from_file(os.getcwd()+"/"+"top_cg_final.gms") 
    t1_final.run(gams_options = opt_final, databases = gdb_final)
    ub = mi_master.sync_db['master_objval'].find_record().level
    lb = t1_final.out_db['final_objval'].find_record().level 
    #print("final objval: {:.3f}".format(lb))
    
    #print("UB: {:.3f} LB: {:.3f} path {:d}".format(ub,lb,npaths-1))
    run_stat = {}
    run_stat['total_time'] = time.time() - start_time
    run_stat['initialize_time'] = initialize_time
    run_stat['preload_time'] = preload_time
    run_stat['n_preload_paths'] = n_preload_paths
    run_stat['enum_time'] = enum_time
    run_stat['enum_paths'] = enum_paths
    run_stat['running_time'] = run_stat['total_time'] - initialize_time
    run_stat['master_time'] = master_time
    run_stat['sub_time'] = sub_time
    run_stat['find_time'] = find_time
    run_stat['n_sub_solve'] = n_sub_solve
    run_stat['UB'] = ub
    run_stat['LB'] = lb
    if lb > 0:
        run_stat['gap'] = (ub - lb)/lb
    else:
        run_stat['gap'] = 2  # indicating infinity
    run_stat['npaths'] = npaths - 1
    run_stat['selected'] = t1_final.out_db['selected'].find_record().value
    
    if plot_sol:
        # plot solution
        pmar=10
        maxx = np.max(demands[:,:,0])
        minx = np.min(demands[:,:,0])
        maxy = np.max(demands[:,:,1])
        miny = np.min(demands[:,:,1])
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.autoscale(tight=False)
        plt.axis('equal')
        plt.xlim((minx-pmar, maxx+pmar))
        plt.ylim((miny-pmar, maxy+pmar))
        fig = plt.gcf()
        ax = fig.gca()
        for j in range(p):
            plt.plot(depots[j,0], depots[j,1], 'bs')
            plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j+1), fontsize = 'medium', color='b')
            plt.text(depots[j,0]+2, depots[j,1]-2, '(' + str(ndrones[j]) + ')', fontsize='medium', color ='b')
        for i in range(n):
            #plt.plot(demands[i,:,0], demands[i,:,1], 'r-')
            plt.arrow(demands[i,0,0], demands[i,0,1], demands[i,1,0]-demands[i,0,0], demands[i,1,1]-demands[i,0,1], head_width=1.5, length_includes_head=True)
            #plt.text(demands[i,0,0], demands[i,0,1], str(i), fontsize = 'large') 
            #plt.text((demands[i,0,0] + demands[i,1,0])/2, (demands[i,0,1] + demands[i,1,1])/2, str(np.round(v[i], decimals=1)), fontsize = 'medium')
        '''
        plt.text(minx-15, miny+20, 'Wind Vector',fontsize = 'medium', color='r')
        circle = plt.Circle((minx-4, miny+8), 10, color='r', linestyle='--', fill=False)
        ax.add_artist(circle)
        plt.plot(minx-4, miny+8, color='r', ms=10)
        plt.arrow(minx-4, miny+8, 5, 3, head_width=2, length_includes_head=True, color = 'r')
        '''        
        for h in range(1,npaths):
            if t1_final.out_db['select'].find_record('h'+str(h)).level == 1:
                this_hlink = hlink[hlink[:,0]==h]
                for hh in range(len(this_hlink)):
                    i = this_hlink[hh,1]
                    j = this_hlink[hh,2]
                    if i < p and j >= p:
                        plt.arrow(depots[i,0], depots[i,1], demands[j-p,0,0] - depots[i,0], demands[j-p,0,1] - depots[i,1], color='g', ls='-', head_width=1.5, length_includes_head=True)
                    elif i >= p and j < p:
                        plt.arrow(demands[i-p,1,0], demands[i-p,1,1], depots[j,0] - demands[i-p,1,0], depots[j,1] - demands[i-p,1,1], color='orange', ls='-', head_width=1.5, length_includes_head=True)
                    elif i >= p and j >= p:
                        plt.arrow(demands[i-p,1,0], demands[i-p,1,1], demands[j-p,0,0] - demands[i-p,1,0], demands[j-p,0,1] - demands[i-p,1,1], color='r', ls='-', head_width=1.5, length_includes_head=True)        
                    else:
                        pass
        #plt.savefig("top_cg_B250.pdf")
    
    return(run_stat)
    

'''
reslim=100
optcr=0
plot_sol = True
maxpaths=200
solverlog = True
gamsfile='top_cg'
'''

# The CG algorithm is implemented entirely in GAMS
def solve_cg(reslim=300, optcr=0, plot_sol = True, maxpaths=200, solverlog = False, gamsfile='top_cg'):
    start_time = time.time()
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.defines["maxpaths"] = str(maxpaths)
    opt.mip = 'cplex'
    opt.reslim = reslim
    opt.optcr = optcr
    
    i_g = gdb.add_set("i", 1, "set of demand points")
    k_g = gdb.add_set("k", 1, "set of batteries")
    c_g = gdb.add_parameter_dc("c", ['i','i'], "distii")
    B_g = gdb.add_parameter("B", 0, "Battery capacity")
    p_g = gdb.add_parameter("p", 0, "number of depots")
    v_g = gdb.add_parameter_dc("v",['i'], "value of order i")
    w_g = gdb.add_parameter_dc("w",['i','k'], "whereabout of drone k")
    pathcost_g = gdb.add_parameter("pathcost",0,"fixed cost per path")    
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
        else:
            i_g.add_record(str(i-p))
    for k in range(K):
        k_g.add_record('k'+str(k))
    for i in range(N):
        for j in range(N):
            if i < p and j < p:
                c_g.add_record(['depot'+str(i), 'depot'+str(j)]).value = cij[i,j]
            elif i < p and j >= p:
                c_g.add_record(['depot'+str(i), str(j-p)]).value = cij[i,j]
            elif i >= p and j < p:
                c_g.add_record([str(i-p), 'depot'+str(j)]).value = cij[i,j]
            else:
                c_g.add_record([str(i-p), str(j-p)]).value = cij[i,j]
    B_g.add_record().value = B
    p_g.add_record().value = p
    pathcost_g.add_record().value = pathcost
    
    for i in range(p,N):
        v_g.add_record(str(i-p)).value = float(revenue[i-p])
    for k in range(K):
        w_g.add_record(['depot'+str(w[k]),'k'+str(k)]).value = 1
    
    t1 = ws.add_job_from_file(os.getcwd()+"/"+gamsfile+".gms")
    if solverlog:
        t1.run(gams_options = opt, databases = gdb, output=sys.stdout)
    else:
        t1.run(gams_options = opt, databases = gdb)
    
    if t1.out_db['iter_lim_reached'].find_record().value == 1:
        print("Iteration limit reached before convergence.")
    total_time = time.time() - start_time;
            
    run_stat = {}
    run_stat['loop_time'] = t1.out_db['loop_time'].find_record().value
    run_stat['final_time'] = t1.out_db['final_time'].find_record().value
    run_stat['master_time'] = t1.out_db['master_time'].find_record().value
    run_stat['sub_time'] = t1.out_db['sub_time'].find_record().value
    run_stat['total_time'] = total_time
    run_stat['final_objval'] = t1.out_db['final_objval'].find_record().level
    run_stat['UB'] = t1.out_db['last_master_objval'].find_record().value
    run_stat['gap'] = t1.out_db['gap'].find_record().value
    run_stat['niter'] = t1.out_db['niter'].find_record().value
    run_stat['iter_lim_reached'] = t1.out_db['iter_lim_reached'].find_record().value
    run_stat['npath_selected'] = t1.out_db['npath_selected'].find_record().value
    run_stat['norders_served'] = t1.out_db['norders_served'].find_record().value
    
    if plot_sol:
        # plot solution
        pmar=10
        maxx = np.max(demands[:,:,0])
        minx = np.min(demands[:,:,0])
        maxy = np.max(demands[:,:,1])
        miny = np.min(demands[:,:,1])
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.autoscale(tight=False)
        plt.axis('equal')
        plt.xlim((minx-pmar, maxx+pmar))
        plt.ylim((miny-pmar, maxy+pmar))
        fig = plt.gcf()
        ax = fig.gca()
        for j in range(p):
            plt.plot(depots[j,0], depots[j,1], 'bs')
            plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j+1), fontsize = 'medium', color='b')
            plt.text(depots[j,0]+2, depots[j,1]-2, '(' + str(ndrones[j]) + ')', fontsize='medium', color ='b')
        for i in range(n):
            #plt.plot(demands[i,:,0], demands[i,:,1], 'r-')
            plt.arrow(demands[i,0,0], demands[i,0,1], demands[i,1,0]-demands[i,0,0], demands[i,1,1]-demands[i,0,1], head_width=1.5, length_includes_head=True)
            #plt.text(demands[i,0,0], demands[i,0,1], str(i), fontsize = 'large') 
            #plt.text((demands[i,0,0] + demands[i,1,0])/2, (demands[i,0,1] + demands[i,1,1])/2, str(np.round(v[i], decimals=1)), fontsize = 'medium')
        '''
        plt.text(minx-15, miny+20, 'Wind Vector',fontsize = 'medium', color='r')
        circle = plt.Circle((minx-4, miny+8), 10, color='r', linestyle='--', fill=False)
        ax.add_artist(circle)
        plt.plot(minx-4, miny+8, color='r', ms=10)
        plt.arrow(minx-4, miny+8, 5, 3, head_width=2, length_includes_head=True, color = 'r')
        '''
        n_paths = int(t1.out_db['npaths'].find_record().value)
        for h in range(1,n_paths+1):
            if t1.out_db['final_select'].find_record('h'+str(h)).level == 1:
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            if i < p and j  < p:
                                pass
                            elif i < p and j >= p:
                                try:
                                    v = t1.out_db['hlink'].find_record(['h'+str(h), 'depot'+str(i), str(j-p)]).value
                                except:
                                    v = 0
                                if v == 1:
                                    #plt.plot([depots[i,0], demands[j-p,0,0]], [depots[i,1], demands[j-p,0,1]], 'g-')
                                    plt.arrow(depots[i,0], depots[i,1], demands[j-p,0,0] - depots[i,0], demands[j-p,0,1] - depots[i,1], color='g', ls='-', head_width=1.5, length_includes_head=True)
                            elif i >= p and j < p:
                                try:
                                    v = t1.out_db['hlink'].find_record(['h'+str(h), str(i-p), 'depot'+str(j)]).value
                                except:
                                    v = 0
                                if v == 1:
                                    #plt.plot([demands[i-p,1,0], depots[j,0]], [demands[i-p,1,1], depots[j,1]], 'k--')
                                    plt.arrow(demands[i-p,1,0], demands[i-p,1,1], depots[j,0] - demands[i-p,1,0], depots[j,1] - demands[i-p,1,1], color='orange', ls='-', head_width=1.5, length_includes_head=True)
                            else:
                                try:
                                    v = t1.out_db['hlink'].find_record(['h'+str(h), str(i-p), str(j-p)]).value
                                except:
                                    v = 0
                                if v == 1:
                                    #plt.plot([demands[i-p,1,0], demands[j-p,0,0]], [demands[i-p,1,1], demands[j-p,0,1]], 'r-')
                                    plt.arrow(demands[i-p,1,0], demands[i-p,1,1], demands[j-p,0,0] - demands[i-p,1,0], demands[j-p,0,1] - demands[i-p,1,1], color='r', ls='-', head_width=1.5, length_includes_head=True)        
        #plt.savefig("top_cg_B250.pdf")
    return(run_stat)

        
# Wrap the gams routine in this function
def solve_gams(timelimit=600, optcr=0, plot_sol = True, solverlog=False):
    start_time = time.time()
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.mip = 'cplex'
    opt.reslim = timelimit
    opt.optcr = optcr
    file = open(os.path.join(ws.working_directory, "cplex.opt"), "w")
    file.write("names=no")
    file.close()
    opt.optfile = 1
    
    i_g = gdb.add_set("i", 1, "set of demand points")
    k_g = gdb.add_set("k", 1, "set of batteries")
    c_g = gdb.add_parameter_dc("c", ['i','i'], "distii")
    B_g = gdb.add_parameter("B", 0, "Battery capacity")
    p_g = gdb.add_parameter("p", 0, "number of depots")
    v_g = gdb.add_parameter_dc("v",['i'], "value of order i")
    w_g = gdb.add_parameter_dc("w",['i','k'], "whereabout of drone k")
    pathcost_g = gdb.add_parameter("pathcost",0,"fixed cost per path")    
    for i in range(N):
        if i < p:
            i_g.add_record('depot'+str(i))
        else:
            i_g.add_record(str(i-p))
    for k in range(K):
        k_g.add_record('k'+str(k))
    for i in range(N):
        for j in range(N):
            if i < p and j < p:
                c_g.add_record(['depot'+str(i), 'depot'+str(j)]).value = cij[i,j]
            elif i < p and j >= p:
                c_g.add_record(['depot'+str(i), str(j-p)]).value = cij[i,j]
            elif i >= p and j < p:
                c_g.add_record([str(i-p), 'depot'+str(j)]).value = cij[i,j]
            else:
                c_g.add_record([str(i-p), str(j-p)]).value = cij[i,j]
    B_g.add_record().value = B
    p_g.add_record().value = p
    pathcost_g.add_record().value = pathcost
    
    for i in range(p,N):
        v_g.add_record(str(i-p)).value = float(revenue[i-p])
    for k in range(K):
        w_g.add_record(['depot'+str(w[k]),'k'+str(k)]).value = 1
    
    t1 = ws.add_job_from_file(os.getcwd()+"/"+"top.gms")
    if solverlog:
        t1.run(gams_options = opt, databases = gdb, output=sys.stdout)
    else:
        t1.run(gams_options = opt, databases = gdb)
    run_time = time.time() - start_time
    
    run_stat = {}
    run_stat['soltime'] = t1.out_db['soltime'].find_record().value
    run_stat['runtime'] = run_time
    run_stat['objval'] = t1.out_db['objval'].find_record().level
    run_stat['UB'] = t1.out_db['UB'].find_record().value
    run_stat['gap'] = t1.out_db['gap'].find_record().value
    run_stat['modelstat'] = t1.out_db['modelstat'].find_record().value
    run_stat['drones_used'] = t1.out_db['n_used'].find_record().value
    run_stat['orders_served'] = t1.out_db['n_served'].find_record().value
    
    if plot_sol:
        # Plot solution
        pmar=10
        maxx = np.max(demands[:,:,0])
        minx = np.min(demands[:,:,0])
        maxy = np.max(demands[:,:,1])
        miny = np.min(demands[:,:,1])
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.autoscale(tight=False)
        plt.axis('equal')
        plt.xlim((minx-pmar, maxx+pmar))
        plt.ylim((miny-pmar, maxy+pmar))
        fig = plt.gcf()
        ax = fig.gca()
        for j in range(p):
            plt.plot(depots[j,0], depots[j,1], 'bs')
            plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j+1), fontsize = 'medium', color='b')
            plt.text(depots[j,0]+2, depots[j,1]-2, '(' + str(ndrones[j]) + ')', fontsize='medium', color ='b')
        for i in range(n):
            #plt.plot(demands[i,:,0], demands[i,:,1], 'r-')
            plt.arrow(demands[i,0,0], demands[i,0,1], demands[i,1,0]-demands[i,0,0], demands[i,1,1]-demands[i,0,1], head_width=1.5, length_includes_head=True)
            #plt.text(demands[i,0,0], demands[i,0,1], str(i), fontsize = 'large') 
            #plt.text((demands[i,0,0] + demands[i,1,0])/2, (demands[i,0,1] + demands[i,1,1])/2, str(np.round(v[i], decimals=1)), fontsize = 'medium')
        for k in range(K):
            if t1.out_db['y'].find_record('k'+str(k)).level == 1:        
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            if i < p and j  < p:
                                pass
                            elif i < p and j >= p:
                                if t1.out_db['x'].find_record(['depot'+str(i), str(j-p), 'k'+str(k)]).level == 1:
                                    #plt.plot([depots[i,0], demands[j-p,0,0]], [depots[i,1], demands[j-p,0,1]], 'g-')
                                    plt.arrow(depots[i,0], depots[i,1], demands[j-p,0,0] - depots[i,0], demands[j-p,0,1] - depots[i,1], color='g', ls='-', head_width=1.5, length_includes_head=True)
                            elif i >= p and j < p:
                                if t1.out_db['x'].find_record([str(i-p), 'depot'+str(j), 'k'+str(k)]).level == 1:
                                    #plt.plot([demands[i-p,1,0], depots[j,0]], [demands[i-p,1,1], depots[j,1]], 'k--')
                                    plt.arrow(demands[i-p,1,0], demands[i-p,1,1], depots[j,0] - demands[i-p,1,0], depots[j,1] - demands[i-p,1,1], color='orange', ls='-', head_width=1.5, length_includes_head=True)
                            else:
                                if t1.out_db['x'].find_record([str(i-p), str(j-p), 'k'+str(k)]).level == 1:
                                    #plt.plot([demands[i-p,1,0], demands[j-p,0,0]], [demands[i-p,1,1], demands[j-p,0,1]], 'r-')
                                    plt.arrow(demands[i-p,1,0], demands[i-p,1,1], demands[j-p,0,0] - demands[i-p,1,0], demands[j-p,0,1] - demands[i-p,1,1], color='r', ls='-', head_width=1.5, length_includes_head=True)        
        plt.show()   
        # Summarize solution
        print("{:.0f} drones are used, {:.0f} orders are served.".format(t1.out_db['n_used'].find_record().value, t1.out_db['n_served'].find_record().value))
        print("Objval = {:.2f}".format(t1.out_db['objval'].find_record().level))
        print("Solution time: {:.1f}".format(run_time))
    return(run_stat)


##### The effect of battery capacity B on problem difficulty
# For a given p = 5, K = 20, n = 30, try different values of B; record solution time, n path selected, objvalue

p = 5
K = 20
n = 30
all_B = [50,100,150,200,250]
for B in all_B:
    demands = np.empty((n,2,2))
    depots = np.empty((p,2))
    trip_dist = np.empty(n)
    
    np.random.seed(p*2022)
    for j in range(p):
        depots[j] = np.around(0+100*np.random.random(2), decimals = 10)
    
    np.random.seed(n*2021)
    for i in range(n):
        demands[i,0] = np.around(0+100*np.random.random(2), decimals = 10)
        trip_dist[i] = np.random.uniform(10,30)
        trip_angle = float(np.random.random(1)*2*np.pi)
        demands[i,1] = np.around(demands[i,0] + [trip_dist[i]*np.cos(trip_angle), trip_dist[i]*np.sin(trip_angle)], decimals = 10)
    
    distij = np.empty((n,p))
    distji = np.empty((p,n))
    distii = np.zeros((n,n))
    distjj = np.zeros((p,p))
    for j1 in range(p):
        for j2 in range(p):
            if j1 != j2:
                distjj[j1,j2] = distance_two_points(depots[j1,], depots[j2,])
    
    for i in range(n):
        for j in range(p):
            distij[i,j] = distance_two_points(demands[i,1], depots[j,])
            distji[j,i] = distance_two_points(demands[i,0], depots[j,])
    for i1 in range(n):
        for i2 in range(n):
            if i1 != i2:
                distii[i1,i2] = distance_two_points(demands[i1,1], demands[i2,0])
    
    N = p + n
    cij = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i < p and j < p:
                cij[i,j] = distjj[i,j]
            elif i < p and j >= p:
                cij[i,j] = distji[i,j-p] + trip_dist[j-p]
            elif i >= p and j < p:
                cij[i,j] = distij[i-p,j]
            elif i >= p and j >= p:
                if i != j:
                    cij[i,j] = distii[i-p, j-p] + trip_dist[j-p]

    dist_to_nearest_depot = np.zeros(n)
    index_of_nearest_depot = np.zeros(n, dtype=int)
    for i in range(n):
        dist_to_nearest_depot[i] = np.min(distij[i,:])
        index_of_nearest_depot[i] = np.argmin(distij[i,:])
    
    np.random.seed(K*p)
    w = np.random.randint(0,p,K)  # initial location of drones
    revenue = np.round(np.random.random(n)*5, decimals=2)
    pathcost = 2
    
    ndrones = [0 for i in range(p)]
    for i in range(p):
        temp = 0
        for k in range(K):
            if w[k] == i:
                temp = temp + 1
        if temp > 0:
            print("{:d} drones are at depot {:d}".format(temp,i))
        ndrones[i] = temp
    
    ps_stat = solve_setcover(plot_sol=True)
    
    file1 = open('top_ps_B.txt','a')
    # p, K, n, B, cg_obj, cg_ub, cg_gap, total_time, npaths, norders_served
    file1.writelines('{:d} {:d} {:d} {:d} {:.2f} {:.2f} {:.3f} {:.1f} {:.0f} {:.0f} {:.0f}'.
                     format(p, K, n, B, ps_stat['objval'], ps_stat['UB'], ps_stat['gap'], ps_stat['total_time'], 
                            ps_stat['total_path'], ps_stat['selected_path'], ps_stat['n_demands_served']))
    file1.writelines("\n")
    file1.close()



# Compare top_cg_0 (mip), top_cg_0 (sort parallel), setcover (parallel), pure gams
outfilename = 'top_compare_methods_p10_K30.txt'
all_p = [10]
all_K = [30]
all_n = [10,20,30,40,50]
all_B = [50, 100, 150, 200, 250]
experiment_start_time = time.time()
for p in all_p:
    for K in all_K:
        for n in all_n:
            # Generate data
            demands = np.empty((n,2,2))
            depots = np.empty((p,2))
            trip_dist = np.empty(n)
            np.random.seed(p*2022)
            for j in range(p):
                depots[j] = np.around(0+100*np.random.random(2), decimals = 10)
            np.random.seed(n*2021)
            for i in range(n):
                demands[i,0] = np.around(0+100*np.random.random(2), decimals = 10)
                trip_dist[i] = np.random.uniform(10,30)
                trip_angle = float(np.random.random(1)*2*np.pi)
                demands[i,1] = np.around(demands[i,0] + [trip_dist[i]*np.cos(trip_angle), trip_dist[i]*np.sin(trip_angle)], decimals = 10)
            distij = np.empty((n,p))
            distji = np.empty((p,n))
            distii = np.zeros((n,n))
            distjj = np.zeros((p,p))
            for j1 in range(p):
                for j2 in range(p):
                    if j1 != j2:
                        distjj[j1,j2] = distance_two_points(depots[j1,], depots[j2,])
            for i in range(n):
                for j in range(p):
                    distij[i,j] = distance_two_points(demands[i,1], depots[j,])
                    distji[j,i] = distance_two_points(demands[i,0], depots[j,])
            for i1 in range(n):
                for i2 in range(n):
                    if i1 != i2:
                        distii[i1,i2] = distance_two_points(demands[i1,1], demands[i2,0])
            N = p + n
            cij = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    if i < p and j < p:
                        cij[i,j] = distjj[i,j]
                    elif i < p and j >= p:
                        cij[i,j] = distji[i,j-p] + trip_dist[j-p]
                    elif i >= p and j < p:
                        cij[i,j] = distij[i-p,j]
                    elif i >= p and j >= p:
                        if i != j:
                            cij[i,j] = distii[i-p, j-p] + trip_dist[j-p]
            dist_to_nearest_depot = np.zeros(n)
            index_of_nearest_depot = np.zeros(n, dtype=int)
            for i in range(n):
                dist_to_nearest_depot[i] = np.min(distij[i,:])
                index_of_nearest_depot[i] = np.argmin(distij[i,:])
            # randomly distribute the starting position of drones
            np.random.seed(K*p)
            w = np.random.randint(0,p,K)  # initial location of drones
            revenue = np.round(np.random.random(n)*5, decimals=2)
            pathcost = 2
            ndrones = [0 for i in range(p)]
            for i in range(p):
                temp = 0
                for k in range(K):
                    if w[k] == i:
                        temp = temp + 1
                #if temp > 0:
                #    print("{:d} drones are at depot {:d}".format(temp,i))
                ndrones[i] = temp
            # run experiments of different battery capacity
            for B in all_B:
                print("p: {:d} K: {:d} n: {:d} B: {:.1f} Elapsed: {:.1f}".format(p, K, n, B, time.time() - experiment_start_time))
                res_top_cg_0 = top_cg_0(subtype='mip', plot_sol=False, timelimit=600)
                outfile = open(outfilename,'a')
                outfile.writelines("{:d} {:d} {:d} {:.1f}  ".format(p, K, n, B))
                outfile.writelines("{:.1f} {:.1f} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:.3f} {:.3f} {:d} {:.0f}  ".format(res_top_cg_0['total_time'], res_top_cg_0['enum_time'], res_top_cg_0['enum_paths'], res_top_cg_0['initialize_time'], res_top_cg_0['running_time'], res_top_cg_0['master_time'], res_top_cg_0['sub_time'], res_top_cg_0['UB'], res_top_cg_0['LB'], res_top_cg_0['gap'], res_top_cg_0['npaths'], res_top_cg_0['selected']))
                outfile.close()
                res_top_cg_0 = top_cg_0(subtype='sort', parallel=True, plot_sol=False, timelimit=600)
                outfile = open(outfilename,'a')
                outfile.writelines("{:.1f} {:.1f} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:.3f} {:.3f} {:d} {:.0f}  ".format(res_top_cg_0['total_time'], res_top_cg_0['enum_time'], res_top_cg_0['enum_paths'], res_top_cg_0['initialize_time'], res_top_cg_0['running_time'], res_top_cg_0['master_time'], res_top_cg_0['sub_time'], res_top_cg_0['UB'], res_top_cg_0['LB'], res_top_cg_0['gap'], res_top_cg_0['npaths'], res_top_cg_0['selected']))
                outfile.close()
                res_setcover = solve_setcover(solverlog=False, parallel=True, timelimit=600)
                outfile = open(outfilename,'a')
                outfile.writelines("{:.1f} {:.1f} {:.1f} {:d} {:.0f} {:.3f} {:.3f} {:.3f} {:.0f}  ".format(res_setcover['enum_time'], res_setcover['mip_time'], res_setcover['total_time'], res_setcover['total_path'], res_setcover['selected_path'], res_setcover['objval'], res_setcover['UB'], res_setcover['gap'], res_setcover['modelstat']))
                outfile.close()
                res_gams = solve_gams(solverlog=False, plot_sol=False, timelimit=600)
                outfile = open(outfilename, 'a')
                outfile.writelines("{:.1f} {:.3f} {:.3f} {:.3f} {:.0f} {:.0f} {:.0f} ".format(res_gams['runtime'],res_gams['objval'], res_gams['UB'], res_gams['gap'], res_gams['modelstat'], res_gams['drones_used'], res_gams['orders_served']))
                outfile.writelines("\n")
                outfile.close()

# Try topcg_0 multi on all instances
outfilename = 'topcg_ps_extra_large_instances.txt'
all_p = [20]
all_K = [200]
all_n = range(200,1001,100)
all_B = [100]
experiment_start_time = time.time()
for p in all_p:
    for K in all_K:
        for n in all_n:
            # Generate data
            demands = np.empty((n,2,2))
            depots = np.empty((p,2))
            trip_dist = np.empty(n)
            np.random.seed(p*2022)
            for j in range(p):
                depots[j] = np.around(0+100*np.random.random(2), decimals = 10)
            np.random.seed(n*2021)
            for i in range(n):
                demands[i,0] = np.around(0+100*np.random.random(2), decimals = 10)
                trip_dist[i] = np.random.uniform(10,30)
                trip_angle = float(np.random.random(1)*2*np.pi)
                demands[i,1] = np.around(demands[i,0] + [trip_dist[i]*np.cos(trip_angle), trip_dist[i]*np.sin(trip_angle)], decimals = 10)
            distij = np.empty((n,p))
            distji = np.empty((p,n))
            distii = np.zeros((n,n))
            distjj = np.zeros((p,p))
            for j1 in range(p):
                for j2 in range(p):
                    if j1 != j2:
                        distjj[j1,j2] = distance_two_points(depots[j1,], depots[j2,])
            for i in range(n):
                for j in range(p):
                    distij[i,j] = distance_two_points(demands[i,1], depots[j,])
                    distji[j,i] = distance_two_points(demands[i,0], depots[j,])
            for i1 in range(n):
                for i2 in range(n):
                    if i1 != i2:
                        distii[i1,i2] = distance_two_points(demands[i1,1], demands[i2,0])
            N = p + n
            cij = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    if i < p and j < p:
                        cij[i,j] = distjj[i,j]
                    elif i < p and j >= p:
                        cij[i,j] = distji[i,j-p] + trip_dist[j-p]
                    elif i >= p and j < p:
                        cij[i,j] = distij[i-p,j]
                    elif i >= p and j >= p:
                        if i != j:
                            cij[i,j] = distii[i-p, j-p] + trip_dist[j-p]
            dist_to_nearest_depot = np.zeros(n)
            index_of_nearest_depot = np.zeros(n, dtype=int)
            for i in range(n):
                dist_to_nearest_depot[i] = np.min(distij[i,:])
                index_of_nearest_depot[i] = np.argmin(distij[i,:])
            # randomly distribute the starting position of drones
            np.random.seed(K*p)
            w = np.random.randint(0,p,K)  # initial location of drones
            revenue = np.round(np.random.random(n)*5, decimals=2)
            pathcost = 2
            ndrones = [0 for i in range(p)]
            for i in range(p):
                temp = 0
                for k in range(K):
                    if w[k] == i:
                        temp = temp + 1
                #if temp > 0:
                #    print("{:d} drones are at depot {:d}".format(temp,i))
                ndrones[i] = temp
            # run experiments of different battery capacity
            for B in all_B:
                print("p: {:d} K: {:d} n: {:d} B: {:.1f} Elapsed: {:.1f}".format(p, K, n, B, time.time() - experiment_start_time))
                '''
                print("solving top_cg_0 (mip)")
                res_top_cg_0 = top_cg_0(subtype='mip', preload=False, timelimit=3600, plot_sol=False)
                outfile = open(outfilename,'a')
                outfile.writelines("{:d} {:d} {:d} {:.1f}   ".format(p, K, n, B))
                outfile.writelines("{:.1f} {:.1f} {:d} {:.1f} {:.1f} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:d} {:.3f} {:.3f} {:.3f} {:d} {:.0f}  ".
                                   format(res_top_cg_0['total_time'], res_top_cg_0['enum_time'], res_top_cg_0['enum_paths'], 
                                          res_top_cg_0['initialize_time'], res_top_cg_0['preload_time'], res_top_cg_0['n_preload_paths'], 
                                          res_top_cg_0['running_time'], res_top_cg_0['master_time'], 
                                          res_top_cg_0['sub_time'], res_top_cg_0['find_time'], res_top_cg_0['n_sub_solve'],
                                          res_top_cg_0['UB'], res_top_cg_0['LB'], res_top_cg_0['gap'], 
                                          res_top_cg_0['npaths'], res_top_cg_0['selected']))
                
                print("solving top_cg_0 (multi)")
                res_top_cg_0 = top_cg_0(subtype='multi', preload=False, maxpaths=7000, look_at_most=5000, return_at_most=5000, timelimit=3600, plot_sol=False)
                outfile = open(outfilename,'a')
                outfile.writelines("{:d} {:d} {:d} {:.1f}   ".format(p, K, n, B))
                outfile.writelines("{:.1f} {:.1f} {:d} {:.1f} {:.1f} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:d} {:.3f} {:.3f} {:.3f} {:d} {:.0f}   ".
                                   format(res_top_cg_0['total_time'], res_top_cg_0['enum_time'], res_top_cg_0['enum_paths'], 
                                          res_top_cg_0['initialize_time'], res_top_cg_0['preload_time'], res_top_cg_0['n_preload_paths'], 
                                          res_top_cg_0['running_time'], res_top_cg_0['master_time'], 
                                          res_top_cg_0['sub_time'], res_top_cg_0['find_time'], res_top_cg_0['n_sub_solve'],
                                          res_top_cg_0['UB'], res_top_cg_0['LB'], res_top_cg_0['gap'], 
                                          res_top_cg_0['npaths'], res_top_cg_0['selected']))

                
                print("solving top_cg_0 (multi + preload)")
                res_top_cg_0 = top_cg_0(subtype='multi', preload=True, total_collections=20000, ncores=16, maxpaths=3000, look_at_most=1000, return_at_most=1000, timelimit=3600, plot_sol=False)
                outfile.close()
                outfile = open(outfilename,'a')
                
                outfile.writelines("{:.1f} {:.1f} {:d} {:.1f} {:.1f} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:d} {:.3f} {:.3f} {:.3f} {:d} {:.0f}   ".
                                   format(res_top_cg_0['total_time'], res_top_cg_0['enum_time'], res_top_cg_0['enum_paths'], 
                                          res_top_cg_0['initialize_time'], res_top_cg_0['preload_time'], res_top_cg_0['n_preload_paths'], 
                                          res_top_cg_0['running_time'], res_top_cg_0['master_time'], 
                                          res_top_cg_0['sub_time'], res_top_cg_0['find_time'], res_top_cg_0['n_sub_solve'],
                                          res_top_cg_0['UB'], res_top_cg_0['LB'], res_top_cg_0['gap'], 
                                          res_top_cg_0['npaths'], res_top_cg_0['selected']))
                outfile.close()
                '''
                print("solving setcover parallel")
                res_setcover = solve_setcover(solverlog=False, parallel=True, timelimit=3600)
                outfile = open(outfilename,'a')
                outfile.writelines("{:d} {:d} {:d} {:.1f}   ".format(p, K, n, B))
                outfile.writelines("{:.1f} {:.1f} {:.1f} {:d} {:.0f} {:d} {:.3f} {:.3f} {:.3f} {:.0f}   ".
                                   format(res_setcover['enum_time'], res_setcover['mip_time'], res_setcover['total_time'], 
                                          res_setcover['total_path'], res_setcover['selected_path'], res_setcover['n_demands_served'], res_setcover['objval'], 
                                          res_setcover['UB'], res_setcover['gap'], res_setcover['modelstat']))
                '''
                outfile.close()

                print("solving gams")
                res_gams = solve_gams(solverlog=False, plot_sol=False, timelimit=600)
                outfile = open(outfilename, 'a')
                outfile.writelines("{:.1f} {:.3f} {:.3f} {:.3f} {:.0f} {:.0f} {:.0f}  ".format(res_gams['runtime'],res_gams['objval'], res_gams['UB'], res_gams['gap'], res_gams['modelstat'], res_gams['drones_used'], res_gams['orders_served']))
                '''
                outfile.writelines("\n")
                outfile.close()




# Experiments on larger instances
p = 10
n = 50
K = 20
B= 150

demands = np.empty((n,2,2))
depots = np.empty((p,2))
trip_dist = np.empty(n)

np.random.seed(p*2022)
for j in range(p):
    depots[j] = np.around(0+100*np.random.random(2), decimals = 10)

np.random.seed(n*2021)
for i in range(n):
    demands[i,0] = np.around(0+100*np.random.random(2), decimals = 10)
    trip_dist[i] = np.random.uniform(10,30)
    trip_angle = float(np.random.random(1)*2*np.pi)
    demands[i,1] = np.around(demands[i,0] + [trip_dist[i]*np.cos(trip_angle), trip_dist[i]*np.sin(trip_angle)], decimals = 10)

distij = np.empty((n,p))
distji = np.empty((p,n))
distii = np.zeros((n,n))
distjj = np.zeros((p,p))
for j1 in range(p):
    for j2 in range(p):
        if j1 != j2:
            distjj[j1,j2] = distance_two_points(depots[j1,], depots[j2,])

for i in range(n):
    for j in range(p):
        distij[i,j] = distance_two_points(demands[i,1], depots[j,])
        distji[j,i] = distance_two_points(demands[i,0], depots[j,])
for i1 in range(n):
    for i2 in range(n):
        if i1 != i2:
            distii[i1,i2] = distance_two_points(demands[i1,1], demands[i2,0])

N = p + n
cij = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i < p and j < p:
            cij[i,j] = distjj[i,j]
        elif i < p and j >= p:
            cij[i,j] = distji[i,j-p] + trip_dist[j-p]
        elif i >= p and j < p:
            cij[i,j] = distij[i-p,j]
        elif i >= p and j >= p:
            if i != j:
                cij[i,j] = distii[i-p, j-p] + trip_dist[j-p]

dist_to_nearest_depot = np.zeros(n)
index_of_nearest_depot = np.zeros(n, dtype=int)
for i in range(n):
    dist_to_nearest_depot[i] = np.min(distij[i,:])
    index_of_nearest_depot[i] = np.argmin(distij[i,:])

np.random.seed(K*p)
w = np.random.randint(0,p,K)  # initial location of drones
revenue = np.round(np.random.random(n)*5, decimals=2)
pathcost = 2

ndrones = [0 for i in range(p)]
for i in range(p):
    temp = 0
    for k in range(K):
        if w[k] == i:
            temp = temp + 1
    if temp > 0:
        print("{:d} drones are at depot {:d}".format(temp,i))
    ndrones[i] = temp
rb = np.copy(revenue)
msm = np.zeros(p)



solve_cg(plot_sol=True, solverlog=False)
top_cg_0(subtype='mip',plot_sol=False)
top_cg_0(subtype='sort',plot_sol=False,parallel=True)
top_cg_0(subtype='sort',plot_sol=False,parallel=False)
solve_gams(solverlog=True)
solve_setcover(solverlog=False,parallel=True)


top_cg_0(subtype='multi', look_at_most=1000, preload=False, return_at_most=1000, timelimit=600, plot_sol=True)
top_cg_0(subtype='multi', look_at_most=5000, preload=True, total_collections=20000, ncores=10, return_at_most=1000, timelimit=3600, plot_sol=False)

solve_setcover_h(timelimit=600, solverlog=False, total_collections = 2000, ncores = 10)
top_cg_0(subtype='multi', look_at_most=2000, preload=False, total_collections=2000, ncores=16, return_at_most=2000, timelimit=600, plot_sol=False)
