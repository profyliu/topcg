# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:38:45 2022

@author: gn0061
"""

import topdefs
import numpy as np

# This is the same as defined in topdefs.py except for the plot_sol part
# plot_sol = 0 no plot, = 1 plot input, = 2 plot solution
def solve_setcover(timelimit=600, solverlog=False, parallel=True, plot_sol=0):  
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
        pmar=30
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
            if plot_sol == 1:
                dx = demands[i,1,0] - demands[i,0,0]
                dy = demands[i,1,1] - demands[i,0,1]
                seglen = distance_two_points(demands[i,0,:], demands[i,1,:])
                plt.text((demands[i,0,0] + demands[i,1,0])/2 - 4*dy/seglen, (demands[i,0,1] + demands[i,1,1])/2 + 4*dx/seglen, str(np.round(revenue[i], decimals=1)), fontsize = 'medium', va = 'center', ha = 'center')
        
        plt.text(minx-15, miny+20, 'Wind Vector',fontsize = 'medium', color='r')
        circle = plt.Circle((minx-4, miny+8), 10, color='r', linestyle='--', fill=False)
        ax.add_artist(circle)
        plt.plot(minx-4, miny+8, color='r', ms=10)
        plt.arrow(minx-4, miny+8, 5, 3, head_width=2, length_includes_head=True, color = 'r')
        
        if plot_sol == 2:
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
        else:
            plt.savefig("setcover_input.pdf")
    return(run_stat)

p=3
n = 8
K = 5
B = 150
seed = 12345
solverlog=False
parallel=True
plot_sol=1  # 1: plot input, 2 plot solution

#def run_a_random_instance(p=3, n = 10, K = 5, B = 150, seed = 2022, solverlog=False, parallel=True, plot_sol=True):
    
demands = np.empty((n,2,2))
depots = np.empty((p,2))
trip_dist = np.empty(n)

'''
np.random.seed(p*seed)
for j in range(p):
    depots[j] = np.around(10+70*np.random.random(2), decimals = 10)
'''
depots[0] = np.array([20,20])
depots[1] = np.array([30,70])
depots[2] = np.array([80,55])

np.random.seed(n*(seed-1))
for i in range(n):
    demands[i,0] = np.around(10+80*np.random.random(2), decimals = 10)
    trip_dist[i] = np.random.uniform(10,30)
    trip_angle = float(np.random.random(1)*2*np.pi)
    demands[i,1] = np.around(demands[i,0] + [trip_dist[i]*np.cos(trip_angle), trip_dist[i]*np.sin(trip_angle)], decimals = 10)

# manually shift
demands[3,0,1] -= 20
demands[3,1,1] -= 5

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

solve_setcover(solverlog=solverlog,parallel=parallel,plot_sol=plot_sol)
