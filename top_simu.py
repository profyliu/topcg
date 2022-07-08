# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:59:58 2021

@author: gn0061
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot,empty_like
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from gams import *
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import time
import csv
import utm
import simplekml
from polycircles import polycircles
import pandas as pd
from shapely.geometry import Point, Polygon
from operator import itemgetter
import networkx as nx
from gams import *
from smallestenclosingcircle import make_circle
import argparse

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2


# Load Troy
pmar = 30
# Load corners from KML file
border = 0
pd.set_option('display.max_colwidth',1000000)
#kml_file = sys.argv[1]
kml_file = 'Troy.kml'
data = pd.read_table(kml_file,sep='\r\t',header=None,skip_blank_lines=False,engine='python')
foundlable = 0
for i in range(0,len(data)):
    strl = data.iloc[i].to_frame().T
    strl2 = strl.to_string()
    strlist = strl2.split()
    if strlist[2] == '<coordinates>':
        foundlable = 1
        continue
    if foundlable == 1:
        #print(strlist)
        break
minx = 1000000000000
miny = 1000000000000
maxx = 0
maxy = 0
location = list()
utmloc = dict()
for i in range(2,len(strlist)):
    location = strlist[i].split(",")
    templst = utm.from_latlon(float(location[1]),float(location[0]))
    #print(templst)
    if templst[0] <= minx:
        minx = templst[0]
    if templst[0] >= maxx:
        maxx = templst[0]
    if templst[1] <= miny:
        miny = templst[1]
    if templst[1] >= maxy:
        maxy = templst[1]
    temploc = {str(i-1):
             {'x': templst[0],
             'y': templst[1]}}
    utmloc.update(temploc)
utmnumber = templst[2]
utmletter = templst[3]
lenx = maxx - minx
leny = maxy - miny
midx = (maxx + minx)/2
midy = (maxy + miny)/2
sqlen = max(lenx,leny)
origx = midx - sqlen*(0.5 + border)
origy = midy - sqlen*(0.5 + border)

location = list()
sqloc = dict()
for i in range(3,len(strlist)):
    location = strlist[i].split(",")
    templst = utm.from_latlon(float(location[1]),float(location[0]))

    temploc = {str(i-1):
             {'x': (templst[0]-origx)/sqlen/(1+border*2)*100,
             'y': (templst[1]-origy)/sqlen/(1+border*2)*100}}
    sqloc.update(temploc)
corners = np.empty((0,2))
for i in sqloc.keys():
    corners = np.append(corners, np.array(list(sqloc[i].values())))
corners.resize((len(sqloc),2))

# Create outer boundary edges, edge_equations and polygon
edges = np.empty((0,2), dtype = int)
for i in range(len(corners)-1):
    edges = np.append(edges, np.reshape([i, i+1], (1,2)), axis = 0)
edges = np.append(edges, np.reshape([len(corners)-1, 0], (1,2)), axis = 0)
edge_equations = np.empty((len(edges),3))
for i in range(len(edges)):
    x1 = corners[edges[i,0],0]
    y1 = corners[edges[i,0],1]
    x2 = corners[edges[i,1],0]
    y2 = corners[edges[i,1],1]
    edge_equations[i,0] = y2 - y1
    edge_equations[i,1] = x1 - x2
    edge_equations[i,2] = edge_equations[i,0]*x1 + edge_equations[i,1]*y1
polygon = Polygon(corners)

hull = ConvexHull(corners)
hull_points = corners[hull.vertices]
hull_polygon = Polygon(hull_points)

canvas_corner = np.array([[0,0],[0,100],[100,0],[100,100]])
depots = np.load('Troy_vorheur_5depots.npy')
p = 5

# Plot area and depots
ax = plt.axes([0,0,1,1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.autoscale(tight=False)
plt.xlim((-30,130))
plt.ylim((-30,130))
plt.axis('equal')
# Plot the polygon
x,y = polygon.exterior.xy
plt.plot(x,y, linewidth = 1, color='gray')
for j in range(p):
    plt.plot(depots[j,0], depots[j,1], 'bs')
    plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j+1), fontsize = 'medium', color='b')
plt.show()

def toMeter(r = 10, rate = sqlen):
    meter = r/100*rate
    return meter

def Watt(kg):
    return 354.4*kg + 3967.5


# Environment configuration baseline
'''
np.random.seed(2022)
wind_spd = 0  # m/s
wind_ang = np.pi/2  # north
wind_dir = [np.cos(wind_ang), np.sin(wind_ang)]
airspeed = 15  # m/s
B = 68160  # Watt minutes
Watt_empty = Watt(0)
pathcost = 3 # dollars per charge
# Simulation configuration
period_length = 5  # how many minutes per period
n_periods = 36 # how many periods do we want to simulate
order_arrival_rate = 1  # how many arrivals per minute
order_value_min = 1  # minimum dollars per order
order_value_max = 5  # maximum dollars per order
order_weight_min = 1  # minimum kg per order
order_weight_max = 10  # maximum kg per order
order_dist_min = 500  # minimum shipment distance in meters
order_dist_max = 6000  # maximum shipment distance in meters
'''


# Environment configuration for studying wind effect
np.random.seed(2022)
wind_spd = 4  # m/s
#wind_ang = np.pi/2  # south wind north pointing 
wind_ang = -np.pi/2
wind_dir = [np.cos(wind_ang), np.sin(wind_ang)]
airspeed = 15  # m/s
B = 68160  # Watt minutes
Watt_empty = Watt(0)
pathcost = 3 # dollars per charge
# Simulation configuration
total_minutes_of_simulation = 180
period_length = 5  # how many minutes per period
n_periods = int(total_minutes_of_simulation/period_length) # how many periods do we want to simulate
order_arrival_rate = 3  # how many arrivals per minute
order_value_min = 1  # minimum dollars per order
order_value_max = 5  # maximum dollars per order
order_weight_min = 5  # minimum kg per order
order_weight_max = 5  # maximum kg per order
order_dist_min = 2000  # minimum shipment distance in meters
order_dist_max = 3000  # maximum shipment distance in meters


'''
# Generate all demands, each demand is represented by:
0 arrival_time, 
1 arrival_period, 
2 origin_x, 
3 origin_y, 
4 dest_x, 
5 dest_y, 
6 trip_dist, 
7 payload_weight, 
8 initial_value,
9 served_flag,
10 period_served
'''
np.random.seed(2021)
all_demands = np.empty((0,11))
cur_time = 0
while cur_time <= period_length*n_periods:
    interarrival_time = np.random.exponential(1/order_arrival_rate, 1)[0]
    cur_time = cur_time + interarrival_time
    if cur_time > period_length*n_periods:
        continue
    cur_period = np.floor(cur_time) // period_length
    candidate_ok = 0
    while candidate_ok == 0:
        candidate_origin = np.around(0+100*np.random.random(2), decimals = 5)
        candidate_dest = np.around(0+100*np.random.random(2), decimals = 5)
        od_dist = distance_two_points(candidate_origin, candidate_dest)
        if toMeter(od_dist) > order_dist_max or toMeter(od_dist) < order_dist_min:
            continue
        if Point(candidate_origin).within(polygon) == True and Point(candidate_dest).within(polygon) == True:                
            candidate_ok = 1
    v = np.round(np.random.uniform(order_value_min, order_value_max, size=1), decimals=2)[0]
    payload = np.round(np.random.uniform(order_weight_min, order_weight_max, size=1), decimals=1)[0]
    all_demands = np.append(all_demands, [[cur_time, cur_period, candidate_origin[0], candidate_origin[1], candidate_dest[0], candidate_dest[1], od_dist, payload, v, 0, n_periods - 1]], axis=0)
n_all_demands = len(all_demands)

# Global state of drones: location and ready time. Location is the current or next location, ready time is the time by which the drone is ready for dispatch at the location.
sys_ndrones = 20
sys_drone_depot = np.empty(sys_ndrones, dtype=int)  # drone location, depot index
sys_drone_readytime = np.empty(sys_ndrones)
for d in range(sys_ndrones):
    sys_drone_depot[d] = d % p
    sys_drone_readytime[d] = 0

# Initialize global bookkeeping variables
cur_period = 0  # scheduling time line, jumps every period_length minutes
simu_done = 0
# n_drones_beg, n_drones_dispatched, n_orders_ready, n_orders_served, n_new_orders, profit, solve_time
period_log = np.empty((0,7))
with PdfPages('top_simul_'+str(order_arrival_rate)+ '_nwsp'+str(wind_spd)+'.pdf') as pdf:
    # Simultion loop starts
    while simu_done == 0:
        # Extract ready orders and ready drones
        demands = np.empty((0,2,2))
        trip_dist = np.empty(0)
        revenue = np.empty(0)
        payload = np.empty(0)
        order_id = np.empty(0, dtype=int)  # the index of the order in the all_demands list
        for i in range(n_all_demands):
            if all_demands[i,1] <= cur_period and all_demands[i,9] == 0:  # arrived but not served
                demands = np.append(demands, [[[all_demands[i,2], all_demands[i,3]], [all_demands[i,4], all_demands[i,5]]]], axis = 0)
                trip_dist = np.append(trip_dist, [all_demands[i,6]], axis = 0)
                delay_period = cur_period - all_demands[i,1]
                revenue = np.append(revenue, [all_demands[i,8]*(1+delay_period)], axis = 0)
                payload = np.append(payload, [all_demands[i,7]], axis = 0)
                order_id = np.append(order_id, [i], axis = 0)
        n = len(demands)
        
        w = np.empty(0, dtype=int)  # depot index of drone k
        drone_id = np.empty(0, dtype=int)  # sys_drone index of drone k
        for d in range(sys_ndrones):
            if sys_drone_readytime[d] <= cur_period*period_length:
                w = np.append(w, [sys_drone_depot[d]], axis = 0)
                drone_id = np.append(drone_id, [d], axis = 0)
        K = len(drone_id)
        
        
        ndrones = [0 for i in range(p)]
        for i in range(p):
            temp = 0
            for k in range(K):
                if w[k] == i:
                    temp = temp + 1
            if temp > 0:
                print("{:d} drones are at depot {:d}".format(temp,i))
            ndrones[i] = temp
        
        # Process demand data
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
        cij = np.zeros((N,N))  # Battery consumption
        for i in range(N):
            for j in range(N):
                if i < p and j < p:
                    if i != j:
                        trip_vector = [depots[j,0] - depots[i,0], depots[j,1] - depots[i,1]]
                        cij[i,j] = Watt_empty*toMeter(distjj[i,j])/(airspeed + wind_spd*(trip_vector[0]*wind_dir[0] + trip_vector[1]*wind_dir[1])/distjj[i,j])/60
                elif i < p and j >= p:
                    trip_vector1 = [demands[j-p,0,0] - depots[i,0], demands[j-p,0,1] - depots[i,1]]
                    trip_vector2 = [demands[j-p,1,0] - demands[j-p,0,0], demands[j-p,1,1] - demands[j-p,0,1]]
                    leg1 = Watt_empty*toMeter(distji[i,j-p])/(airspeed + wind_spd*(trip_vector1[0]*wind_dir[0] + trip_vector1[1]*wind_dir[1])/distji[i,j-p])/60
                    leg2 = Watt(payload[j-p])*toMeter(trip_dist[j-p])/(airspeed + wind_spd*(trip_vector2[0]*wind_dir[0] + trip_vector2[1]*wind_dir[1])/trip_dist[j-p])/60
                    cij[i,j] = leg1 + leg2
                elif i >= p and j < p:
                    trip_vector = [depots[j,0] - demands[i-p,1,0], depots[j,1] - demands[i-p,1,1]]
                    cij[i,j] = Watt_empty*toMeter(distij[i-p,j])/(airspeed + wind_spd*(trip_vector[0]*wind_dir[0] + trip_vector[1]*wind_dir[1])/distij[i-p,j])/60         
                elif i >= p and j >= p:
                    if i != j:
                        trip_vector1 = [demands[j-p,0,0] - demands[i-p,1,0], demands[j-p,0,1] - demands[i-p,1,1]]
                        trip_vector2 = [demands[j-p,1,0] - demands[j-p,0,0], demands[j-p,1,1] - demands[j-p,0,1]]
                        leg1 = Watt_empty*toMeter(distii[i-p, j-p])/(airspeed + wind_spd*(trip_vector1[0]*wind_dir[0] + trip_vector1[1]*wind_dir[1])/distii[i-p,j-p])/60
                        leg2 = Watt(payload[j-p])*toMeter(trip_dist[j-p])/(airspeed + wind_spd*(trip_vector2[0]*wind_dir[0] + trip_vector2[1]*wind_dir[1])/trip_dist[j-p])/60
                        cij[i,j] = leg1 + leg2
       
        dist_to_nearest_depot = np.zeros(n)
        index_of_nearest_depot = np.zeros(n, dtype=int)
        for i in range(n):
            dist_to_nearest_depot[i] = np.min(distij[i,:])
            index_of_nearest_depot[i] = np.argmin(distij[i,:])
        
        # Continue only when there are both available drones and ready orders
        if K < 1 or n < 1:
            if len(period_log) == 0:
                n_new_orders = n
            else:
                n_new_orders = n - period_log[-1,2] + period_log[-1,3]
            period_log = np.append(period_log, [[K, 0, n, 0, n_new_orders, 0, 0]], axis=0)
            cur_period = cur_period + 1        
            if cur_period < n_periods or 0 < n:  
                simu_done = 0
            else:
                simu_done = 1
            continue

        
        # Run optimization: 
        timelimit=600
        solverlog=False
        parallel=True
        plot_sol=True
        
        # copy of the run_setcover function body
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
            ax = plt.axes([0,0,1,1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.autoscale(tight=False)
            plt.xlim((-30,130))
            plt.ylim((-30,130))
            plt.axis('equal')
            plt.title("Time: {:d}".format(cur_period*period_length))
            # Plot the polygon
            x,y = polygon.exterior.xy
            plt.plot(x,y, linewidth = 1, color='gray')
            fig = plt.gcf()
            ax = fig.gca()
            for j in range(p):
                plt.plot(depots[j,0], depots[j,1], 'bs')
                plt.text(depots[j,0]+2, depots[j,1]+2, 'Depot ' + str(j), fontsize = 'medium', color='b')
                plt.text(depots[j,0]+2, depots[j,1]-2, '(' + str(ndrones[j]) + ')', fontsize='medium', color ='b')
            for i in range(n):
                plt.arrow(demands[i,0,0], demands[i,0,1], demands[i,1,0]-demands[i,0,0], demands[i,1,1]-demands[i,0,1], head_width=1.5, length_includes_head=True, color= 'k')
        
            plt.text(0, 20, 'Wind Vector',fontsize = 'medium', color='r')
            scale_wind = 0.5
            circle = plt.Circle((10, 10), 15*scale_wind, color='r', linestyle='--', fill=False)
            ax.add_artist(circle)
            plt.arrow(10,10, scale_wind*wind_spd*np.cos(wind_ang), scale_wind*wind_spd*np.sin(wind_ang), head_width=2, length_includes_head=True, color = 'r')
            plt.text(25, -3, 'Period: '+str(cur_period), fontsize = 'medium', color='k')
        # Update order and drone status based on dispath solution of the current period
        # First update the order status
        n_orders_served = 0
        for i in range(n):
            if t1.out_db['eq_once'].find_record(str(i)).level == 1:
                all_demands[order_id[i], 9] = 1
                all_demands[order_id[i], 10] = cur_period
                n_orders_served = n_orders_served + 1
                
        # Then update the drone status. To do so, first get all path info, and then assign drones to path
        
        path_hs = np.empty(0, dtype=int)  # starting depot of path
        path_he = np.empty(0, dtype=int)  # ending depot of path
        path_time = np.empty(0)
        
        selected_paths_index = [0 for i in range(int(selected_paths))]
        counter = 0
        for h in range(n_paths):
            if t1.out_db['y'].find_record('h'+str(h)).level == 1:
                selected_paths_index[counter] = h
                counter += 1
        for h in selected_paths_index:
            this_path = paths[h]
            path_hs = np.append(path_hs, [this_path[0]], axis=0)
            path_he = np.append(path_he, [this_path[-1]], axis=0)
            this_path_time = 0
            # starting depot to first demand
            j = this_path[0]
            i = this_path[1]
            trip_vector1 = [demands[i,0,0] - depots[j,0], demands[i,0,1] - depots[j,1]]
            trip_vector2 = [demands[i,1,0] - demands[i,0,0], demands[i,1,1] - demands[i,0,1]]
            leg1_time = toMeter(distji[j,i])/(airspeed + wind_spd*(trip_vector1[0]*wind_dir[0] + trip_vector1[1]*wind_dir[1])/distji[j,i])/60
            leg2_time = toMeter(trip_dist[i])/(airspeed + wind_spd*(trip_vector2[0]*wind_dir[0] + trip_vector2[1]*wind_dir[1])/trip_dist[i])/60
            this_path_time = this_path_time + leg1_time + leg2_time
            if plot_sol:
                plt.arrow(depots[j,0], depots[j,1], demands[i,0,0] - depots[j,0], demands[i,0,1] - depots[j,1], color='g', ls='-', head_width=1.5, length_includes_head=True)
            if len(this_path) > 3:
                # from demand i to demand j
                for this_node in range(1,len(this_path)-2):
                    i = this_path[this_node]
                    j = this_path[this_node + 1]
                    trip_vector1 = [demands[j,0,0] - demands[i,1,0], demands[j,0,1] - demands[i,1,1]]
                    trip_vector2 = [demands[j,1,0] - demands[j,0,0], demands[j,1,1] - demands[j,0,1]]
                    leg1_time = toMeter(distii[i, j])/(airspeed + wind_spd*(trip_vector1[0]*wind_dir[0] + trip_vector1[1]*wind_dir[1])/distii[i,j])/60
                    leg2_time = toMeter(trip_dist[j])/(airspeed + wind_spd*(trip_vector2[0]*wind_dir[0] + trip_vector2[1]*wind_dir[1])/trip_dist[j])/60
                    this_path_time = this_path_time + leg1_time + leg2_time
                    if plot_sol:
                        plt.arrow(demands[i,1,0], demands[i,1,1], demands[j,0,0] - demands[i,1,0], demands[j,0,1] - demands[i,1,1], color='r', ls='-', head_width=1.5, length_includes_head=True)        
            # last demand i to ending depot j
            i = this_path[len(this_path)-2]
            j = this_path[len(this_path)-1]
            trip_vector = [depots[j,0] - demands[i,1,0], depots[j,1] - demands[i,1,1]]
            this_path_time = this_path_time + toMeter(distij[i,j])/(airspeed + wind_spd*(trip_vector[0]*wind_dir[0] + trip_vector[1]*wind_dir[1])/distij[i,j])/60         
            if plot_sol:
                plt.arrow(demands[i,1,0], demands[i,1,1], depots[j,0] - demands[i,1,0], depots[j,1] - demands[i,1,1], color='orange', ls='-', head_width=1.5, length_includes_head=True)
            path_time = np.append(path_time, [this_path_time], axis=0)
            print("path {:d}, start depot {:d}, end depot {:d}, time {:.1f}".format(h, path_hs[-1], path_he[-1], path_time[-1])) 
        if plot_sol:
            pdf.savefig()
            plt.show()
            plt.close()
            
        # Now assign paths to drones and update drone status
        n_selected_paths = len(path_hs)
        drone_num_at_depot = np.zeros(p, dtype=int)
        for h in range(n_selected_paths):
            start_depot_indx = path_hs[h]
            this_drone_indx = np.where(w == start_depot_indx)[0][drone_num_at_depot[start_depot_indx]]
            drone_num_at_depot[start_depot_indx] = drone_num_at_depot[start_depot_indx] + 1
            this_drone_global_indx = drone_id[this_drone_indx]
            sys_drone_readytime[this_drone_global_indx] = cur_period*period_length + path_time[h]
            sys_drone_depot[this_drone_global_indx] = path_he[h]
        
        if len(period_log) == 0:
            n_new_orders = n
        else:
            n_new_orders = n - period_log[-1,2] + period_log[-1,3]
        period_log = np.append(period_log, [[sum(ndrones), n_selected_paths, n, n_orders_served, n_new_orders, t1.out_db['objval'].find_record().level, run_stat['total_time']]], axis=0)
        cur_period = cur_period + 1        
        
        if cur_period < n_periods or n_orders_served < n:  
            simu_done = 0
        else:
            simu_done = 1
        
'''
0 arrival_time, 
1 arrival_period, 
2 origin_x, 
3 origin_y, 
4 dest_x, 
5 dest_y, 
6 trip_dist, 
7 payload_weight, 
8 initial_value,
9 served_flag,
10 period_served
'''
# Analyze the simulation results
# arrival_time, delayed_period, trip_dist,trip_heading, payload, initial_value
file1 = open("top_simul_result_"+str(order_arrival_rate)+ '_nwsp'+str(wind_spd)+".txt",'w')
for i in range(n_all_demands):
    if all_demands[i,9] == 1:
        trip_angle = np.arctan((all_demands[i,5] - all_demands[i,3])/(all_demands[i,4] - all_demands[i,2]))
        if all_demands[i,4] - all_demands[i,2] < 0:
            trip_angle = trip_angle + np.pi
        if trip_angle < 0:
            trip_angle = 2*np.pi + trip_angle
        file1.writelines("{:.1f}, {:.1f}, {:.1f}, {:.4f}, {:.1f}, {:.2f} \n".format(all_demands[i,0], (all_demands[i,10] - all_demands[i,1]), toMeter(all_demands[i,6])/1000, trip_angle, all_demands[i,7], all_demands[i,8]))
file1.close()

n_period_log = len(period_log)
file2 = open("top_simul_period_"+str(order_arrival_rate)+ '_nwsp'+str(wind_spd)+".txt",'w')
for i in range(n_period_log):
    file2.writelines("{:.0f}, {:.0f}, {:.0f}, {:.0f}, {:.0f}, {:.2f}, {:.1f} \n".format(period_log[i,0], period_log[i,1], period_log[i,2], period_log[i,3], period_log[i,4], period_log[i,5], period_log[i,6]))
file2.close()