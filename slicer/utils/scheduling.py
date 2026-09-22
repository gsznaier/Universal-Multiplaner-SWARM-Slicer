import math
import numpy as np
import trimesh
from trimesh import transformations
from sklearn.neighbors import KDTree
import datetime
from itertools import combinations
from tqdm import tqdm
import time
import os

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pprint

from slicer.utils import general
from slicer.utils import control
from slicer.utils import visualization as vis

#############################################
# Printer Scheduling and Collision Checking #
#############################################

def pairwise_collision_check(printer_state, printer_shells, printers, p1, p2, fast=False):
    """
    function computes if a collision has occurred between
    two CTRs
    """
    
    # get the point information for agent 1
    p1 = int(p1)
    printer_lvl = int(printer_state[2*p1])
    path_ind = int(printer_state[2*p1+1])
    
    level_ids = list(printer_shells[str(p1)]["levels"].keys())
    path_ind = int(printer_state[2*int(p1)+1])
    tip1 = printer_shells[str(p1)]["levels"][level_ids[printer_lvl]]["path"][path_ind]
    r1 = printers[str(p1)]["CTR radius"]

    # get the point information for agent 2
    p2 = int(p2)
    printer_lvl = int(printer_state[2*p2])
    path_ind = int(printer_state[2*p2+1])
    
    level_ids = list(printer_shells[str(p2)]["levels"].keys())

    path_ind = int(printer_state[2*int(p2)+1])
    tip2 = printer_shells[str(p2)]["levels"][level_ids[printer_lvl]]["path"][path_ind]
    r2 = printers[str(p2)]["CTR radius"]

    # get the local frame for agent 1
    # and convert path into local frame
    tf = printers[str(p1)]['tf']
    wtl_tf = printers[str(p1)]['inverse_tf']
    r_coeff, z_coeff, disp = printers[str(p1)]["coeffs"]
    max_z_dist = -1*printers[str(p1)]["max_z_dist"]
    min_z_dist = -1*printers[str(p1)]["min_z_dist"]
    lower_u = printers[str(p1)]["lower_u"]
    upper_u = printers[str(p2)]["upper_u"]
    
    local_1 = (wtl_tf@np.concatenate((tip1,[1]),axis=-1).T).T[:3]
    
    # convert into cylindrical coordinates
    tip_r, tip_ang, tip_z = general.calcCylindCord(local_1)
    tip_p = np.asarray([tip_r, tip_ang, tip_z])
    local_spine, reachable = control.compute_local_occupied_region(r_coeff = r_coeff,  # r coefficients for radial polynomial
                                        z_coeff = z_coeff,          # z coefficients for depth polynomial
                                        min_height = min_z_dist, 
                                        max_height = max_z_dist,
                                        tip_p=tip_p,                # current tube tip location
                                        l_bound=lower_u,            # minimum allowed control effort
                                        u_bound=upper_u,            # maximum allowed control effort
                                        resolution=r1       # sampling resolution
                                        )
    
    # convert generated ctr spline back into carteasian frame
    # convert local points back into global frame
    l_coords = general.calcCartFromCylind(local_spine)
    local_spine = np.stack(l_coords,axis=1)
    global_spine_1 = (tf@np.concatenate((local_spine,np.ones((local_spine.shape[0],1))),axis=-1).T).T
    
    # repeat process for agent 2 point
    tf = printers[str(p2)]['tf']
    wtl_tf = printers[str(p2)]['inverse_tf']
    r_coeff, z_coeff, disp = printers[str(p2)]["coeffs"]
    max_z_dist = -1*printers[str(p2)]["max_z_dist"]
    min_z_dist = -1*printers[str(p2)]["min_z_dist"]
    lower_u = printers[str(p2)]["lower_u"]
    upper_u = printers[str(p2)]["upper_u"]
    
    local_2 = (wtl_tf@np.concatenate((tip2,[1]),axis=-1).T).T[:3]
    
    tip_r, tip_ang, tip_z = general.calcCylindCord(local_2)
    tip_p = np.asarray([tip_r, tip_ang, tip_z])
    local_spine, reachable = control.compute_local_occupied_region(r_coeff = r_coeff,  # r coefficients for radial polynomial
                                        z_coeff = z_coeff,          # z coefficients for depth polynomial
                                        min_height = min_z_dist, 
                                        max_height = max_z_dist,
                                        tip_p=tip_p,                # current tube tip location
                                        l_bound=lower_u,            # minimum allowed control effort
                                        u_bound=upper_u,            # maximum allowed control effort
                                        resolution=r2       # sampling resolution
                                        )
    
    l_coords = general.calcCartFromCylind(local_spine)
    local_spine = np.stack(l_coords,axis=1)
    global_spine_2 = (tf@np.concatenate((local_spine,np.ones((local_spine.shape[0],1))),axis=-1).T).T

    # perform n-n comparison to see if any point their is a collision
    tree = KDTree(global_spine_1)
    dist, ind = tree.query(global_spine_2, k=1)
    
    # if any intersection happens return true
    return np.any(dist <= np.maximum(r1,r2))
        
    

def collision(printers, printer_shells, printer_state, force=None, fast=False):
    """
    function computes if a collision has occurred between
    any pairwise permutation of all CTRs used
    """
    check_start = datetime.datetime.now()
    collision_lambda = lambda x:(pairwise_collision_check(printer_state, printer_shells, printers, x[0], x[1], fast=fast), int(x[0]), int(x[1]))
    check_collision = np.asarray(list(map(collision_lambda, list(combinations(list(printers.keys()),2)))))
    
    return np.any(check_collision[:,0]), check_collision

def increment_printer_state(printer, printer_shells, delay_state, printer_state):
    """
    Function increments the current CTR state by one and performs the necessary
    wrap arounds when a new shell is started.
    """
    printer = int(printer)
    printer_delay = delay_state[printer]
    printer_lvl = int(printer_state[2*printer])
    path_ind = int(printer_state[2*printer+1])
    
    updated_printer_state = printer_state.copy()
    
    # Is printer is allowed to continue? (ie. no delay enforced)
    if printer_delay == 0:
        
        # increment the printer forward by 1
        updated_path_ind = path_ind+1
        
        
        level_ids = list(printer_shells[str(printer)]["levels"].keys())
        lvl_pth_len = printer_shells[str(printer)]["levels"][level_ids[printer_lvl]]["path"].shape[0]
        
        # print(f"length of lvl path {lvl_pth_len}")
        
        # Are we at the end of the path for the current level? (ie. at last level path index)
        # we are not at the end of the current level path
        if updated_path_ind < lvl_pth_len:
            updated_printer_state[2*printer+1] = updated_path_ind
                
        # we are at the end of the current level path
        else:
            
            # Have we reached the end of the full printer path?
            # We have not reached the end of the full path
            if printer_lvl < len(level_ids) - 1:
                updated_printer_state[2*printer] = printer_lvl+1
                updated_printer_state[2*printer+1] = 0
                
            # We have reached the end of the full path
            else:
                updated_printer_state[2*printer] = len(level_ids)-1
                updated_printer_state[2*printer+1] = lvl_pth_len-1
            
            

        
        
    return updated_printer_state

def get_printer_path_len(printer_shells):
    """
    Helper function that returns the length of a given
    CTR print path.
    """
    levels = printer_shells["levels"]
    return np.sum([len(levels[lvl]["path"]) for lvl in levels])
 
def compute_total_num_of_collisions(printers, printer_shells, given_schedule, fast=True):
    """
    This function computes the total number collisions that would occur
    for a given schedule and print job
    """
    max_printer_id = np.max([int(k) for k in list(printers.keys())])
    printer_state = np.zeros(2*(max_printer_id+1)).astype(int)
    
    count_collisions = 0
    pbar = tqdm(total=len(given_schedule))
    for delay in given_schedule:
        
        # update the candidate printer state based on the delay state
        collision_exists = False
        for printer in printers:
            printer_state = increment_printer_state(printer, printer_shells, delay, printer_state)
            # print(printer_state)
            collision_result = collision(printers, printer_shells, printer_state, fast=fast)
            if collision_result[0]:
                # print(f"who caused the collision? {collision_result[1]}")
                collision_exists = True
                
            
        if collision_exists == True:
            count_collisions+=1
        
        pbar.update(1)
    pbar.close()
            
    return count_collisions