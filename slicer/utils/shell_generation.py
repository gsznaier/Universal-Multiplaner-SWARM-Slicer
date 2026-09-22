import os
import cv2
import json
import time
import heapq
import decimal
import numpy as np
import open3d as o3d
import copy


import sympy
from sympy import Plane, Point3D, Point, Line3D, Line, Rational

from mesh_to_sdf import mesh_to_sdf
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.ndimage import distance_transform_edt, label, convolve
from scipy.spatial.distance import cdist

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

from slicer.utils import general
from slicer.utils import control
from slicer.utils import metrics
from slicer.utils import visualization as vis


#############################################
#     Shell Generation and Path Planning    #
#############################################
def generate_cylindrical_shell(radius, thickness, min_height, max_height, offset_theta = 0):
    
    if radius <= thickness:
        n = int(np.ceil((2*np.pi)/thickness))
        theta = np.linspace(0, 2*np.pi, num=n, endpoint=False) 
    else:
        step = thickness/radius
        
        n = int(np.ceil((2*np.pi)/step))
        theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)

    n = int(np.ceil((max_height-min_height)/thickness))
    z = np.linspace(min_height, max_height,num=n)
    theta, z = np.meshgrid(theta + offset_theta, z)
    
    x, y = radius*np.cos(theta), radius*np.sin(theta)

    
    return x, y, z
 
def generate_print_mask(mesh, print_volume, printers, printer_id, local_samples, tf, use_fast=False):
    
    coeffs = printers[str(printer_id)]["coeffs"]
    min_z_dist = -1*printers[str(printer_id)]["min_z_dist"]
    max_z_dist = -1*printers[str(printer_id)]["max_z_dist"]
    lower_u = printers[str(printer_id)]["lower_u"]
    upper_u = printers[str(printer_id)]["upper_u"]
    
    # mask points so only reachable points are considered:
    cr,ca,cz = general.calcCylindCord(local_samples[:,:3])
    c_pts = np.stack((cr,ca,cz),axis=-1)
    reachable_mask = control.check_if_reachable(coeffs[0], coeffs[1], min_z_dist, max_z_dist, c_pts, l_bound=lower_u, u_bound=upper_u)
    
    # rotate local points into global frame of reference
    global_samples = (tf@local_samples.T).T
    
    # make sure that shell is not considered beyond the print volume
    min_x,min_y,min_z = np.min(print_volume.vertices,axis=0)
    max_x,max_y,max_z = np.max(print_volume.vertices,axis=0)
    
    mask_x = (global_samples[:, 0] >= min_x) & (global_samples[:, 0] <= max_x)
    mask_y = (global_samples[:, 1] >= min_y) & (global_samples[:, 1] <= max_y)
    mask_z = (global_samples[:, 2] >= min_z) & (global_samples[:, 2] <= max_z)
    
    start = time.time()
    #TODO MAKE FAST METHOD MORE STABLE
    # Fast method: Uses open3d method. This is fast, but fails on small features due to 
    #       ray casting issues. As such it is common for it to flip the sign of an outside point
    # Slow method: Uses mesh-to-sdf method. This is very slow, but much more accurate as it considers
    #       multiple rays at once and neighboring points when considering sdf sign. This method should only
    #       be used during final path generation and not code debugging!!
    if use_fast: 
        open3d_mesh = mesh.as_open3d
        open3d_mesh.compute_vertex_normals()
        open3d_mesh.compute_triangle_normals()
        legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(legacy_mesh)
        query_points = o3d.core.Tensor(global_samples[:,:3].tolist(), dtype=o3d.core.Dtype.Float32)
        sdf = scene.compute_signed_distance(query_points).numpy()
        
    else:
        sdf = mesh_to_sdf(mesh, global_samples[:,:3], surface_point_method='sample', 
                            sign_method='normal', bounding_radius=None, 
                            scan_count=100, scan_resolution=400, 
                            sample_point_count=10000000, normal_sample_count=11)
        
    end_time = time.time()
    
    print(f"time in sec: {end_time - start}")
    print(f"time in min: {(end_time - start)/60}")
    print(f"time in hour: {(end_time - start)/(60**2)}")
    
    print_mask = np.ones(sdf.shape)

    shell_mask = print_mask*mask_x*mask_y*mask_z
    
    print_mask *= mask_x*mask_y*mask_z*(sdf <= 0)*reachable_mask
    
    return shell_mask, print_mask, sdf
    
def flatten_provided_paths(printers, printer_id, radius, contour_paths, contour_print_masks, volume):
    # algorithm takes the contours generated and
    # concatenates them into a flattened array
    # rather than a list of lists
    
    # flatten and reverse order of paths
    paths = []
    flatten_paths = []
    if len(contour_paths) >= 1:
        for i in range(0,len(contour_paths)):
             for path in contour_paths[i]:
                flatten_paths.append(path)
        paths = np.concatenate(flatten_paths,0)
    else:
        # print(paths)
        paths = np.asarray(contour_paths)
       
    # determine the path in 3d now 
    print_masks = []
    if len(paths) > 0: 
        paths = volume[paths[:,1],paths[:,0],:]

        flatten_print_masks = []
        if len(contour_print_masks) >= 1:
            for i in range(0,len(contour_print_masks)):
                if len(flatten_print_masks) > 0:
                    flatten_print_masks.append([0])
                    
                for mask in contour_print_masks[i]:
                    flatten_print_masks.append(mask)
                    
            print_masks = np.concatenate(flatten_print_masks,0)
        else:
            print_masks = np.asarray(contour_print_masks)

        for i in range(len(print_masks)):
            if len(paths) > 1:
                if np.linalg.norm(paths[i]-paths[i+1]) > radius:
                    # update print mask because now we are handling paths longer than
                    # our desired resolution
                    print_masks[i] = 0
        

    else:
        paths = []
        print_masks = []

    return np.asarray(paths), np.asarray(print_masks)

def organize_contour(contour_img):
    # work_img is physically consumed (pixels set to 0) to prevent backtracking
    work_img = (contour_img > 0).astype(np.uint8)
    h, w = work_img.shape
    
    full_path = []
    mask = []
    # Track drawn segments to prevent intersecting lines (X-crossings)
    drawn_edges = set() 

    # Clockwise offsets: N, NE, E, SE, S, SW, W, NW
    moore_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                     (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # Kernel for endpoint/tip detection to choose the best starting points
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Label clumps to ensure one is finished before moving to the next
    labeled_img, num_features = label(work_img, structure=np.ones((3,3)))

    for clump_id in range(1, num_features + 1):
        clump_mask = (labeled_img == clump_id)
        
        while np.any(work_img & clump_mask):
            # 1. START SELECTION: Identify the best "tip" of the remaining clump
            remaining_clump = work_img & clump_mask
            coords = np.argwhere(remaining_clump > 0)
            
            # Find pixels with the fewest unvisited neighbors
            neighbor_counts = convolve(remaining_clump, kernel, mode='constant')
            current_counts = neighbor_counts[coords[:, 0], coords[:, 1]]
            candidates = coords[current_counts == np.min(current_counts)]

            if full_path:
                last_c, last_r = full_path[-1]
                # Closest unvisited tip to the current pen position
                dists = np.max(np.abs(candidates - [last_r, last_c]), axis=1)
                start_node = tuple(candidates[np.argmin(dists)])
                
                # JUMP LOGIC: Only mask=1 if it is an 8-neighbor
                dr, dc = start_node[0] - last_r, start_node[1] - last_c
                is_neighbor = max(abs(dr), abs(dc)) == 1
                
                mask_val = 0
                if is_neighbor:
                    mask_val = 1
                    # CROSSING CHECK: Don't draw across an existing diagonal segment
                    if abs(dr) == 1 and abs(dc) == 1:
                        cross_edge = tuple(sorted([(last_r, start_node[1]), (start_node[0], last_c)]))
                        if cross_edge in drawn_edges:
                            mask_val = 0
                
                mask.append(mask_val)
                if mask_val == 1:
                    drawn_edges.add(tuple(sorted([(last_r, last_c), start_node])))
            else:
                start_node = tuple(candidates[0])

            full_path.append(np.array([start_node[1], start_node[0]]))
            work_img[start_node[0], start_node[1]] = 0 # Consume
            
            curr_r, curr_c = start_node
            backtrack_idx = 0 
            
            # 2. CONTINUOUS MOORE TRACE (Shell Peeling)
            while True:
                found_next = False
                for i in range(8):
                    idx = (backtrack_idx + i) % 8
                    dr, dc = moore_offsets[idx]
                    nr, nc = curr_r + dr, curr_c + dc
                    
                    # Search for unconsumed pixels within this clump
                    if 0 <= nr < h and 0 <= nc < w and work_img[nr, nc] == 1 and clump_mask[nr, nc]:
                        
                        mask_val = 1
                        # CROSSING CHECK: Prevent X-intersections
                        if abs(dr) == 1 and abs(dc) == 1:
                            cross_edge = tuple(sorted([(curr_r, nc), (nr, curr_c)]))
                            if cross_edge in drawn_edges:
                                mask_val = 0
                        
                        mask.append(mask_val)
                        if mask_val == 1:
                            drawn_edges.add(tuple(sorted([(curr_r, curr_c), (nr, nc)])))
                            
                        full_path.append(np.array([nc, nr]))
                        work_img[nr, nc] = 0 # Consume
                        
                        curr_r, curr_c = nr, nc
                        # Backtrack rotates to where we entered from to keep the search on the edge
                        backtrack_idx = (idx + 5) % 8
                        found_next = True
                        break
                
                if not found_next:
                    break

    return full_path, mask

def generate_contour_paths(print_mask, shape):
    bin_img = print_mask.reshape(shape).astype(np.uint8).copy()
    
    layers = []
    print_masks = []
        
    contour_paths, masks = organize_contour(bin_img.copy())
    print_masks.append([masks])
    layers.append([np.asarray(contour_paths)])
   
    return layers, print_masks

def add_intermediate_points(level, paths, print_masks, printer):
    coeffs = printer["coeffs"]
    l_bound = printer["lower_u"]
    u_bound = printer["upper_u"]
    max_z_dist = -1*printer["max_z_dist"]
    min_z_dist = -1*printer["min_z_dist"]
    min_r = printer["min_r"]
    scale = printer['resolution']
    
    # get printer's Transformation matrix
    tf = printer['tf']
    wtl_tf = printer['inverse_tf']
    
    update_paths = paths.copy()
    update_masks = print_masks.copy()
    
    # iterate through each point and generate the necessary intermediate points
    count = 0
    for i in range(len(paths)-1):
        n1 = paths[i,:]
        n2 = paths[i+1,:]
        
        # check if path length is too long
        path_length = np.linalg.norm(n1-n2)
        if path_length >= scale:
            t = np.linspace(0,1.0,int(path_length/scale)+1)[1:-1]
            
            # check if any points where generated
            if t.shape[0] > 0:

                # generate intermediate points
                itermediate_pts = (n2 - n1)*t.reshape(-1,1) + n1
                itermediate_pts = itermediate_pts.reshape(-1,3)
                
                # rotate into local frame of reference
                local_intermediate_points = np.concatenate((itermediate_pts,np.ones((itermediate_pts.shape[0],1))),axis=-1)
                local_intermediate_points = (wtl_tf@local_intermediate_points.T).T
                
                cr,ca,cz = general.calcCylindCord(local_intermediate_points)
                
                #update any unreachable points:
                local_reachable_pts = control.return_reachable_pts(coeffs, min_r, level, min_z_dist, max_z_dist, 
                                                           local_intermediate_points, l_bound=l_bound, u_bound=u_bound)
                
                local_reachable_pts = np.concatenate((local_reachable_pts,np.ones((local_reachable_pts.shape[0],1))),axis=-1)
                reachable_pts = ((tf@local_reachable_pts.T).T)[:,:3]
                
                pts = reachable_pts

                empty_mask = np.zeros(pts.shape[0])

                # insert points into path
                update_paths = np.insert(update_paths,[count+1],pts,axis=0)
                update_masks = np.insert(update_masks,[count+1],empty_mask,axis=0)
                
                #update based on number of points added
                count+=len(pts)
                
        # update counter based on for loop update
        count+=1

    return update_paths, update_masks

def fill_Z_layers(mesh, printers, printer_id, print_volume, use_fast=False):
    
    max_z_dist = -1*printers[str(printer_id)]["max_z_dist"] 
    min_z_dist = -1*printers[str(printer_id)]["min_z_dist"]
    tf = printers[str(printer_id)]['tf']
    wtl_tf = printers[str(printer_id)]['inverse_tf']
    resolution = printers[printer_id]['resolution']
    safety_radius = printers[printer_id]["safety radius"]
    potential_levels = printers[printer_id]["potential_levels"]
    potential_levels = [lvl for lvl in potential_levels if lvl <=safety_radius]

    local_xy_pts = []
    offset_theta = np.linspace(0,2*np.pi, len(potential_levels))
    for r_id, r in enumerate(potential_levels):

        r = np.abs(r) # special case where radius of zero gets a negative sign
        c = 2*np.pi*r
        n = int(c/resolution)
        if n == 0:
            t = np.array([0])
        else:
            t = np.linspace(0,2*np.pi, n, endpoint=True)

        t = t + offset_theta[r_id]
        x = r*np.cos(t)
        y = r*np.sin(t)
        layer_xy_pts = np.stack((x,y),axis=1)
        layer_xy_pts = np.roll(layer_xy_pts, r_id, axis=0)

        local_xy_pts.insert(0,layer_xy_pts)
        
        
    
    local_xy_pts = np.concatenate(local_xy_pts,axis=0)

    # determine z-axis direction
    bb = mesh.bounding_box.vertices
    bb = np.concatenate((bb,np.ones(len(bb)).reshape(-1,1)),axis=1)
    
    # # # rotate around x axis and remove origin
    v_tf = tf.copy()
    
    # generate z axis based on printer's coordinate system
    zs = np.append(np.arange(min_z_dist,max_z_dist,resolution),max_z_dist)
    zs = zs[::-1]


    # generate 2D slices moving along z axis of the specified printer
    local_pts = np.concatenate((np.tile(local_xy_pts,(len(zs),1)),
                                np.repeat(zs, len(local_xy_pts),axis=0).reshape(-1,1)),
                               axis=1)
    local_pts = np.concatenate((local_pts,np.ones(len(local_pts)).reshape(-1,1)),axis=1)
    
    global_pts = (v_tf@local_pts.T).T
    reachable_local_points = (wtl_tf@global_pts.T).T
    shell_mask, print_mask, sdf = generate_print_mask(mesh, print_volume, 
                                                      printers, printer_id, 
                                                      reachable_local_points, tf, use_fast=use_fast)
    deconflict_mask = np.ones(len(global_pts))
    global_pts = global_pts[:,:3]
    for r in potential_levels:
        pr, pt, pz = general.calcCylindCord(reachable_local_points[:,:3])
        
        inds_to_consider = np.logical_and((pr > np.abs(r)-resolution/2),(pr < np.abs(r)+resolution/2))
        sub_global_pts = global_pts[inds_to_consider]

        radius_index = np.where(printers[str(printer_id)]['potential_levels']==r)[0]

        sub_deconflict_mask = printer_shell_deconflicting(sub_global_pts, printers, printer_id, radius_index).copy()
        deconflict_mask[inds_to_consider]=sub_deconflict_mask
 
    
    if np.any(print_mask==1):
        first_ind = np.argwhere(print_mask==1).flatten()[0]
        last_ind = np.argwhere(print_mask==1).flatten()[-1]
        print(first_ind)
        print(last_ind)
        global_pts = global_pts[first_ind:last_ind+1,:]
        shell_mask = shell_mask[first_ind:last_ind+1]
        print_mask = print_mask[first_ind:last_ind+1]
        sdf = sdf[first_ind:last_ind+1]
        
    else:
        global_pts = []
        shell_mask = []
        print_mask = []
        sdf = []
        
    print_masks = []
    if len(global_pts) > 1:
        paths = [global_pts[0,:3].reshape(1,-1)]
        for i in range(len(global_pts)-1):
            p1 = global_pts[i,:3]
            p2 = global_pts[i+1,:3]
            
            if print_mask[i]==print_mask[i+1]==1:
                m = 1
            else:
                m = 0
                
            edge = np.asarray([p1, p2])
            edge_mask = np.asarray([m])
            edge, edge_mask = add_intermediate_points(safety_radius, edge, edge_mask, printers[printer_id])
            paths.append(edge[1:,:])
            print_masks.append(edge_mask)
        
        if len(paths) > 1:
            paths = np.concatenate(paths,axis=0)
            print_masks = np.concatenate(print_masks,axis=0)
        else:
            paths = np.asarray(paths)
            print_masks = np.asarray(print_masks)
        
    else:
        if len(global_pts) == 0:
            paths = []
            print_masks = []
        else:
            paths = global_pts
            
        paths = np.asarray(paths)
        print_masks = np.asarray(print_masks)

    return paths, print_masks, global_pts, sdf, shell_mask
              
def generate_printer_shell_using_contour_paths(mesh, print_volume, printers, printer_id, radius_id, use_fast=False):

    radius = printers[str(printer_id)]["potential_levels"][radius_id]
    max_radius = printers[str(printer_id)]["max_r"]
    min_radius = printers[str(printer_id)]["min_r"]
    max_z_dist = -1*printers[str(printer_id)]["max_z_dist"]
    min_z_dist = -1*printers[str(printer_id)]["min_z_dist"]
    scale = printers[str(printer_id)]['resolution']
    shell_method = printers[str(printer_id)]["shell_method"]
    boundary_set = printers[str(printer_id)]["boundary set"]
    
    # get printer's Transformation matrix
    tf = printers[str(printer_id)]['tf']
    wtl_tf = printers[str(printer_id)]['inverse_tf']
    
    # generate local shell and convert it into samples of shape n x 3
    # add a spiral offset to prevent a single seam from occuring
    offset_theta = np.linspace(0,2*np.pi, len(printers[str(printer_id)]["potential_levels"]))[radius_id]
    x, y, z = generate_cylindrical_shell(radius, scale, min_z_dist, max_z_dist, offset_theta)
    
    shell_shape = x.shape
    local_samples  = np.asarray([x.reshape(-1), y.reshape(-1), z.reshape(-1), np.ones(z.reshape(-1).shape[0])]).T

    shell_mask, print_mask, sdf = generate_print_mask(mesh, print_volume, printers, printer_id, local_samples, tf, use_fast=use_fast)
    
    # rotate local points into global frame of reference
    global_samples = (tf@local_samples.T).T
    
    deconflict_mask = printer_shell_deconflicting(global_samples, printers, printer_id, radius_id)
    
    print_mask *= deconflict_mask
    

    paths, print_masks = generate_contour_paths(print_mask, x.shape)

    volume = global_samples[:,:3].reshape((shell_shape[0], shell_shape[1], 3))
    paths, print_masks = flatten_provided_paths(printers, printer_id, radius, paths, print_masks, volume)
           
    print("finished making shell")
    
    
    # if last shell is populated with more than one point 
    # we want to ensure that the printer goes from
    # innermost to outermost to avoid getting stuck
    if len(paths) > 1:
        if min_radius == radius:
            local_paths = np.concatenate((paths.reshape(-1,3), np.ones(paths.shape[0]).reshape(-1,1)),axis=1)
            local_paths = (wtl_tf@local_paths.T).T[:,:3]
            pr,pt, pz = general.calcCylindCord(local_paths)
            points = np.stack((pr,pt,pz),axis=1)
            inds = np.lexsort((points[:,0],points[:,1],points[:,2]))
            paths = paths[inds]
            
            # Now we need to update the mask between points
            # if any edges are through empty space set the mask value to zero
            edges_to_check = []
            edge_index = []
            edge_masks = np.zeros(paths.shape[0]-1)
            for i in range(len(paths)-1):
                n1 = paths[i,:]
                n2 = paths[i+1,:]
                
                # check if path length is too long
                path_length = np.linalg.norm(n1-n2)
                if path_length >= scale:
                    t = np.linspace(0,1.0,int(path_length/scale)+1)[1:-1]
                    
                    # check if any points where generated
                    if t.shape[0] > 0:
                        # generate intermediate points
                        itermediate_pts = (n2 - n1)*t.reshape(-1,1) + n1
                        itermediate_pts = itermediate_pts.reshape(-1,3)
                        edge_candidates = itermediate_pts
                        
                    # no points were generated but we want to check if edge ends
                    # are in free space or not
                    else:
                        edge_candidates = paths[i:i+1,:]
                else:
                    edge_candidates = paths[i:i+1,:]
                    
                edges_to_check.append(edge_candidates)
                edge_index.append(i*np.ones(len(edge_candidates)))
                
            edge_candidates = np.concatenate(edges_to_check, axis=0)    
            edge_index = np.concatenate(edge_index, axis=0).astype(int)
            

            #TODO MAKE FAST METHOD MORE STABLE
            # Fast method: Uses open3d method. This is fast, but fails on small features due to 
            #       ray casting issues. As such it is common for it to flip the sign of an outside point
            # Slow method: Uses mesh-to-sdf method. This is very slow, but much more accurate as it considers
            #       multiple rays at once and neighboring points when considering sdf sign. This method should only
            #       be used during final path generation and not code debugging!!
            if use_fast: 
                legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d)
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(legacy_mesh)
                query_points = o3d.core.Tensor(edge_candidates.tolist(), dtype=o3d.core.Dtype.Float32)
                sdf = scene.compute_signed_distance(query_points).numpy()
                
            else:
                sdf = mesh_to_sdf(mesh, edge_candidates, surface_point_method='sample', 
                                    sign_method='normal', bounding_radius=None, 
                                    scan_count=100, scan_resolution=400, 
                                    sample_point_count=10000000, normal_sample_count=11)
                
            mask_inds = np.unique(edge_index[sdf<=0])
            edge_masks[mask_inds] = 1

                
            print_masks = edge_masks
        
    if len(paths) > 1: 
        print(f"generated path length: {print_masks.shape}")

    schedule = []
    return global_samples, sdf, shell_mask, paths, print_masks, schedule, shell_shape
  
def generate_shells(mesh, print_volume, printers, units, desired_time=1, desired_velocity=0.5, use_fast=False, dirname = "", add_safety=True):
    
    # Build the print shell for each CTR
    shells = {}
    shell_samples = {}
    for printer in printers.items():

        shells[printer[0]] = {"o": printer[1]['origin'], "n": printer[1]['normal'], 'levels': {}}
        shell_samples[printer[0]] = {'levels': {}}
        
        # store information for print job
        # NOTE PYTHON JSON IS SILLY AND FORCES NP ARRAYS TO BE A LIST 
        printer_info = printers[str(printer[0])]
        printer_job = {}
        printer_job["printer info"] = {'printer_id': printer_info['printer_id'],
                                       'origin': printer_info['origin'].tolist(),
                                       'normal': printer_info['normal'].tolist(),
                                       'resolution': printer_info['resolution'],
                                       'CTR radius': printer_info['CTR radius'],
                                       'safety radius': printer_info['safety radius'],
                                       'coeffs': [printer_info['coeffs'][0].tolist(),
                                                  printer_info['coeffs'][1].tolist(),
                                                  printer_info['coeffs'][2].tolist()],
                                       'shell_method': printer_info['shell_method'],
                                       'color': printer_info['color'],
                                       'lower_u': printer_info['lower_u'],
                                       'upper_u': printer_info['upper_u'],
                                       'boundary set': [(b[0].tolist(),b[1].tolist())  for b in printer_info['boundary set']],
                                       'max_r': printer_info['max_r'],
                                       'min_r': printer_info['min_r'],
                                       'max_z_dist': printer_info['max_z_dist'],
                                       'min_z_dist': printer_info['min_z_dist'],
                                       'potential_levels': printer_info['potential_levels'].tolist(),
                                       'tf': printer_info['tf'].tolist(),
                                       'inverse_tf': printer_info['inverse_tf'].tolist(),
                                       }
        printer_job["levels"] ={}
        printer_job["shell info"] ={}

        max_radius = printer[1]["max_r"]
        min_radius = printer[1]["min_r"]
        levels = printer[1]["potential_levels"]
 
        # loop through each potential level and try to generate a shell
        for lvl_id, level in enumerate(levels):
            if (level > printer[1]["safety radius"] and add_safety) or (not add_safety):
                print(f"printer: {printer[0]} current level: {level}")
                
                # if printer[1]["planning_method"] == "contour":
                samples, sdf, shell_mask, path, print_mask, schedule, shape = generate_printer_shell_using_contour_paths(mesh, print_volume, 
                                                                                                                        printers, int(printer[0]), 
                                                                                                                        lvl_id, use_fast=use_fast)
                # connect previous layer to current layer
                # add layer to print job
                if len(path) > 0:
                    if str(levels[lvl_id-1]) in shells[printer[0]]['levels'].keys():
                        if len(shells[printer[0]]['levels'][str(levels[lvl_id-1])]["path"]) > 0:
                            last_point = shells[printer[0]]['levels'][str(levels[lvl_id-1])]["path"][-1]

                            transition_path = np.asarray([last_point, path[0]])
                            
                            transition_mask = np.asarray([0])
                            transition_path, transition_mask = add_intermediate_points(levels[lvl_id-1], transition_path, transition_mask, printers[printer[0]])

                            path = np.insert(path, 0, transition_path[:-1], axis=0)
                            print_mask = np.insert(print_mask, 0, transition_mask, axis=0)
                            

                    # save data
                    shells[printer[0]]['levels'][str(level)] = {'path':path, 'mask': print_mask}
                    shell_samples[printer[0]]['levels'][str(level)] = {'samples': samples, 'sdf':sdf, 'shell_mask': shell_mask}
                    
                    printer_job["levels"][str(level)] = {'path':path.tolist(), 'mask': print_mask.tolist()}
                    printer_job["shell info"][str(level)] = {'samples': samples.tolist(), 'sdf':sdf.tolist(), 'shell_mask': shell_mask.tolist()}
                    # pprint.pp(printer_job)
                filename = os.path.join(dirname,'printer_'+str(printer[0])+'.json')
                print(f"saving file: {filename}")
                out_file = open(filename,"w")
                #print(printer_job)
                json.dump(printer_job, out_file, indent = 4)
                out_file.close()


            # terminate early if we start checking levels under the safety radius
            if (level <= printer[1]["safety radius"] and add_safety):
                break

        # add safety shells if requested
        if add_safety:
            
            path, print_mask, samples, sdf, shell_mask = fill_Z_layers(mesh, printers, str(printer[0]), print_volume, use_fast=use_fast)
            
            if len(path) > 0:

                if len(list(shells[printer[0]]['levels'].keys())) > 0:                 

                    print(f"first point: {path[0]}")
                    print(list(shells[printer[0]]['levels'].keys()))
                    level_key = list(shells[printer[0]]['levels'].keys())[-1]
                    print(f"level key: {level_key}")
                    print(f"printer id: {str(printer[0])}")
                    last_point = shells[str(printer[0])]['levels'][str(level_key)]["path"][-1]
                    
                    
                    transition_path = np.asarray([last_point, path[0]])
                    transition_mask = np.asarray([0])
                    transition_path, transition_mask = add_intermediate_points(float(level_key), transition_path, transition_mask, printers[printer[0]])

                    path = np.insert(path, 0, transition_path[:-1], axis=0)
                    print_mask = np.insert(print_mask, 0, transition_mask, axis=0)
                
                # save data
                shells[printer[0]]['levels'][str(0.)] = {'path':path, 'mask': print_mask}
                shell_samples[printer[0]]['levels'][str(0.)] = {'samples': samples, 'sdf':sdf, 'shell_mask': shell_mask}
                
                printer_job["levels"][str(0.)] = {'path':path.tolist(), 'mask': print_mask.tolist()}
                printer_job["shell info"][str(0.)] = {'samples': samples.tolist(), 'sdf':sdf.tolist(), 'shell_mask': shell_mask.tolist()}
                
                filename = os.path.join(dirname,'printer_'+str(printer[0])+'.json')
                print(f"saving file: {filename}")
                out_file = open(filename,"w")
                json.dump(printer_job, out_file, indent = 4)
                out_file.close()
        
        # start at origin and return to origin after finishing print job
        if len(shells[printer[0]]['levels']) > 0:
            filename = os.path.join(dirname,'printer_'+str(printer[0])+'.json')
            print(f"saving file: {filename}")
            out_file = open(filename,"w")
            json.dump(printer_job, out_file, indent = 4)
            out_file.close()

        else:
            print(f"WARNING!! printer has no levels stored!")
            print("adding a single level and origin point to avoid scheduling problems")
            print(f"failure Printer!: {printer[0]}")
            
            #TODO check if we can replace with printer_job["printer info"][...] instead
            origin_r =  printers[str(printer[0])]['min_r']
            origin_z = -1*printers[str(printer[0])]['min_z_dist']
            coeffs = printers[str(printer[0])]["coeffs"]
            l_bound = printers[str(printer[0])]["lower_u"]
            u_bound = printers[str(printer[0])]["upper_u"]
            max_z_dist = -1*printers[str(printer[0])]["max_z_dist"]
            min_z_dist = -1*printers[str(printer[0])]["min_z_dist"]
            min_r = printers[str(printer[0])]["min_r"]
            tf = printers[str(printer[0])]['tf']
    
            # situate the origin in global coordinates and ensure that point is reachable
            origin_x,origin_y,origin_z = general.calcCartFromCylind(np.asarray([origin_r,0,origin_z]).reshape(1,-1))
            
            
            local_origin_pt = np.array([origin_x[0],origin_y[0],origin_z[0],1]).reshape(1,-1)
            print(local_origin_pt.shape)
            local_reachable_pts = control.return_reachable_pts(coeffs, min_r, level, min_z_dist, max_z_dist, 
                                                        local_origin_pt, l_bound=l_bound, u_bound=u_bound)
            local_reachable_pts = np.concatenate((local_reachable_pts,np.ones((local_reachable_pts.shape[0],1))),axis=-1)
            reachable_pts = ((tf@local_reachable_pts.T).T)[:,:3]
            
            print(reachable_pts.shape)
            
            # add origin to shell json
            shells[str(printer[0])]['levels'][str(0.)] = {'path':reachable_pts, 'mask': np.array([])}
            shell_samples[str(printer[0])]['levels'][str(0.)] = {'samples': reachable_pts, 'sdf':np.zeros(1), 'shell_mask': np.array([])}
            
            # add origin to print job json since its not in the object we set to sdf to 1 the shell mask to zero
            printer_job["levels"][str(0.)] = {'path':reachable_pts.tolist(), 'mask': []}
            printer_job["shell info"][str(0.)] = {'samples': reachable_pts.tolist(), 'sdf':[1], 'shell_mask': [0]}
            print("resulting print job!")
            print(printer_job)
            
            # 
            filename = os.path.join(dirname,'printer_'+str(printer[0])+'.json')
            print(f"saving file: {filename}")
            out_file = open(filename,"w")
            json.dump(printer_job, out_file, indent = 4)
            out_file.close()
            
        # input("waiting for human input before continuing!")

    return shells, shell_samples

def print_job_statistics(printers, shells, units, desired_time=1, desired_velocity=0.5):
    # initialize statistics regarding percentage printed, path lengths and distance
    # traveled for the CTRs
    total_percent_on = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_percent_off = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_paths_len = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled_on = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled_off = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    desired_time = desired_time*60*60 # time in seconds

    # generate statistics
    for printer in printers:
        levels = shells[printer]['levels']
        for level in levels:
            path = shells[printer]['levels'][level]["path"]
            print_mask = shells[printer]['levels'][level]["mask"]
            print(f"printer {printer}")
            print(f"level {level}")
            print(f"path shape: {path.shape}")
            print(f"mask shape: {print_mask.shape}")
            percent_time_on, percent_time_off, dist_travelled, dist_travelled_on, dist_travelled_off = metrics.compute_path_print_efficiency(path, print_mask)
            total_percent_on[int(printer[0])]+=percent_time_on
            total_percent_off[int(printer[0])]+=percent_time_off
            total_paths_len[int(printer[0])]+=len(print_mask)
            total_dist_travelled[int(printer[0])]+=dist_travelled
            total_dist_travelled_on[int(printer[0])]+=dist_travelled_on
            total_dist_travelled_off[int(printer[0])]+=dist_travelled_off

    print(f"check total path lengths: {total_paths_len}")

    print(f"Desired print time in seconds: {desired_time}")
    print(f"Desired Velocity cm/sec: {desired_velocity}")
    print(f"Percentage printers spent on per printer: {np.around(total_percent_on/total_paths_len,3)}")
    print(f"Percentage printers spent off per printer: {np.around(total_percent_off/total_paths_len,3)}")
    print(f"total distance travelled per printer {np.around(total_dist_travelled*units,3)} in cm")
    print(f"total distance travelled per printer {np.around(total_dist_travelled_on*units,3)} in cm")
    print(f"total distance travelled per printer {np.around(total_dist_travelled_off*units,3)} in cm")
    print(" ")
    print(f"estimated velocity required to print within specified time: {np.around((total_dist_travelled*units)/desired_time,3)} cm/sec")
    print(f"mean velocity required to print within specified time: {np.around(np.mean((total_dist_travelled*units)/desired_time),3)} cm/sec")
    print(f"max velocity required to print within specified time: {np.around(np.max((total_dist_travelled*units)/desired_time),3)} cm/sec")
    print(f"min velocity required to print within specified time: {np.around(np.min((total_dist_travelled*units)/desired_time),3)} cm/sec")
    print(" ")
    print(f"estimated time required to print using specified velocity: {np.around((total_dist_travelled*units)/desired_velocity,3)} sec")
    print(f"mean time required to print using specified velocity: {np.around(np.mean((total_dist_travelled*units)/desired_velocity),3)} sec")
    print(f"max time required to print using specified velocity: {np.around(np.max((total_dist_travelled*units)/desired_velocity),3)} sec")
    print(f"min time required to print using specified velocity: {np.around(np.min((total_dist_travelled*units)/desired_velocity),3)} sec")
    print(" ")
    print(f"estimated time required to print using specified velocity: {np.around((total_dist_travelled*units)/desired_velocity),3/60} min")
    print(f"mean time required to print using specified velocity: {np.around(np.mean((total_dist_travelled*units)/desired_velocity),3)/60} min")
    print(f"max time required to print using specified velocity: {np.around(np.max((total_dist_travelled*units)/desired_velocity),3)/60} min")
    print(f"min time required to print using specified velocity: {np.around(np.min((total_dist_travelled*units)/desired_velocity),3)/60} min")
    print(" ")
    print(f"estimated time required to print using specified velocity: {np.around((total_dist_travelled*units)/desired_velocity,3)/(60*60)} hour")
    print(f"mean time required to print using specified velocity: {np.around(np.mean((total_dist_travelled*units)/desired_velocity),3)/(60*60)} hour")
    print(f"max time required to print using specified velocity: {np.around(np.max((total_dist_travelled*units)/desired_velocity),3)/(60*60)} hour")
    print(f"min time required to print using specified velocity: {np.around(np.min((total_dist_travelled*units)/desired_velocity),3)/(60*60)} hour")
    print(" ")

def printer_shell_deconflicting(samples, printers, printer_id, radius_id, epsilon=1e-4):
    overall_mask = np.ones(samples.shape[0])
    
    printer_r = printers[str(printer_id)]["potential_levels"][radius_id]
    self_o = printers[str(printer_id)]["origin"]
    self_n = printers[str(printer_id)]["normal"]
    self_max_z = -1*printers[str(printer_id)]["max_z_dist"]
    self_min_z = -1*printers[str(printer_id)]["min_z_dist"]
    self_a = self_min_z * self_n + self_o
    self_b = self_max_z * self_n + self_o

    for key in printers.keys():

        # prevent self checking (ie. only compare to other printers)
        if str(key) != str(printer_id):

            # get printer information
            printer = printers[str(key)]
            o = printer["origin"]
            n = printer["normal"]
            max_z_dist = -1*printer["max_z_dist"]
            min_z_dist = -1*printer["min_z_dist"]

            potential_radii = printer["potential_levels"][printer["potential_levels"] <= printer_r]
            
            # get the radius of other printer shells that will cover inner points
            for radius in potential_radii:
                a = min_z_dist*n + o
                b = max_z_dist*n + o
                ab = b - a
                
                # compute the projection point for each sample point onto the other printer shells
                ap = samples[:,:3] - a[None,...]
                proj = np.dot(ap,ab)/np.dot(ab,ab)
                result = a[None,...] + (proj[...,None])*ab[None,...]
                
                # get distance from cylinder center line
                pr = np.linalg.norm(samples[:,:3] - result, axis=-1)
                
                # determine if point is already considered in a later shell
                r_mask = pr <= radius
                l_mask = np.logical_and((proj < 1), (proj > 0))
                overall_mask *= np.logical_not(l_mask*r_mask)
            
    return overall_mask

  
    
