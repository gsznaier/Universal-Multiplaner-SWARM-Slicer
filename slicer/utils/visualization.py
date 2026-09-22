import os
import numpy as np
from datetime import datetime

import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
from scipy.spatial.transform import Rotation as Rot
from sklearn.neighbors import KDTree
import pandas as pd
import datetime as dt

import blender_plots as bplt # wait until ready
import bpy
from colorsys import hls_to_rgb

from slicer.utils import general
from slicer.utils import control
from slicer.utils import metrics
from slicer.utils import shell_generation as sg


#############################################
#       Printer Visualization Code          #
#############################################
def draw_path_edges(ax, path, print_mask, selected_printer, printer_id, 
                    resolution, colors, user_alpha, show_off_edges=True):
    
    if selected_printer is not None:
        selected_printer = int(selected_printer)
    printer_id = int(printer_id)
    
    for j in range(len(print_mask)):
        n1 = path[j,:]
        n2 = path[j+1,:]
        alpha = 1 if (selected_printer is None) or (selected_printer == printer_id) else user_alpha
        
        if print_mask[j] == 1:
            ax.plot([n1[0], n2[0]], 
                    [n1[1], n2[1]], 
                    [n1[2], n2[2]], c=colors[printer_id], linewidth=resolution, alpha=alpha)
        elif show_off_edges:
            ax.plot([n1[0], n2[0]], 
                        [n1[1], n2[1]], 
                        [n1[2], n2[2]], c='k', 
                        linewidth=resolution,
                        alpha=alpha)
            
    return ax

def look_at(location, target, up):
    z = (location - target)
    z_ = z / np.linalg.norm(z)
    x = np.cross(up, z, axis=-1)
    x_ = x / np.linalg.norm(x)
    y = np.cross(z, x, axis=-1)
    y_ = y / np.linalg.norm(y)

    R = np.stack([x_, y_, z_], axis=-1)
    return R     
     
def draw_paths_blender(mesh, sim_scale, print_volume, printers, shells, units, desired_time=1, 
               desired_velocity=0.5, show_figure=True, save_figure=False, 
               save_path='./video', use_fast=False, alpha=0.2, printer_select=None, tf_scale=10, zoom=3,
               show='all', show_off_edges=True, camera_view='top down', num_frames=5, color_mode='same color',
               draw_ctr=False, p_state=None, hid_ctr_keys=None):
    
    if hid_ctr_keys is None:
        hid_ctr_keys = []

    print(f"printers: {printers.keys()}")

    n, l = 50, 100
    
    # Define our new purple colors (RGBA)
    dark_purple = (0.3, 0.0, 0.5, 1.0)
    light_purple = (0.7, 0.4, 0.9, 1.0)
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.samples = 256
    
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"])
    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    bpy.ops.object.light_add(
        type="SUN", radius=1, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.data.objects["Sun"].data.energy = 10
    bpy.data.objects["Sun"].data.angle = np.pi / 2
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        "Strength"
    ].default_value = 0.5
    
    # Create the global materials
    mat_dark = bpy.data.materials.new(name="Dark_Purple")
    mat_dark.diffuse_color = dark_purple
    
    mat_light = bpy.data.materials.new(name="Light_Purple")
    mat_light.diffuse_color = light_purple

    # Black material for the CTR
    if draw_ctr:
        mat_ctr = bpy.data.materials.new(name="CTR_Black")
        mat_ctr.diffuse_color = (0.05, 0.05, 0.05, 1.0)
    
    for printer in printers.items():
        p_idx = str(printer[0])
        print_resolution = printers[p_idx]['resolution']
        
        full_on_paths = {'dark': [], 'light': []}
        full_off_path = []
        print(f"printer {p_idx}")
        
        safety_radius = float(printers[p_idx]['safety radius'])
        
        # Determine the target chronological limits
        limit_paths = draw_ctr and (p_state is not None)
        if limit_paths:
            target_lvl_idx = int(p_state[2 * int(p_idx)])
            target_path_idx = int(p_state[2 * int(p_idx) + 1])
        else:
            target_lvl_idx = float('inf')
            target_path_idx = float('inf')
            
        level_ids = list(shells[p_idx]['levels'].keys())
        
        for curr_lvl_idx, level in enumerate(level_ids):
            
            # --- 1. Skip unprinted future levels strictly by index ---
            if curr_lvl_idx > target_lvl_idx:
                break
            # --------------------------------------------------------

            is_inside_safety = float(level) < safety_radius
            
            if show == "safety only" and not is_inside_safety:
                continue
            elif show == "cylindrical only" and is_inside_safety:
                continue
            
            if color_mode == "different color":
                active_mat = mat_dark if is_inside_safety else mat_light
                color_key = 'dark' if is_inside_safety else 'light'
            else:
                active_mat = mat_dark
                color_key = 'dark'

            print(f"level done {level}")
            path = shells[p_idx]['levels'][str(level)]['path']
            print_mask = shells[p_idx]['levels'][str(level)]['mask']
            
            # --- 2. Iterate forward chronologically ---
            for j in range(len(path) - 1):
                if j >= len(print_mask):
                    break
                    
                # --- 3. Stop exactly at the printer's current path state ---
                if curr_lvl_idx == target_lvl_idx and j >= target_path_idx:
                    break
                # -----------------------------------------------------------
                
                if print_mask[j] == 1:
                    full_on_paths[color_key].append([path[j], path[j+1]])
                    curve = bpy.data.curves.new(name=str(p_idx)+"_"+ str(j),type='CURVE')
                    curve.dimensions = '3D'
                    curve.bevel_depth = 1 # Set thickness
                    polyline = curve.splines.new('POLY')
                    coords = path[j:j+2,:]
                    n_pts = coords.shape[0]-1
                    polyline.points.add(n_pts)
                    for k, coord in enumerate(coords):
                        px,py,pz = coord
                        polyline.points[k].co = (px, py, pz, 1)
                    obj = bpy.data.objects.new(str(p_idx)+"_"+ str(j), curve)
                    obj.data.materials.append(active_mat)
                    scn = bpy.context.scene
                    scn.collection.objects.link(obj)
                    bpy.context.view_layer.objects.active = obj
                else:
                    if show_off_edges:
                        full_off_path.append([path[j], path[j+1]])
                        
        for c_key, p_list in full_on_paths.items():
            if len(p_list) > 0: 
                p_arr = np.concatenate(p_list, axis=0)
                scatter_color = dark_purple if c_key == 'dark' else light_purple
                bplt.Scatter(p_arr[:,0], p_arr[:,1], p_arr[:,2], 
                            color=scatter_color, name=f"{int(p_idx)}_{c_key}", marker_scale=[print_resolution/10]*3)

        if len(full_off_path) > 0:
            full_off_path = np.concatenate(full_off_path,axis=0)

        # CTR Drawing Logic
        if draw_ctr and p_state is not None:
            if int(p_idx) not in hid_ctr_keys:
                max_z_dist = -1 * printers[p_idx]["max_z_dist"]
                min_z_dist = -1 * printers[p_idx]["min_z_dist"]
                r_coeff, z_coeff, disp = printers[p_idx]["coeffs"]
                lower_u = printers[p_idx]["lower_u"]
                upper_u = printers[p_idx]["upper_u"]
                ctr_radius = printers[p_idx]["CTR radius"]
                tf = printers[p_idx]['tf']
                wtl_tf = printers[p_idx]['inverse_tf']
                
                # Fetch tip point exactly at the p_state bounds
                printer_lvl = int(p_state[2 * int(p_idx)])
                path_ind = int(p_state[2 * int(p_idx) + 1])
                tip_point = shells[p_idx]["levels"][level_ids[printer_lvl]]["path"][path_ind]
                
                local_coord_pt = (wtl_tf @ np.concatenate((tip_point, [1]), axis=-1).T).T[:3]
                tip_r, tip_ang, tip_z = general.calcCylindCord(local_coord_pt)
                tip_p = np.asarray([tip_r, tip_ang, tip_z])
                
                local_spine, reachable = control.compute_local_occupied_region(
                    r_coeff=r_coeff, z_coeff=z_coeff,
                    min_height=min_z_dist, max_height=max_z_dist,
                    tip_p=tip_p, l_bound=lower_u, u_bound=upper_u,
                    resolution=ctr_radius
                )
                
                l_coords = general.calcCartFromCylind(local_spine)
                local_spine = np.stack(l_coords, axis=1)
                indices = np.argsort(local_spine[:, 2])
                local_spine = local_spine[indices]
                global_spine = (tf @ np.concatenate((local_spine, np.ones((local_spine.shape[0], 1))), axis=-1).T).T
                
                ctr_curve = bpy.data.curves.new(name=f"CTR_Curve_{p_idx}", type='CURVE')
                ctr_curve.dimensions = '3D'
                ctr_curve.bevel_depth = ctr_radius 
                ctr_polyline = ctr_curve.splines.new('POLY')
                
                n_pts_ctr = global_spine.shape[0] - 1
                ctr_polyline.points.add(n_pts_ctr)
                
                for k, coord in enumerate(global_spine):
                    px, py, pz = coord[:3]
                    ctr_polyline.points[k].co = (px, py, pz, 1)
                    
                ctr_obj = bpy.data.objects.new(f"CTR_Obj_{p_idx}", ctr_curve)
                ctr_obj.data.materials.append(mat_ctr)
                bpy.context.scene.collection.objects.link(ctr_obj)
        
    angles = np.linspace(0, 2*np.pi, num_frames)
    camera_origins = np.zeros((num_frames, 3))
    up_vectors = np.zeros((num_frames, 3))

    if camera_view == 'top down':
        camera_origins[:, 2] = 150.0
        up_vectors[:, 0] = np.cos(angles)
        up_vectors[:, 1] = np.sin(angles)
    elif camera_view == 'bottom up':
        camera_origins[:, 2] = -150.0
        up_vectors[:, 0] = np.cos(angles)
        up_vectors[:, 1] = np.sin(angles)
    elif camera_view == 'side':
        camera_origins[:, 0] = 150.0 * np.cos(angles)
        camera_origins[:, 1] = 150.0 * np.sin(angles)
        up_vectors[:, 2] = 1.0
    elif camera_view == 'angle top down side':
        camera_origins[:, 0] = 100.0 * np.cos(angles)
        camera_origins[:, 1] = 100.0 * np.sin(angles)
        camera_origins[:, 2] = 100.0
        up_vectors[:, 2] = 1.0
    elif camera_view == 'angle bottom up side':
        camera_origins[:, 0] = 100.0 * np.cos(angles)
        camera_origins[:, 1] = 100.0 * np.sin(angles)
        camera_origins[:, 2] = -100.0
        up_vectors[:, 2] = 1.0
    else:
        print(f"Warning: View '{camera_view}' not recognized. Defaulting to 'top down'.")
        camera_origins[:, 2] = 150.0
        up_vectors[:, 0] = np.cos(angles)
        up_vectors[:, 1] = np.sin(angles)

    for i in range(num_frames):
        co = camera_origins[i]
        up_vec = up_vectors[i]
        
        cR = look_at(location=co, target=np.array([0,0,0]), up=up_vec)
        cR = Rot.from_matrix(cR).as_matrix()
        ltw_cR = np.linalg.inv(cR)
        
        camera_data = bpy.data.cameras.new(name="New Camera")
        camera_object = bpy.data.objects.new("New Camera", camera_data)
        mat = np.eye(4)
        mat[:3,:3] = ltw_cR
        camera_object.matrix_basis = mat
        camera_object.location = co
        bpy.context.collection.objects.link(camera_object)
        bpy.context.scene.camera = camera_object
        
        if save_figure:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            savepath = os.path.join(save_path, f'printers_{len(printers.keys())}_'+"{:06d}".format(i)+'.png')
            bpy.context.scene.render.filepath = savepath
            bpy.context.scene.render.resolution_x = 512
            bpy.context.scene.render.resolution_y = 512
            bpy.context.scene.cycles.samples = 100
            bpy.ops.render.render(use_viewport = True, write_still=True)
        else:
            bpy.context.scene.render.use_file_extension = False
            bpy.ops.render.render(write_still=False)
            img = np.array(bpy.data.images['Render Result'])
            plt.draw()
            plt.imshow(img)
            plt.pause(.001)

        bpy.data.objects.remove(camera_object, do_unlink=True)
                        
def draw_paths(mesh, sim_scale, print_volume, printers, shells, units, desired_time=1, 
               desired_velocity=0.5, show_figure=True, save_figure=False, 
               save_path='./video', use_fast=False, alpha=0.2, printer_select=None, tf_scale=10, zoom=3,
               show='all', show_off_edges=True):

    print(f"printers: {printers.keys()}")

    # color code the printers
    viridis = mpl.colormaps['rainbow']
    colors = viridis(np.linspace(0, 1, np.max(np.asarray(list(printers.keys()), dtype=int))+1))
    
    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0, 0)
    vertices = np.asarray(mesh.vertices)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    total_percent_on = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_percent_off = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_paths_len = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled_on = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    total_dist_travelled_off = np.zeros(np.max(np.asarray(list(printers.keys()), dtype=int))+1)
    desired_time = desired_time*60*60 # time in seconds

    for printer in printers.items():
        print(f"current printer {printer[0]}")
        o = printers[printer[0]]["origin"]
    
        print_resolution = printers[printer[0]]['resolution']
    
        if printer[0] not in shells:
            continue
        
        print(f"number of levels: {len(shells[printer[0]]['levels'].keys())}")
        for level in list(shells[printer[0]]['levels'].keys())[:-1]:
            print(f"current level {level}")
            path = shells[printer[0]]['levels'][str(level)]['path']
            print_mask = shells[printer[0]]['levels'][str(level)]['mask']
            
            
            print("--------------------------------------")
            print(f"checking printer {printer[0]}")
            print(f"checking path size: {path.shape}")
            print(f"checking mask size: {print_mask.shape}")
            print(f"any non-zero: {np.any(print_mask)}")
            print("--------------------------------------")
            print(" ")
            
            percent_time_on, percent_time_off, dist_travelled, dist_travelled_on, dist_travelled_off = metrics.compute_path_print_efficiency(path, print_mask)
            total_percent_on[int(printer[0])]+=percent_time_on
            total_percent_off[int(printer[0])]+=percent_time_off
            total_paths_len[int(printer[0])]+=len(print_mask)
            total_dist_travelled[int(printer[0])]+=dist_travelled
            total_dist_travelled_on[int(printer[0])]+=dist_travelled_on
            total_dist_travelled_off[int(printer[0])]+=dist_travelled_off
            
            if show == "all":
                ax = draw_path_edges(ax, path.copy(), print_mask.copy(), printer_select, int(printer[0]), 
                                    print_resolution, colors, alpha, show_off_edges=show_off_edges)
            elif show == "safety only" and float(level) < float(printers[printer[0]]['safety radius']):
                ax = draw_path_edges(ax, path.copy(), print_mask.copy(), printer_select, int(printer[0]), 
                                    print_resolution, colors, alpha, show_off_edges=show_off_edges)
            elif show == "cylindrical only" and float(level) >= float(printers[printer[0]]['safety radius']):
                ax = draw_path_edges(ax, path.copy(), print_mask.copy(), printer_select, int(printer[0]), 
                                    print_resolution, colors, alpha, show_off_edges=show_off_edges)
        print(" ")
            
    ax.set_xlim(ax.get_xlim()[0]/zoom,ax.get_xlim()[1]/zoom)
    ax.set_ylim(ax.get_ylim()[0]/zoom,ax.get_ylim()[1]/zoom)
    ax.grid(False)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    
    if save_figure:
        # Rotate the axes and update
        angle_increment = 10
        angles = np.arange(0,1*(360)+angle_increment,angle_increment)
        for c, angle in enumerate(angles):
        
            # Normalize the angle to the range [-180, 180] for display
            angle_norm = (angle + 180) % 360 - 180
            # Cycle through a full rotation of elevation, then azimuth, roll, and all
            elev = azim = roll = 0
            if angle <= 360:
                elev = angle_norm
            elif angle <= 360*2:
                azim = angle_norm
            elif angle <= 360*3:
                roll = angle_norm
            else:
                elev = azim = roll = angle_norm

            # Update the axis view and title
            ax.view_init(azim, elev, roll)
            plt.draw()
            
            if save_figure:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                savepath = os.path.join(save_path, f'printers_{len(printers.keys())}_'+"{:06d}".format(c)+'.png')
                plt.savefig(savepath, bbox_inches='tight')
                print(f"figure {c} is saved")
        
            if show_figure:
                plt.pause(.001)
                
    if show_figure:
        plt.legend()
        plt.show()
        
    plt.close(fig)  
                  
def draw_bounding_box(ax, vertices, color='k', width=3, alpha=0.2):
    v1 = vertices[0]
    v2 = vertices[1]
    v3 = vertices[2]
    v4 = vertices[3]
    v5 = vertices[4]
    v6 = vertices[5]
    v7 = vertices[6]
    v8 = vertices[7]
    
    
    ax.plot([v1[0], v2[0]], 
        [v1[1], v2[1]], 
        [v1[2], v2[2]], c=color, linewidth=width)
    ax.plot([v1[0], v3[0]], 
        [v1[1], v3[1]], 
        [v1[2], v3[2]], c=color, linewidth=width)
    ax.plot([v2[0], v4[0]], 
        [v2[1], v4[1]], 
        [v2[2], v4[2]], c=color, linewidth=width)
    ax.plot([v3[0], v4[0]], 
        [v3[1], v4[1]], 
        [v3[2], v4[2]], c=color, linewidth=width)
    
    yy, zz = np.meshgrid(np.linspace(-np.abs(v1[1]-v3[1])/2, np.abs(v1[1]-v3[1])/2, 10),np.linspace(-np.abs(v1[2]-v2[2])/2, np.abs(v1[2]-v2[2])/2, 10))
    xx = v2[0]*np.ones(yy.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    ax.plot([v5[0], v6[0]], 
        [v5[1], v6[1]], 
        [v5[2], v6[2]], c=color, linewidth=width)
    ax.plot([v5[0], v7[0]], 
        [v5[1], v7[1]], 
        [v5[2], v7[2]], c=color, linewidth=width)
    ax.plot([v6[0], v8[0]], 
        [v6[1], v8[1]], 
        [v6[2], v8[2]], c=color, linewidth=width)
    ax.plot([v7[0], v8[0]], 
        [v7[1], v8[1]], 
        [v7[2], v8[2]], c=color, linewidth=width)
    
    yy, zz = np.meshgrid(np.linspace(-np.abs(v5[1]-v7[1])/2, np.abs(v5[1]-v7[1])/2, 10),np.linspace(-np.abs(v5[2]-v6[2])/2, np.abs(v5[2]-v6[2])/2, 10))
    xx = v6[0]*np.ones(yy.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    
    ax.plot([v1[0], v5[0]], 
        [v1[1], v5[1]], 
        [v1[2], v5[2]], c=color, linewidth=width)
    xx, zz = np.meshgrid(np.linspace(-np.abs(v1[0]-v5[0])/2, np.abs(v1[0]-v5[0])/2, 10),np.linspace(-np.abs(v1[2]-v2[2])/2, np.abs(v1[2]-v2[2])/2, 10))
    yy = v5[1]*np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    ax.plot([v2[0], v6[0]], 
        [v2[1], v6[1]], 
        [v2[2], v6[2]], c=color, linewidth=width) 
    ax.plot([v3[0], v7[0]], 
        [v3[1], v7[1]], 
        [v3[2], v7[2]], c=color, linewidth=width)
    xx, zz = np.meshgrid(np.linspace(-np.abs(v3[0]-v7[0])/2, np.abs(v3[0]-v7[0])/2, 10),np.linspace(-np.abs(v3[2]-v4[2])/2, np.abs(v3[2]-v4[2])/2, 10))
    yy = v7[1]*np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    xx, yy = np.meshgrid(np.linspace(-np.abs(v1[0]-v5[0])/2, np.abs(v1[0]-v5[0])/2, 10),np.linspace(-np.abs(v1[1]-v3[1])/2, np.abs(v1[1]-v3[1])/2, 10))
    zz = v5[2]*np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    ax.plot([v4[0], v8[0]], 
        [v4[1], v8[1]], 
        [v4[2], v8[2]], c=color, linewidth=width)
    
    xx, yy = np.meshgrid(np.linspace(-np.abs(v4[0]-v8[0])/2, np.abs(v4[0]-v8[0])/2, 10),np.linspace(-np.abs(v2[1]-v4[1])/2, np.abs(v2[1]-v4[1])/2, 10))
    zz = v8[2]*np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=alpha,color='b')
    
    return ax

def draw_allowed_printer_region(ax, printer, level, alpha=0.5, show_ideal_shell=False, facecolor=None):
    coeffs = printer["coeffs"]
    max_z_dist = -1*printer["max_z_dist"]
    min_z_dist = -1*printer["min_z_dist"]
    lower_u = printer["lower_u"]
    upper_u = printer["upper_u"]
    scale = printer['resolution']
    origin = printer["origin"]
    normal = printer["normal"]
    min_r = printer['min_r']
    max_r = printer['max_r']
    
    # bound given radius by printer's minimum and maximum radius
    level = np.abs(np.maximum(min_r,np.minimum(max_r, level)))
    print(f"level used: {level}")
    
    # get printer's coordinate frame and generate corresponding Transformation matrix
    e1, e2, e3 = general.generate_tf(normal)

    tf = np.eye(4)
    tf[:3,:3] = np.asarray([e1,e2,e3]).T
    tf[:3,3] = origin
    
    # generate local shell and convert it into samples of shape n x 3
    x, y, z = sg.generate_cylindrical_shell(level, scale, min_z_dist, max_z_dist)
    x_shape = x.shape
    
    cylind_pnts = np.stack((x.reshape(-1),y.reshape(-1),z.reshape(-1), np.ones(z.shape).reshape(-1)),axis=0)
    cylind_pnts = ((tf@cylind_pnts)[:3]).T
    
    if show_ideal_shell:
        ax.plot_surface(cylind_pnts[:,0].reshape(x_shape), cylind_pnts[:,1].reshape(x_shape), cylind_pnts[:,2].reshape(x_shape), alpha=alpha)
    
    pts = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)),axis=1)
    
    r, a, z = general.calcCylindCord(pts)
    pts = np.stack((r.reshape(-1), a.reshape(-1), z.reshape(-1)),axis=1)
    
    mask = []
    updated_pnts = []
    for tip_pnt in pts:
        flag = control.check_if_reachable(coeffs[0], coeffs[1], min_z_dist, max_z_dist, tip_pnt, l_bound=lower_u, u_bound=upper_u)
        mask.append(flag)
        if flag == False:
            new_r = control.compute_r_tip_from_z_tip(coeffs[0], coeffs[1], -1*tip_pnt[2], l_bound=lower_u, u_bound=upper_u)
            updated_pnts.append([new_r,tip_pnt[1],tip_pnt[2]])
        else:
            updated_pnts.append([tip_pnt[0],tip_pnt[1],tip_pnt[2]])
        
    pnts = np.asarray(updated_pnts)
    x,y,z = general.calcCartFromCylind(pnts)
    pnts = np.stack((x,y,z, np.ones(z.shape)),axis=0)
    pnts = ((tf@pnts)[:3]).T

    if facecolor is None:
        ax.plot_surface(pnts[:,0].reshape(x_shape), pnts[:,1].reshape(x_shape), pnts[:,2].reshape(x_shape), alpha=alpha, facecolor='red')
    else:
        ax.plot_surface(pnts[:,0].reshape(x_shape), pnts[:,1].reshape(x_shape), pnts[:,2].reshape(x_shape), alpha=alpha, facecolor=facecolor)
    
    return ax

def draw_mesh(mesh, print_volume, printers, skip_num=1,init_view=(0,0,0)):
    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(*init_view)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    ax.scatter(vertices[::skip_num, 0], vertices[::skip_num,1],vertices[::skip_num,2], c='b',s=0.01)
    ax = draw_bounding_box(ax, print_volume.vertices, color='k', width=3)
    
    for printer in printers: 
         # get printer's coordinate frame and generate corresponding Transformation matrix
         o = printers[str(printer)]["origin"]
         n = printers[str(printer)]["normal"]
         e1, e2, e3 = general.generate_tf(printers[str(printer)]["normal"])
        
         ax.quiver(o[0], o[1], o[2], e1[0], e1[1], e1[2],color="r",linewidth=2)
         ax.quiver(o[0], o[1], o[2], e2[0], e2[1], e2[2],color="g",linewidth=2)
         ax.quiver(o[0], o[1], o[2], e3[0], e3[1], e3[2],color="b",linewidth=2)
         ax.text(o[0], o[1], o[2]-10*np.sign(n[-1]), f"P {printer}", color='k', size=20)
    
    for printer in printers:
        last_level = printers[printer]["potential_levels"][0]
        ax = draw_allowed_printer_region(ax, printers[str(printer)], float(last_level), alpha=0.1, show_ideal_shell=False, facecolor='r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect('equal', 'box')
    plt.show()
