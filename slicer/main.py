
# general imports
import os
import glob
import json
import numpy as np

# mesh/stl handling imports
import trimesh
from trimesh import transformations

# visualization imports
import matplotlib.pyplot as plt
from matplotlib import colormaps

# slicer imports
from slicer.utils import general
from slicer.utils import dataloader
from slicer.utils import control
from slicer.utils import dataloader
from slicer.utils import shell_generation as sg
from slicer.utils import visualization as vis
from slicer.utils import scheduling


def main():

    # generate parser function with all function parameters
    parser = dataloader.parser_funct()

    # Parse arguments from command line
    args = parser.parse_args()

    print("starting program!")
    args = dataloader.load_job_profile(args)

    total_number_of_collisions = []
    total_number_of_paths = []

    # iterate through every profile and build the corresponding job
    for printer_profile in args.printer_profiles:
        profile_id = int(printer_profile.split("/")[-1].split("_")[0])

        # iterate through every job and build the print paths
        for name, folder_name in zip(args.stl_names, args.job_names):
            print(f"generating paths for: {name}")
            mesh = trimesh.load(name, process=True, force_mesh=True)

            # repair the stl file as best as we can to improve odds
            # of generating good print paths.
            print(f"is mesh water tight (initial)?: {mesh.is_watertight}")
            mesh.fill_holes()
            mesh.fix_normals()
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)
            print(trimesh.repair.broken_faces(mesh))
            print(f"is mesh water tight (final)?: {mesh.is_watertight}")
            
            # set mesh size to reflect real world size (mm)
            # and shift mesh to be at the center of print volume
            mesh.apply_translation(-1*mesh.center_mass)
            max_extents = np.max(mesh.extents)
            if args.obj_scale is not None:
                max_dim = ((args.obj_scale/general.UNIT_cm)*general.UNIT_mm)
            else:
                max_dim = max_extents
            mesh.apply_scale(max_dim/max_extents)
    
            # get mesh bounding box vertices and mesh vertices
            print(f"max extents mm: {np.max(mesh.bounding_box.extents)}")
            vertices = np.asarray(mesh.vertices)
            

            # create save location if it does not exist
            data_folder = folder_name + f"_{profile_id}_printers_cylindrical_volume"
            dirname = os.path.join(args.load_data_path,data_folder)
            print(f"Job Folder: {data_folder}")
            print(f"Save Path: {dirname}")
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            
            
            # specify print volume box (mm):
            print_vol_width = args.print_wdh_dimensions[0]*args.sim_scale 
            print_vol_depth = args.print_wdh_dimensions[1]*args.sim_scale
            print_vol_height = args.print_wdh_dimensions[2]*args.sim_scale #+ 60

            # create print volume space
            tf = np.eye(4)
            print_volume = trimesh.creation.box((print_vol_width,print_vol_depth,print_vol_height),tf)

            # if load data flag is not set then
            # regenerate the print paths from scratch
            if not args.load_data:
                
                # retrieve the specified printer definitions
                printers = dataloader.load_printer_profile(printer_profile,
                                                args.sim_scale,
                                                print_vol_width,
                                                print_vol_depth,
                                                print_vol_height,
                                                mesh)
                
                

                if args.visualize_fit:
                    vis.draw_mesh(mesh, print_volume, printers, init_view=(0, 0, 0))

                print("Generating a cylindrical Shells!")
                shells, shell_samples = sg.generate_shells(mesh, print_volume, printers, 
                                        units=general.UNIT_cm/general.UNIT_mm, desired_time=args.desired_print_time, 
                                        desired_velocity=args.velocity, use_fast=args.use_fast_sdf, dirname=dirname,
                                        add_safety=args.add_safety)
                
                print("FINISHED GENERATING SHELLS")
            else:
                printers, shells, shell_samples = dataloader.load_printer_job(args.load_data_path, data_folder)

            # print out print statistics
            sg.print_job_statistics(printers, shells, 
                                units=general.UNIT_cm/general.UNIT_mm, 
                                desired_time=args.desired_print_time, 
                                desired_velocity=args.velocity)
            
            #  Visualize all paths at once "cylindrical only" "safety only" "all"
            if args.visualize_result == "matplotlib":
                vis.draw_paths(mesh=mesh, sim_scale=args.sim_scale, print_volume=print_volume, 
                        printers=printers, shells=shells, units=general.UNIT_cm/general.UNIT_mm, 
                        desired_time=args.desired_print_time, desired_velocity=args.velocity, show_figure=(not args.save_visualization), 
                        save_figure=args.save_visualization, save_path=args.save_visualization_path, use_fast=args.use_fast_sdf, 
                        alpha=0, printer_select=args.vis_printer_id, tf_scale=args.vis_frame_scale, zoom=args.vis_zoom_scale, show=args.vis_path_type, show_off_edges=args.vis_show_off_edges)

            elif args.visualize_result == "blender":
                vis.draw_paths_blender(mesh=mesh, sim_scale=args.sim_scale, print_volume=print_volume, 
                    printers=printers, shells=shells, units=general.UNIT_cm/general.UNIT_mm, 
                    desired_time=args.desired_print_time, desired_velocity=args.velocity, show_figure=(not args.save_visualization), 
                    save_figure=args.save_visualization, save_path=args.save_visualization_path, use_fast=args.use_fast_sdf, 
                    alpha=args.vis_alpha_value, printer_select=args.vis_printer_id, tf_scale=args.vis_frame_scale, zoom=args.vis_zoom_scale, show=args.vis_path_type, show_off_edges=args.vis_show_off_edges, 
                    camera_view=args.vis_camera_view, num_frames=args.vis_frames, color_mode=args.vis_color_mode,
                    draw_ctr=args.vis_draw_ctr, p_state=args.vis_p_state, hid_ctr_keys=args.vis_hid_ctr_keys)
            
            print("finished visualization")

            # generate initial_schedule for CTR without considering collisions
            longest_path = np.max([scheduling.get_printer_path_len(shells[id]) for id in printers]).astype(int)
            print(f"longest path: {longest_path}")
            print(f"total number of paths: {np.sum([scheduling.get_printer_path_len(shells[id]) for id in printers]).astype(int)}")
            schedule = []
            printer_count_without_paths = 0
            for printer in printers:

                # determine path length for each printer
                path_length = scheduling.get_printer_path_len(shells[printer]).astype(int)
                if path_length == 0:
                    printer_count_without_paths+=1

                # generate a schedule of where when printing value is 1 and when
                # finished move value is set to 0
                printer_schedule = [0]*path_length + [1]*(longest_path-path_length)
                printer_schedule = np.asarray(printer_schedule)
                schedule.append(printer_schedule)
                
            # print statistics regarding what printers move and number collisions estimated
            print(f"number of printers without a path: {printer_count_without_paths}")
            schedule = np.stack(schedule,1)
            print(printers)
            num_of_collisions = 0
            if len(printers.keys()) > 1:
                num_of_collisions = scheduling.compute_total_num_of_collisions(printers, shells, schedule.astype(int), fast=True)
            print(f"total number of collisions in without a schedule: {num_of_collisions}")
            total_number_of_collisions.append(num_of_collisions)
            total_number_of_paths.append(np.sum([scheduling.get_printer_path_len(shells[id]) for id in printers]).astype(int))


        # evalute the number of collisions that would occur
        print(f"total number of collisions collected: {total_number_of_collisions}")
        print(f"total number of path lengths collected: {total_number_of_paths}")
        print(f"percent collision: {[col/pth for (col, pth) in zip(total_number_of_collisions,total_number_of_paths)]}")
        print(f"percent collision: {np.mean([col/pth for (col, pth) in zip(total_number_of_collisions,total_number_of_paths)])}")

        printer_shells = {}
        for printer in printers:
            printer_shells[printer] = {}
            printer_shells[printer]["levels"] = shells[printer]["levels"]
            printer_shells[printer]["schedule"] = schedule[:,int(printer)].tolist()
            printer_shells[printer]["shell info"] = {}
            printer_shells[printer]["shell info"]["levels"] = shell_samples[printer]["levels"]
        
        dataloader.save_printer_jobs(dirname, printers, printer_shells)
            
            
            
            


if __name__=="__main__":
    main()