
import numpy as np
import glob
import json
import pprint
import os
import argparse

from slicer.utils import general
from slicer.utils import control

############################################################
#         Data Loader and Writer Helper Functions          #
############################################################

def parser_funct():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Script with custom parameters.")

    parser.add_argument(
        "--job", 
        type=str, 
        default=None, 
        help="What job profile to run.\
        if none specified will use user specified parameters."
    )

    parser.add_argument(
        "--job_names", 
        type=str, 
        nargs="+",
        default=["example"],
        help="what to save the job as (this should match\
        the number of stls used).\
         Can be set through a job profile."
    )

    parser.add_argument(
        "--stl_names",
        type=str,
        nargs="+",
        default=["hollow_cube.stl"],
        help="List of STL file names to generate print jobs.\
        Can be set through a job profile."
    )

    parser.add_argument(
        "--printer_profiles",
        type=str,
        nargs="+",
        default=["1_printer_profile_cylindrical_contour_1mm.json"],
        help="List of printer profiles to use.\
        Can be set through a job profile."
    )

    parser.add_argument(
        "--obj_scale", 
        type=float, 
        default=None, 
        help="scale object to fit inside printer volume.\
         Can be set through a job profile."
    )

    parser.add_argument(
        "--sim_scale", 
        type=float, 
        default=1., 
        help="simulated volume ratio.\
         scales up/down simulated printer volume.\
         Can be set through job profile."
    )

    parser.add_argument(
        "--print_wdh_dimensions",
        type=float,
        nargs="+",
        default=[139.2,139.2, 196.8],
        help="Dimensions of the printer volume (mm).\
        Can be set through a job profile."
    )

    parser.add_argument(
        "--velocity", 
        type=float, 
        default=50., 
        help="Specify desired printer speed in mm/sec.\
            Only used for estimating print time\
            Can be set through job profile."
    )
    parser.add_argument(
        "--desired_print_time", 
        type=float, 
        default=1., 
        help="Specify desired printer completion time (hr).\
        Only used for estimating needed flat speed.\
        Can be set through job profile."
    )

    parser.add_argument(
        "--load_data", 
        type=bool, 
        default=False, 
        help="Specifies if printer job should be loaded\
            or should be regenerated.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--load_data_path", 
        type=str, 
        default="../results/shells", 
        help="Folder where saved printer jobs can be loaded from"
    )
    
    parser.add_argument(
        "--use_fast_sdf", 
        type=bool, 
        default=False, 
        help="Specifies if faster but less accurate\
            sdf computation should be used. Note\
            this can result in artifacts\
            Can be set through job profile."
    )

    parser.add_argument(
        "--add_safety", 
        type=bool, 
        default=False, 
        help="Specifies if printer job should use\
            safety threshold to separate print job\
            into cylindrical and axial print shells.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--fast_collision_check", 
        type=bool, 
        default=True, 
        help="Specifies if printer job should use\
            fast or slow collision check.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--visualize_fit", 
        type=bool, 
        default=False, 
        help="Specifies if user wants to visualize\
            how well object fits in print space.\
            Recommended for large or tall objects.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--visualize_result", 
        type=str, 
        default='blender', 
        help="Specifies if user wants to visualize\
            The final print paths (matplotlib or blender).\
            Recommended for for first time prints.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--save_visualization", 
        type=bool, 
        default=True, 
        help="Specifies if user wants to save or view the visualizations.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--save_visualization_path", 
        type=str, 
        default="../results/complete_job", 
        help="Specifies visualization save path.\
            Can be set through job profile."
    )
    

    parser.add_argument(
        "--vis_alpha_value", 
        type=float, 
        default=0.0, 
        help="Specifies alpha value during visualization.\
            0 means no alpha value applied (fully opaque).\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_printer_id", 
        type=str, 
        default=None, 
        help="Specifies which printer to highlight during visualization.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_frame_scale", 
        type=float, 
        default=10.0, 
        help="Specifies frame size during visualization.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_zoom_scale", 
        type=float, 
        default=1.0, 
        help="Specifies zoom level during visualization.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_path_type", 
        type=str, 
        default="all", 
        help="Specifies whether to show just 'cylindrical only', 'safety only' or 'all'.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_show_off_edges", 
        type=bool, 
        default=False, 
        help="Specifies if user wants to view edges where CTRs are not printing.\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_camera_view", 
        type=str, 
        default="angle top down side", 
        help="Specifies camera angle for viewer (only for blender view).\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_frames", 
        type=int, 
        default=9, 
        help="Specifies number of frames to save from viewer (only for blender view).\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_color_mode", 
        type=str, 
        default='different color', 
        help="Specifies if cylindrical and safety paths should have same color (only for blender view).\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_draw_ctr", 
        type=bool, 
        default=False, 
        help="Specifies if ctrs should be visualized as well (only for blender view).\
            Can be set through job profile."
    )

    parser.add_argument(
        "--vis_p_state",
        type=int,
        nargs="+",
        default=None,
        help="specifies what state the ctrs should be rendered in (only for blender view).\
        Can be set through a job profile."
    )
    
    parser.add_argument(
        "--vis_hid_ctr_keys",
        type=int,
        nargs="+",
        default=None,
        help="specifies which ctrs should be hidden in (only for blender view).\
     Can be set through a job profile."
    )




    return parser

def load_job_profile(args):
    print("#####################################################")
    if args.job is not None:
        print(f'chosen job plan:\n{args.job}')
        with open(args.job, 'r') as f:
            job_profile = json.load(f)
        
        
        for (key, val) in job_profile.items():
            if key != "job":
                setattr(args, key, val)
    else:
        
        print("no job plan was specified.\
        using user specified parameters instead.")

    print("#####################################################")
    print("job specifies following parameters")
    for (key, val) in vars(args).items():
        print(f"--{key}: {val}")

    return args

def load_printer_profile(profile_file_path, sim_scale, print_vol_width, print_vol_depth, print_vol_height, mesh):
    printers = {}
    
    print(f'chosen printer profile plan: {profile_file_path}')
    with open(profile_file_path, 'r') as f:
        printer_plan = json.load(f)
        
        for printer_id in printer_plan:
            printers[str(printer_id)] = {'printer_id': printer_plan[printer_id]['printer_id'],
                                        'origin': np.asarray(printer_plan[printer_id]['origin'])*sim_scale,
                                        'normal': np.asarray(printer_plan[printer_id]['normal']),
                                        'resolution': printer_plan[printer_id]['resolution'],
                                        'CTR radius': printer_plan[printer_id]['CTR radius'],
                                        "safety radius": printer_plan[printer_id]['safety radius'],
                                        'coeffs': [np.asarray(printer_plan[printer_id]['coeffs'][0]),
                                                    np.asarray(printer_plan[printer_id]['coeffs'][1]),
                                                    np.asarray(printer_plan[printer_id]['coeffs'][2])],
                                        'shell_method': printer_plan[printer_id]['shell_method'],
                                        'planning_method': printer_plan[printer_id]['planning_method'],
                                        'color': printer_plan[printer_id]['color']
                                        }
            
            
    for printer_id in printers:
            printer = printers[str(printer_id)]
            min_u, max_u = control.compute_range_of_allowed_control_efforts(printer,
                                                                r_lower_range=0, r_upper_range=200,
                                                                z_lower_range=-1000, z_upper_range=0,
                                                                min_u_bound=-100, max_u_bound=0,
                                                                resolution=1)
            printers[str(printer_id)]["lower_u"] = min_u
            printers[str(printer_id)]["upper_u"] = max_u
            
            
    for printer in printers:
        # get all necessary printer parameters
        coeffs = printers[str(printer)]["coeffs"]
        lower_u = printers[str(printer)]["lower_u"]
        upper_u = printers[str(printer)]["upper_u"]
        origin = printers[str(printer)]["origin"]
        normal = printers[str(printer)]["normal"]
        resolution = printers[str(printer)]['resolution']
        
        # compute max and min allowable r and z values for each printer
        max_radius  = control.compute_val_from_control_effort(coeffs[0], lower_u).item()
        min_radius = control.compute_val_from_control_effort(coeffs[0], upper_u).item()
        max_z_dist = control.compute_z_tip_from_r_tip(coeffs[0], coeffs[1], max_radius, l_bound=lower_u, u_bound=upper_u).item() - coeffs[2]
        min_z_dist = control.compute_z_tip_from_r_tip(coeffs[0], coeffs[1], min_radius, l_bound=lower_u, u_bound=upper_u).item()
        
        printers[str(printer)]["max_r"] = max_radius
        printers[str(printer)]["min_r"] = min_radius
        printers[str(printer)]["max_z_dist"] = max_z_dist
        printers[str(printer)]["min_z_dist"] = min_z_dist
        

        e1, e2, e3 = general.generate_tf(normal)
        tf = np.eye(4)
        tf[:3,:3] = np.asarray([e1,e2,e3]).T
        tf[:3,3] = origin
        
        printers[str(printer)]['tf'] = tf
        printers[str(printer)]['inverse_tf'] = np.linalg.inv(tf)
        
        # compute the minimum amount of levels necessary to fully reprint the desired object
        bounding_box_verts = mesh.bounding_box.vertices
        bounding_box_verts = np.concatenate((bounding_box_verts, np.ones((bounding_box_verts.shape[0],1))),axis=-1)
        local_verts = (np.linalg.inv(tf)@bounding_box_verts.T).T
        bounding_box_r = np.max(np.stack(general.calcCylindCord(local_verts[:,:3]))[0,...])
        
        min_max_radius = np.minimum(bounding_box_r, max_radius)
        printers[str(printer)]["boundary set"] = []
        
        levels = np.arange(min_radius, min_max_radius,resolution)
        printers[str(printer)]["potential_levels"] = np.flip(levels)

        print(f"possible levels:\n{levels}")
        print(" ")

    return printers

def load_printer_job(data_load_path, data_folder):
    files = glob.glob(os.path.join(data_load_path,data_folder,"*.json"))
    # for file in files:
    #     print(f"loading file: {file}")
    
    printers = {}
    shells = {}
    shell_samples = {}
    for file in files:
        print(f"trying: {file.split('/')[-1]}")
        with open(file, 'r') as f:
            print(f"successfully opened: {file}")
            printer_plan = json.load(f)
            print(f"successfully read in file!")
            printer_id = str(printer_plan['printer info']["printer_id"])
            printers[str(printer_id)] = {'printer_id': printer_plan['printer info']['printer_id'],
                                        'origin': np.asarray(printer_plan['printer info']['origin']),
                                        'normal': np.asarray(printer_plan['printer info']['normal']),
                                        'resolution': printer_plan['printer info']['resolution'],
                                        'CTR radius': printer_plan['printer info']['CTR radius'],
                                        "safety radius": printer_plan['printer info']['safety radius'],
                                        'coeffs': [np.asarray(printer_plan['printer info']['coeffs'][0]),
                                                    np.asarray(printer_plan['printer info']['coeffs'][1]),
                                                    np.asarray(printer_plan['printer info']['coeffs'][2])],
                                        'shell_method': printer_plan['printer info']['shell_method'],
                                        'color': printer_plan['printer info']['color'],
                                        'lower_u': printer_plan['printer info']['lower_u'],
                                        'upper_u': printer_plan['printer info']['upper_u'],
                                        'boundary set': [(np.asarray(b[0]),np.asarray(b[1]))  for b in printer_plan['printer info']['boundary set']],
                                        'max_r': printer_plan['printer info']['max_r'],
                                        'min_r': printer_plan['printer info']['min_r'],
                                        'max_z_dist': printer_plan['printer info']['max_z_dist'],
                                        'min_z_dist': printer_plan['printer info']['min_z_dist'],
                                        'potential_levels': np.asarray(printer_plan['printer info']['potential_levels']),
                                        
                                        }
            e1, e2, e3 = general.generate_tf(np.asarray(printer_plan['printer info']['normal']))
            tf = np.eye(4)
            tf[:3,:3] = np.asarray([e1,e2,e3]).T
            tf[:3,3] = np.asarray(printer_plan['printer info']['origin'])
            
            printers[str(printer_id)]['tf'] = tf
            printers[str(printer_id)]['inverse_tf'] = np.linalg.inv(tf)
            
            
            
            shells[printer_id] = {'levels': {}}
            shell_samples[printer_id] = {'levels': {}}
            
            # print(f"is there a schedule? {'schedule' in printer_plan.keys()}")
            if "schedule" in printer_plan.keys():
                shells[printer_id]["schedule"] =  np.asarray(printer_plan["schedule"]).astype(int)

            for level in printer_plan['levels']:
                if str(level) in printer_plan['levels']:
                    shells[printer_id]['levels'][level] = {'path': np.asarray(printer_plan['levels'][level]['path']),
                                                                    'mask': np.asarray(printer_plan['levels'][level]['mask'])
                                                                    } 
                
                if str(level) in printer_plan["shell info"]:
                    shell_samples[printer_id]['levels'][level] = {'samples': np.asarray(printer_plan["shell info"][level]['samples']),
                                                                            'sdf': np.asarray(printer_plan["shell info"][level]['sdf']),
                                                                            'shell_mask': np.asarray(printer_plan["shell info"][level]['shell_mask']),
                                                                            }

    return printers, shells, shell_samples

def save_printer_jobs(dirname, printers, printer_shells):
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    for printer in printers:
        printer_info = printers[printer]
        printer_job = {}
        printer_job["printer info"] = {'printer_id': printer_info['printer_id'],
                                        'origin': printer_info['origin'].tolist(),
                                        'normal': printer_info['normal'].tolist(),
                                        'resolution': printer_info['resolution'],
                                        'CTR radius': printer_info['CTR radius'],
                                        "safety radius": printer_info['safety radius'],
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
        if "schedule" in printer_shells[printer].keys():
            printer_job["schedule"] = printer_shells[printer]["schedule"]
            
        printer_job["shell info"] ={}
        for level in printer_shells[printer]["levels"]:
            path = printer_shells[printer]["levels"][level]["path"]
            print_mask = printer_shells[printer]["levels"][level]["mask"]
            printer_job["levels"][str(level)] = {'path':path.tolist(), 'mask': print_mask.tolist()}

            if "shell info" in printer_shells[printer].keys():
                if "samples" in printer_shells[printer]["shell info"]["levels"][str(level)].keys():
                    samples = printer_shells[printer]["shell info"]["levels"][str(level)]['samples'].tolist()
                else:
                    samples = []
                    
                if "sdf" in printer_shells[printer]["shell info"]["levels"][level].keys():
                    sdf = printer_shells[printer]["shell info"]["levels"][level]['sdf'].tolist()
                else:
                    sdf = []
                    
                if "shell_mask" in printer_shells[printer]["shell info"]["levels"][level].keys():
                    shell_mask = printer_shells[printer]["shell info"]["levels"][level]['shell_mask'].tolist()
                else:
                    shell_mask = []

                printer_job["shell info"][str(level)] = {'samples': samples, 'sdf':sdf, 'shell_mask': shell_mask}
             
        printer_job["collision info"] = {}
        if "collisions" in printer_shells.keys(): 
            if printer in printer_shells["collisions"]:
                printer_job["collision info"] = printer_shells["collisions"][printer]
                
        filename = os.path.join(dirname,'printer_'+str(printer)+'.json')
        print(f"saving to file {filename}")
        out_file = open(filename,"w")
        json.dump(printer_job, out_file, indent = 4)
        out_file.close()
