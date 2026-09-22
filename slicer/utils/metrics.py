import numpy as np

#############################################
#  Printing Time Computation and Efficiency #
#############################################

def compute_path_print_efficiency(path, print_mask):
    percent_time_on = np.sum(print_mask)
    percent_time_off = len(print_mask) - percent_time_on
    
    
    total_dist_travelled = 0
    dist_travelled_on = 0
    dist_travelled_off = 0
    if len(print_mask) > 1:
        for i in range(len(print_mask)):
            n1 = path[i,:]
            n2 = path[i+1,:]
            
            total_dist_travelled += np.linalg.norm(n2-n1)
            
            if print_mask[i] == 1:
                dist_travelled_on += np.linalg.norm(n2-n1)
            else:
                dist_travelled_off += np.linalg.norm(n2-n1)

    
    return percent_time_on, percent_time_off, total_dist_travelled, dist_travelled_on, dist_travelled_off