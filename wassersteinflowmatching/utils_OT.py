import ott # type: ignore
from ott.solvers import linear # type: ignore




def ot_mat_from_distance(distance_matrix, eps = 0.1, lse_mode = False): #produces deltas from x to y


    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'mean'),
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    return(ot_solve.matrix)


def ot_mat(pc_x, pc_y, eps = 0.1, lse_mode = False): #produces deltas from x to y

    
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'mean'),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    return(ot_solve.reg_ot_cost)

def transport_plan(pc_x, pc_y, eps = 0.001, lse_mode = True): #produces deltas from x to y

    
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    
    potentials = ot_solve.to_dual_potentials()
    delta = potentials.transport(pc_x)-pc_x
    return(delta)