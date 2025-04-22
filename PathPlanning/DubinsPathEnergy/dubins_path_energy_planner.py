"""

Dubins path Energy-optimized planner code
modified for Energy optimization

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from math import sin, cos, atan2, sqrt, acos, pi, hypot, sqrt
from scipy.integrate import quad
import numpy as np
from utils.angle import angle_mod, rot_mat_2d
import multiprocessing

show_animation = True

def plan_dubins_path(s_x, s_y, s_yaw, s_velocity,
                     g_x, g_y, g_yaw, g_velocity,
                     curvature_search_range, step_size=0.1, selected_types=None,
                     velocity_range=(0.0, 3.0), velocity_step=0.5):
    """
    Plan dubins path optimized for Energy consumption

    Parameters
    ----------
    s_x : float
        x position of the start point [m]
    s_y : float
        y position of the start point [m]
    s_yaw : float
        yaw angle of the start point [rad]
    s_velocity : float
        velocity of the start point [m/s]
    g_x : float
        x position of the goal point [m]
    g_y : float
        y position of the end point [m]
    g_yaw : float
        yaw angle of the end point [rad]
    g_velocity : float
        velocity of the goal point [m/s]
    curvature_search_range : list of float
        curvature search range for curve [1/m]
    step_size : float (optional)
        step size between two path points [m]. Default is 0.1
    selected_types : a list of string or None
        selected path planning types. If None, all types are used for
        path planning, and minimum Energy cost result is returned.
        You can select used path plannings types by a string list.
        e.g.: ["RSL", "RSR"]
    velocity_range : tuple of float
        range of velocities to consider for optimization (min, max) [m/s]
    velocity_step : float
        step size for velocity optimization [m/s]

    Returns
    -------
    x_list: array
        x positions of the path
    y_list: array
        y positions of the path
    yaw_list: array
        yaw angles of the path
    velocity_list: array
        velocity values of the path
    modes: array
        mode list of the path
    lengths: array
        length list of the path segments.
    segment_velocities: array
        velocity list of the path segments.
    segment_coordinates: array
        coordinates list of the path segments.
    segment_energy_costs: array
        energy cost list of the path segments.
    segment_power_costs: array
        power cost list of the path segments.
    """
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # Calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    # AFTER: produce velocity_list right away along with other optimized parameters
    lp_x, lp_y, lp_yaw, velocity_list, modes, lengths, segment_velocities, lsegment_coordinates, segment_energy_costs, segment_power_costs = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, s_velocity, g_velocity, curvature_search_range, step_size,
        planning_funcs, velocity_range, velocity_step)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    lsegment_coordinates_x, lsegment_coordinates_y = lsegment_coordinates
    converted_segment_coordinates = np.stack([lsegment_coordinates_x, lsegment_coordinates_y]).T @ rot
    segment_coordinates_x = converted_segment_coordinates[:, 0] + s_x
    segment_coordinates_y = converted_segment_coordinates[:, 1] + s_y
    segment_coordinates = (segment_coordinates_x, segment_coordinates_y)

    return x_list, y_list, yaw_list, velocity_list, modes, lengths, segment_velocities, segment_coordinates, segment_energy_costs, segment_power_costs

def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)

def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL, }


def _calculate_energy_cost(d1, d2, d3, mode, v1, v2, v3, v4, curvature):
    """
    Calculate Energy cost based on the path segments and vehicle parameters

    Parameters
    ----------
    d1, d2, d3 : float
        Lengths of each segment in the path
    mode : list of str
        List of modes for each segment ('L', 'S', 'R')
    v1, v2, v3, v4, : float
        End velocities of each segment [m/s]
        (v1 is just the start velocity of first segment)
    curvature: float
        Curvature of right and left Dubin turns.
    Returns
    -------
    energy_cost : float
        Total Energy cost for the path
    segment_energy_costs : list of float
        Energy cost for each segment
    segment_power_costs : list of float
        Power cost for each segment
    """
    if d1 is None:
        return float('inf')

    segments = [d1, d2, d3]
    velocities = [v1, v2, v3, v4]
    radius = 1 / curvature
    energy_cost = 0.0
    segment_energy_costs = []
    segment_power_costs = []
    gravity = 9.81

    for i, (segment_mode, distance) in enumerate(zip(mode, segments)):
        v_start = velocities[i]
        v_end = velocities[i+1]

        # TODO Ask Leonard if this is correct
        time = 2 * distance / max((v_start + v_end), 0.0001)

        tangential_acceleration = abs((v_end**2 - v_start**2) / (2 * distance)) if distance > 0 else 0

        velocity_mid = (v_end**2 + v_start**2) / 2
        centripetal_acceleration = (velocity_mid / radius) if radius > 0 else 0
        if segment_mode == "S":
            segment_acceleration = sqrt(tangential_acceleration**2 + gravity**2)
        else:
            segment_acceleration = sqrt(tangential_acceleration**2 + gravity**2 + centripetal_acceleration**2)

        segment_power_consumption = segment_acceleration
        # segment_energy_consumption = abs(v_end -  1) + 0.005 # For debugging only
        segment_energy_consumption = quad(lambda t: segment_power_consumption, 0, time)[0]

        energy_cost += segment_energy_consumption
        segment_energy_costs.append(segment_energy_consumption)
        segment_power_costs.append(segment_power_consumption)

    return energy_cost, segment_energy_costs, segment_power_costs

"""
# THIRD TRIAL:
# - Add curvature to grid based processing as well (3 inner loops in meshgrid)
# - Try doing it in batches with multiprocessing

def _worker_function_batch(planner, alpha, beta, hypot_dx_dy, v1, v4, batch):
    '''
    Batch worker function for multiprocessing that can be used by multiple processes.
    
    Parameters
    ----------
    planner, alpha, beta, hypot_dx_dy, v1, v4, batch
        
    Returns
    -------
    tuple
        (energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur)
    '''
    results = np.empty((len(batch), 10), dtype=object)  
    for i, c in enumerate(batch):
        v2_val, v3_val, cur_val = c
        d1, d2, d3, mode = planner(alpha, beta, hypot_dx_dy * cur_val)
        if d1 is None:
            results[i] = (float('inf'), None, None, None, None, None, None, None, None, None)
            continue
        
        energy_cost, segment_energy_costs, segment_power_costs = _calculate_energy_cost(d1, d2, d3, mode, v1, v2_val, v3_val, v4, cur_val)
        results[i] = (energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur_val)

    return results

def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, start_velocity, end_velocity, curvature_search_range,
                                      step_size, planning_funcs, velocity_range, velocity_step):
    
    v1, v4 = start_velocity, end_velocity
    dx = end_x
    dy = end_y
    hypot_dx_dy = hypot(dx, dy)

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    best_result = None
    best_segment_energy_costs = []
    best_segment_power_costs = []
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    velocity_search_range = np.arange(velocity_range[0], velocity_range[-1]+velocity_step, velocity_step)
    v2, v3, cur = np.meshgrid(velocity_search_range, velocity_search_range, curvature_search_range)
    combinations = np.column_stack((v2.ravel(), v3.ravel(), cur.ravel()))

    batch_size = len(combinations) // 12  # must be multiple of 3
    print(f"Batch size: {batch_size}, ", f"len(combinations): {len(combinations)}")

    for planner in planning_funcs:
        with multiprocessing.Pool(processes=15) as pool:
            results = []
            for i in range(0, len(combinations), batch_size):
                batch = combinations[i:i + batch_size]
                result = pool.apply_async(_worker_function_batch, (planner, alpha, beta, hypot_dx_dy, v1, v4, batch))
                results.append(result)

            all_results = np.empty((len(combinations), 10), dtype=object)  # Preallocate for all results
            for idx, result in enumerate(results):
                start_index = idx * batch_size
                all_results[start_index : start_index + len(result.get())] = result.get()

            # Find the result with the lowest cost
            lowest_cost_result = min(all_results, key=lambda x: x[0])
            current_lowest_cost = lowest_cost_result[0]

            if current_lowest_cost < best_cost:
                best_cost = current_lowest_cost
                best_result = lowest_cost_result

    best_cost, best_segment_energy_costs, best_segment_power_costs, b_d1, b_d2, b_d3, b_mode, b_v2, b_v3, b_cur = best_result
    segment_velocities = [v1, b_v2, b_v3, v4]
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list, velocity_list, lsegment_coordinates = _generate_local_course(lengths, b_mode, b_cur, step_size, segment_velocities)
    lengths = [length / b_cur for length in lengths]

    return x_list, y_list, yaw_list, velocity_list, b_mode, lengths, segment_velocities, lsegment_coordinates, best_segment_energy_costs, best_segment_power_costs
"""

"""
# SECOND TRIAL:
# - Add curvature to grid based processing as well (3 inner loops in meshgrid)

def _worker_function(planner, alpha, beta, hypot_dx_dy, v1, v2_val, v3_val, v4, cur_val):
    '''
    Worker function for multiprocessing that can be used by multiple processes.
    
    Parameters
    ----------
    planner, alpha, beta, hypot_dx_dy, v1, v2_val, v3_val, v4, cur_val
        
    Returns
    -------
    tuple
        (energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur)
    '''
    
    d1, d2, d3, mode = planner(alpha, beta, hypot_dx_dy * cur_val)
    if d1 is None:
        return float('inf'), None
    
    # Return all the data needed to update the shared data
    energy_cost, segment_energy_costs, segment_power_costs = _calculate_energy_cost(d1, d2, d3, mode, v1, v2_val, v3_val, v4, cur_val)

    return(energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur_val)


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, start_velocity, end_velocity, curvature_search_range,
                                      step_size, planning_funcs, velocity_range, velocity_step):
    
    v1, v4 = start_velocity, end_velocity
    dx = end_x
    dy = end_y
    hypot_dx_dy = hypot(dx, dy)

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    best_result = None
    best_segment_energy_costs = []
    best_segment_power_costs = []
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    velocity_search_range = np.arange(velocity_range[0], velocity_range[-1]+velocity_step, velocity_step)
    v2, v3, cur = np.meshgrid(velocity_search_range, velocity_search_range, curvature_search_range)
    combinations = np.column_stack((v2.ravel(), v3.ravel(), cur.ravel()))

    for planner in planning_funcs:
        with multiprocessing.Pool(processes=14) as pool:
            inputs = [(planner, alpha, beta, hypot_dx_dy, v1, c[0], c[1], v4, c[2]) for c in combinations]
            
            results = pool.starmap(_worker_function, inputs)

            # Find the result with the lowest cost
            lowest_cost_result = min(results, key=lambda x: x[0])
            current_lowest_cost = lowest_cost_result[0]

            if current_lowest_cost < best_cost:
                best_cost = current_lowest_cost
                best_result = lowest_cost_result

    best_cost, best_segment_energy_costs, best_segment_power_costs, b_d1, b_d2, b_d3, b_mode, b_v2, b_v3, b_cur = best_result
    segment_velocities = [v1, b_v2, b_v3, v4]
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list, velocity_list, lsegment_coordinates = _generate_local_course(lengths, b_mode, b_cur, step_size, segment_velocities)
    lengths = [length / b_cur for length in lengths]

    return x_list, y_list, yaw_list, velocity_list, b_mode, lengths, segment_velocities, lsegment_coordinates, best_segment_energy_costs, best_segment_power_costs
"""

"""
# FIRST TRIAL:

# Lessons learned:
# - multiprocessing is not faster than single core --> Likely because of the small size of the velocity space (small number of tasks)
# - using meshgrid is not faster than just iterating over the curvature search range 
# - using multiprocessing.Pool(processes=3) is faster than multiprocessing.Pool(processes=10)

def _worker_function(d1, d2, d3, mode, v1, v2_val, v3_val, v4, cur):
    '''
    Worker function for multiprocessing that can be used by multiple processes.
    
    Parameters
    ----------
    d1, d2, d3, mode, v1, v2_val, v3_val, v4, cur
        
    Returns
    -------
    tuple
        (energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur)
    '''
    
    # Return all the data needed to update the shared data
    energy_cost, segment_energy_costs, segment_power_costs = _calculate_energy_cost(d1, d2, d3, mode, v1, v2_val, v3_val, v4, cur)

    return(energy_cost, segment_energy_costs, segment_power_costs, d1, d2, d3, mode, v2_val, v3_val, cur)

def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, start_velocity, end_velocity, curvature_search_range,
                                      step_size, planning_funcs, velocity_range, velocity_step):
    
    v1, v4 = start_velocity, end_velocity
    dx = end_x
    dy = end_y
    hypot_dx_dy = hypot(dx, dy)

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    best_result = None
    best_segment_energy_costs = []
    best_segment_power_costs = []
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    # TODO: Obviously optimize this entire search below :)
    # TODO: Try meshgrid opt -> either numpy or pytorch for use of even gpu possibly

    velocity_search_range = np.arange(velocity_range[0], velocity_range[-1]+velocity_step, velocity_step)
    v2, v3 = np.meshgrid(velocity_search_range, velocity_search_range)
    velocity_pairs = np.column_stack((v2.ravel(), v3.ravel()))

    for planner in planning_funcs: 

        # Search along different radius sizes for turns within path
        # 1 fixed radius for a whole path type (= for both turns of a path)

        # tuples = [(*planner(alpha, beta, hypot_dx_dy * cur), cur) for cur in curvature_search_range]
        # for d1, d2, d3, mode, cur in tuples:
        #     if d1 is None:
        #         continue

        for cur in curvature_search_range:
            d1, d2, d3, mode = planner(alpha, beta, hypot_dx_dy * cur)
            if d1 is None:
                continue

            # Sample the intermediate velocities within path
            with multiprocessing.Pool(processes=3) as pool:
                inputs = [(d1, d2, d3, mode, v1, pair[0], pair[1], v4, cur) for pair in velocity_pairs]
                
                results = pool.starmap(_worker_function, inputs)

                # Find the result with the lowest cost
                lowest_cost_result = min(results, key=lambda x: x[0])
                current_lowest_cost = lowest_cost_result[0]

                if current_lowest_cost < best_cost:
                    best_cost = current_lowest_cost
                    best_result = lowest_cost_result

    best_cost, best_segment_energy_costs, best_segment_power_costs, b_d1, b_d2, b_d3, b_mode, b_v2, b_v3, b_cur = best_result
    segment_velocities = [v1, b_v2, b_v3, v4]
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list, velocity_list, lsegment_coordinates = _generate_local_course(lengths, b_mode, b_cur, step_size, segment_velocities)
    lengths = [length / b_cur for length in lengths]

    return x_list, y_list, yaw_list, velocity_list, b_mode, lengths, segment_velocities, lsegment_coordinates, best_segment_energy_costs, best_segment_power_costs
"""


# BEFORE:
# Baseline Configs:
# curvature_search_range = (0.6, 1.0)
# velocity_range = (0.5, 6.0)  
# velocity_step = 0.5
# Baseline Runtime: 0.01579 seconds, Best cost: 25.2071

def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, start_velocity, end_velocity, curvature_search_range,
                                      step_size, planning_funcs, velocity_range, velocity_step):
    
    v1, v4 = start_velocity, end_velocity
    dx = end_x
    dy = end_y

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    best_segment_energy_costs = []
    best_segment_power_costs = []
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    # TODO: Obviously optimize this entire search below :)
    # TODO: Try meshgrid opt -> either numpy or pytorch for use of even gpu possibly
    for planner in planning_funcs: 

        # Search along different radius sizes for turns within path
        # 1 fixed radius for a whole path type (= for both turns of a path)

        for cur in curvature_search_range:
            d = hypot(dx, dy) * cur

            d1, d2, d3, mode = planner(alpha, beta, d)
            if d1 is None:
                continue

            # Sample the intermediate velocities within path
            velocity_search_range = np.arange(velocity_range[0], velocity_range[-1]+velocity_step, velocity_step)
            for v2 in velocity_search_range:
                for v3 in velocity_search_range:
                    # Calculate Power-based cost
                    cost, segment_energy_costs, segment_power_costs = _calculate_energy_cost(
                        d1, d2, d3, mode, v1, v2, v3, v4, cur)

                    if best_cost > cost:  # Select minimum energy cost path
                        b_d1, b_d2, b_d3, b_v2, b_v3, b_mode, b_cur, best_cost, best_segment_energy_costs, best_segment_power_costs = d1, d2, d3, v2, v3, mode, cur, cost, segment_energy_costs, segment_power_costs

    segment_velocities = [v1, b_v2, b_v3, v4]
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list, velocity_list, lsegment_coordinates = _generate_local_course(lengths, b_mode, b_cur, step_size, segment_velocities)
    lengths = [length / b_cur for length in lengths]

    return x_list, y_list, yaw_list, velocity_list, b_mode, lengths, segment_velocities, lsegment_coordinates, best_segment_energy_costs, best_segment_power_costs


def interpolate_velocities(p_x, p_y, segment_velocities, segment_coordinates_x, segment_coordinates_y, max_acc=1.0):
    """
    Interpolates velocities along a path based on segment velocities and segment coordinates,
    taking into account acceleration and deceleration limits.

    Args:
        p_x: List of x coordinates of the path.
        p_y: List of y coordinates of the path.
        segment_velocities: List of 4 velocities at specific segments.
        segment_coordinates_x: List of 4 x coordinates corresponding to segment_velocities.
        segment_coordinates_y: List of 4 y coordinates corresponding to segment_velocities.
        max_acc: Maximum acceleration/deceleration allowed.

    Returns:
        interpolated_velocities: List of velocities along the path.
    """

    path_len = len(p_x)
    interpolated_velocities = np.zeros(path_len)

    # Calculate cumulative distance along the path
    path_s = np.zeros(path_len)
    for i in range(1, path_len):
        dx = p_x[i] - p_x[i-1]
        dy = p_y[i] - p_y[i-1]
        path_s[i] = path_s[i-1] + np.hypot(dx, dy)

    # Calculate s positions of velocity segments
    segment_s = []
    for sx, sy in zip(segment_coordinates_x, segment_coordinates_y):
        # Find closest point on path
        dists = np.hypot(np.array(p_x) - sx, np.array(p_y) - sy)
        closest_idx = np.argmin(dists)
        segment_s.append(path_s[closest_idx])

    # Interpolate target velocities linearly over the segments
    try:
        interpolated_target_velocities = np.interp(path_s, segment_s, segment_velocities)
    except:
        print(f"segment_coordinates_x: {segment_coordinates_x}")
        print(f"segment_coordinates_y: {segment_coordinates_y}")
        print(f"segment_s: {segment_s}")
        print(f"path_s: {path_s}")
        print(f"segment_velocities: {segment_velocities}")
        raise Exception("Error interpolating velocities")

    # Forward pass (acceleration limit)
    interpolated_velocities[0] = interpolated_target_velocities[0]
    for i in range(1, path_len):
        ds = path_s[i] - path_s[i-1]
        v_prev = interpolated_velocities[i-1]
        v_target = interpolated_target_velocities[i]
        v_possible = np.sqrt(v_prev**2 + 2 * max_acc * ds)
        interpolated_velocities[i] = min(v_possible, v_target)

    # Backward pass (deceleration limit)
    for i in range(path_len-2, -1, -1):
        ds = path_s[i+1] - path_s[i]
        v_next = interpolated_velocities[i+1]
        v_target = interpolated_velocities[i]
        v_possible = np.sqrt(v_next**2 + 2 * max_acc * ds)
        interpolated_velocities[i] = min(v_target, v_possible)

    return interpolated_velocities.tolist()

def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size, segment_velocities):
    p_x, p_y, p_yaw, p_velocity = [0.0], [0.0], [0.0], [0.0]
    lsegment_coordinates_x, lsegment_coordinates_y = [], []

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]
        lsegment_coordinates_x.append(origin_x)
        lsegment_coordinates_y.append(origin_y)

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)
    
    lsegment_coordinates_x.append(p_x[-1])
    lsegment_coordinates_y.append(p_y[-1])
    
    if p_x != [0.0]:
        segment_velocities = segment_velocities[:len(lsegment_coordinates_x)]

        p_velocity = interpolate_velocities(
            p_x, p_y, segment_velocities, lsegment_coordinates_x, lsegment_coordinates_y)

    return p_x, p_y, p_yaw, p_velocity, (lsegment_coordinates_x[1:], lsegment_coordinates_y[1:])


def main():
    print("Dubins path planner sample start!!")
    import matplotlib.pyplot as plt
    from utils.plot import plot_arrow
    import time

    start_x = 1.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.deg2rad(45.0)  # [rad]
    start_velocity = 1.5

    end_x = -3.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.deg2rad(-45.0)  # [rad]
    end_velocity = 1.5

    curvature = 1.0

    curvature_search_range = (0.6, 1.0)
    velocity_range = (0.5, 6.0)  # [m/s]
    velocity_step = 0.5  # [m/s]

    step_size=0.05

    # Distance based
    from PathPlanning.DubinsPath import dubins_path_planner
    path_x, path_y, path_yaw, mode, lengths, energy_cost, segment_energy_costs, segment_power_costs = dubins_path_planner.plan_dubins_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size=step_size)

    # Energy based
    start_time = time.time()
    energy_path_x, energy_path_y, energy_path_yaw, energy_velocity_list, \
        energy_mode, energy_lengths, segment_velocities, segment_coordinates, \
            segment_energy_costs, segment_power_costs = plan_dubins_path(
        start_x, start_y, start_yaw, start_velocity,
        end_x, end_y, end_yaw, end_velocity,
        curvature_search_range, selected_types=None,
        velocity_range=velocity_range, velocity_step=velocity_step)
    end_time = time.time()
    print(f"Energy based path planning time: {round(end_time - start_time, 5)} seconds, Best cost: {round(sum(segment_energy_costs), 5)}")

    # Baseline Configs:
    # curvature_search_range = (0.6, 1.0)
    # velocity_range = (0.5, 6.0)  # [m/s]
    # velocity_step = 0.5  # [m/s]
    # Baseline Runtime: 0.01579 seconds, Best cost: 25.2071

    print(
        f"Optimal end velocitities for each energy efficient path: {[round(v,2) for v in segment_velocities]} m/s")

    if show_animation:
        plt.figure(figsize=(10, 5))

        # Plot distance-based path
        plt.subplot(121)
        plt.plot(path_x, path_y, label=f"Path type: {''.join(mode)}")
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.title("Distance-optimized path")

        # Plot energy-based path
        plt.subplot(122)
        plt.plot(energy_path_x, energy_path_y,
                 label=f"Path type: {''.join(energy_mode)}, v={[round(v,2) for v in segment_velocities]} m/s")
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.title("energy-optimized path with optimal velocity")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
