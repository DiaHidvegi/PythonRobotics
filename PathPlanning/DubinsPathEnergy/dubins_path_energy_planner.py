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
    path_cost: float
        cost of Dubin path (summed cost of 3 segments)
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
    lp_x, lp_y, lp_yaw, velocity_list, modes, lengths, segment_velocities, lsegment_coordinates, path_cost = _dubins_path_planning_from_origin(
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

    return x_list, y_list, yaw_list, velocity_list, modes, lengths, segment_velocities, segment_coordinates, path_cost

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
    """
    if d1 is None:
        return float('inf')

    segments = [d1, d2, d3]
    velocities = [v1, v2, v3, v4]
    radius = 1 / curvature
    energy_cost = 0.0
    segment_costs = []
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

        # def time_integrand(time):
        #     pass

        # power = segment_acceleration
        segment_energy_consumption = quad(lambda t: segment_acceleration, 0, time)[0]
        # segment_energy_consumption = abs(v_end -  1) + 0.005
        energy_cost += segment_energy_consumption
        segment_costs.append(segment_energy_consumption)

    return energy_cost, segment_costs

def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, start_velocity, end_velocity, curvature_search_range,
                                      step_size, planning_funcs, velocity_range, velocity_step):
    
    v1, v4 = start_velocity, end_velocity
    dx = end_x
    dy = end_y

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    best_segment_costs = []
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    # TODO: Obviously optimize this entire search below :)
    # TODO: Try meshgrid opt -> either numpy or pytorch for use of even gpu possibly
    for planner in planning_funcs:

        # Search along different radius sizes for turns within path
        # 1 fixed radius for a whole path type (= for both turns of a path)

        # curvature_tuples = [(tup, cur) for cur in curvature_search_range if (tup := planner(alpha, beta, hypot(dx, dy) * cur))[0] is not None]
        # for tup, cur in curvature_tuples:
        #     d1, d2, d3, mode = tup

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
                    cost, segment_costs = _calculate_energy_cost(
                        d1, d2, d3, mode, v1, v2, v3, v4, cur)

                    if best_cost > cost:  # Select minimum energy cost path
                        b_d1, b_d2, b_d3, b_v2, b_v3, b_mode, b_cur, best_cost, best_segment_costs = d1, d2, d3, v2, v3, mode, cur, cost, segment_costs

    segment_velocities = [v1, b_v2, b_v3, v4]
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list, velocity_list, lsegment_coordinates = _generate_local_course(lengths, b_mode, b_cur, step_size, segment_velocities)
    lengths = [length / b_cur for length in lengths]

    return x_list, y_list, yaw_list, velocity_list, b_mode, lengths, segment_velocities, lsegment_coordinates, best_segment_costs

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

    start_x = 1.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.deg2rad(45.0)  # [rad]
    start_velocity = 1.5

    end_x = -3.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.deg2rad(-45.0)  # [rad]
    end_velocity = 1.5

    curvature = 1.0

    curvature_search_range = [0.3, 1.0, 1.5]
    velocity_range = (0.5, 3.0)  # [m/s]cur
    velocity_step = 0.5  # [m/s]
    step_size=0.05

    # Distance based
    from PathPlanning.DubinsPath import dubins_path_planner
    path_x, path_y, path_yaw, mode, lengths, energy_cost, segment_energy_costs = dubins_path_planner.plan_dubins_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size=step_size)

    # Energy based
    energy_path_x, energy_path_y, energy_path_yaw, energy_velocity_list, energy_mode, energy_lengths, segment_velocities, segment_coordinates, path_cost = plan_dubins_path(
        start_x, start_y, start_yaw, start_velocity,
        end_x, end_y, end_yaw, end_velocity,
        curvature_search_range, step_size=step_size, selected_types=None,
        velocity_range=velocity_range, velocity_step=velocity_step)

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
