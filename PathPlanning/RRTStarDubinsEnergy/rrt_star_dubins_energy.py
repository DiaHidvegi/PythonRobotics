"""
Path planning Sample Code with RRT and Energy-optimized Dubins path
modified for energy optimization

"""
import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # root dir
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from PathPlanning.RRTStar.rrt_star import RRTStar
from PathPlanning.DubinsPathEnergy import dubins_path_energy_planner
from DubinsPath import dubins_path_planner
from RRTStar.rrt_star import RRTStar
from utils.plot import plot_arrow

show_animation = True


class RRTStarDubinsEnergy(RRTStar):
    """
    Class for RRT star planning with energy-optimized Dubins path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw, velocity):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []
            self.velocity = velocity

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=10,
                 max_iter=200,
                 connect_circle_dist=50.0,
                 robot_radius=0.0,
                 velocity_range=(0.5, 3.0),
                 velocity_step=0.5,
                 curvature_search_range=(0.3, 1.0, 2.0),
                 energy_weights=None):
        """
        Setting Parameter

        start:Start Position [x,y,yaw,end velocity]
        goal:Goal Position [x,y,yaw,end velocity]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        robot_radius: robot body modeled as circle with given radius
        velocity_range: range of velocities to consider (min, max) [m/s]
        velocity_step: step size for velocity optimization [m/s]
        energy_weights: weights for energy cost calculation
        """
        self.start = self.Node(start[0], start[1], start[2], start[3])
        self.end = self.Node(goal[0], goal[1], goal[2], goal[3])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist

        self.curvature = 1.0  # for dubins path
        self.goal_yaw_th = np.deg2rad(1.0)
        self.goal_xy_th = 0.5
        self.robot_radius = robot_radius

        # Parameters for energy-based planning
        self.curvature_search_range = curvature_search_range
        self.velocity_range = velocity_range
        self.velocity_step = velocity_step
        self.energy_weights = energy_weights
        if self.energy_weights is None:
            self.energy_weights = {
                'straight': 1.0,  # Base energy cost for straight segments
                'turn': 1.5,      # Higher energy cost for turning segments
                'switch': 0.5,    # Penalty for switching between modes
                'velocity': 0.1   # Weight for velocity impact on energy
            }

    def planning(self, animation=True, search_until_max_iter=True):
        """
        RRT Star planning with energy-optimized Dubins paths

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
                # Optionally display optimal velocity for each segment
                # mid_x = (node.x + node.parent.x) / 2
                # mid_y = (node.y + node.parent.y) / 2
                # if node.velocity:
                #     plt.text(mid_x, mid_y, f"{node.velocity:.1f}", fontsize=8)

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):
        """
        Generate a path from from_node to to_node using energy-optimized Dubins paths
        """
        px, py, pyaw, pvelocities, mode, course_lengths, path_cost = \
            dubins_path_energy_planner.plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw, from_node.velocity,
                to_node.x, to_node.y, to_node.yaw, to_node.velocity,
                self.curvature_search_range,
                velocity_range=self.velocity_range,
                velocity_step=self.velocity_step,
                energy_weights=self.energy_weights)

        if len(px) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]
        new_node.velocity = pvelocities[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.path_velocity = pvelocities
        new_node.cost = path_cost
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):
        """
        Calculate energy cost of new Dubins path
        """
        _, _, _, _, _, _, path_cost = dubins_path_energy_planner.plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw, from_node.velocity,
            to_node.x, to_node.y, to_node.yaw, to_node.velocity,
            self.curvature_search_range,
            velocity_range=self.velocity_range,
            velocity_step=self.velocity_step,
            energy_weights=self.energy_weights)

        return from_node.cost + path_cost

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-math.pi, math.pi),
                            random.sample(list(np.arange(self.velocity_range[0],
                                                    self.velocity_range[-1]+self.velocity_step,
                                                    self.velocity_step)), 1)[0]

                            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw, self.end.velocity)

        return rnd

    def search_best_goal_node(self):
        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:  # TODO check if this works with fully energy based cost 
                print(f"self.node_list[i].cost == min_cost: {self.node_list[i].cost == min_cost}")
                return i

        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y]]
        velocities = []
        node = self.node_list[goal_index]

        # Collect velocities along the path
        current_node = node
        while current_node.parent:
            if current_node.velocity:
                velocities.append(current_node.velocity)
            current_node = current_node.parent

        # Reverse velocities to match path order
        velocities = list(reversed(velocities))

        # Build the path
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.start.x, self.start.y])

        # Print the velocity profile if available
        if velocities:
            print("Velocity profile along the path:")
            for i, v in enumerate(velocities):
                print(f"Segment {i+1}: {v:.2f} m/s")

        return path, velocities

def show_final_2D_trajectory(fig, pos, rrtstar_dubins: RRTStar, obstacleList, path_found, title: str):
    ax = fig.add_subplot(pos)
    for node in rrtstar_dubins.node_list:
        if node.parent:
            plt.plot(node.path_x, node.path_y, "-g", alpha=0.3)
    for (ox, oy, size) in obstacleList:
        plt.plot(ox, oy, "ok", ms=30 * size)
    plt.plot(rrtstar_dubins.start.x, rrtstar_dubins.start.y, "xr")
    plt.plot(rrtstar_dubins.end.x, rrtstar_dubins.end.y, "xr")
    plot_arrow(rrtstar_dubins.start.x, rrtstar_dubins.start.y, rrtstar_dubins.start.yaw)
    plot_arrow(rrtstar_dubins.end.x, rrtstar_dubins.end.y, rrtstar_dubins.end.yaw)
    plt.plot([x for (x, y) in path_found], [y for (x, y) in path_found], '-r', linewidth=2)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    return ax

def main():
    print("Start RRT star with energy-optimized Dubins planning")

    # ====Search Path with RRT====
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size(radius)]

    start = [0.0, 0.0, np.deg2rad(0.0), 0.0]
    goal = [10.0, 10.0, np.deg2rad(0.0), 0.0]

    energy_weights = {
        'straight': 1.0,  # Standard straight segments
        'turn': 2.0,      # Higher energy cost for turning segments
        'switch': 1.0,    # Standard penalty for switching between modes
        'velocity': 0.1   # Standard velocity weight
    }
    
    curvature_search_range = (0.6, 1.0, 1.5)
    velocity_range = (0.5, 3.0)  # [m/s]
    velocity_step = 0.5  # [m/s]

    # Run planners
    # from PathPlanning.RRTStarDubins.rrt_star_dubins import RRTStarDubins
    # print("Running standard distance-based RRT* Dubins...")
    # rrtstar_dubins = RRTStarDubins(
    #     start, goal, rand_area=[-2.0, 15.0], obstacle_list=obstacleList)
    # standard_path = rrtstar_dubins.planning(animation=show_animation)

    print("Running energy-optimized RRT* Dubins...")
    rrtstar_dubins_energy = RRTStarDubinsEnergy(
        start, goal, rand_area=[-2.0, 15.0], obstacle_list=obstacleList,
        velocity_range=velocity_range, velocity_step=velocity_step,
        curvature_search_range=curvature_search_range,
        energy_weights=energy_weights)
    energy_path, velocities = rrtstar_dubins_energy.planning(animation=show_animation)

    if show_animation:
        fig = plt.figure(figsize=(10, 5))
        
        # Plot both RRT graphs and optimal paths side by side
        # ax1 = show_final_2D_trajectory(fig, 121, rrtstar_dubins, obstacleList, standard_path, "Distance-based Path")
        ax2 = show_final_2D_trajectory(fig, 122, rrtstar_dubins_energy, obstacleList, energy_path, "Energy-optimized Path")
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
