import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time
import cv2
import imageio

# Record the start time for performance measurement
startT = time.time()

# Calculate an angle for Hexagon
angle = 150 * np.cos(np.pi / 6)

# Define obstacle blocks
Obstacles_Blocks = [
    np.array([(100, 100), (100, 500), (175, 500), (175, 100)]),
    np.array([(275, 0), (275, 400), (350, 400), (350, 0)]),
    np.array([
        (650 - angle, 400 - 150 - 150 * 0.5),
        (650 - angle, 400 - 150 * 0.5),
        (650, 400),
        (650 + angle, 400 - 150 * 0.5),
        (650 + angle, 400 - 150 - 150 * 0.5),
        (650, 100),
    ]),
    np.array([(900, 450), (1100, 450), (1100, 50), (900, 50), (900, 125), (1020, 125), (1020, 375), (900, 375)]),
]

# Convert the obstacle blocks to NumPy arrays for better performance
Obstacles_Blocks = [np.array(space) for space in Obstacles_Blocks]

# Tolerance_Space is a threshold given
Tolerance_Space = 5

# Maximum x and y coordinates for the Map
max_x, max_y = 1200, 500

def add_values_pos():
    # Continuously prompt the user for initial and goal node coordinates
    while True:
        try:
            # Get input for initial and goal node coordinates
            x_initial, y_initial = map(int, input("Initial Node (X Y) (Separate By Space): ").split())
            x_goal, y_goal = map(int, input("Goal Node (X Y) (Separate By Space): ").split())

            # Check if the input coordinates are valid using the points_okay function
            if not points_okay(x_initial, y_initial) or not points_okay(x_goal, y_goal):
                print("Not Valid Position")
            else:
                # If the coordinates are valid, return the tuple of initial and goal nodes
                return (x_initial, y_initial), (x_goal, y_goal)
        except ValueError:
            # Handle ValueError (non-integer input) and provide appropriate instructions
            print('''
                  Enter Appropriate Values in the range of 0-1200 for X
                  and 0-500 for Y''')


def points_okay(x, y):
    # Check if the point is inside any obstacle space using point_inside_polygon function
    for obstacle in Obstacles_Blocks:
        if point_inside_polygon(x, y, obstacle):
            print("Point is in obstacle space.")
            return False
    
    # Check if the point is within the valid range and not too close to the environment boundaries
    return Tolerance_Space <= x <= max_x - Tolerance_Space and Tolerance_Space <= y <= max_y - Tolerance_Space

def point_inside_polygon(x, y, space):
    # Determine if a point (x, y) is inside a polygon defined by space
    n = len(space)
    is_inside = False
    x, y = float(x), float(y)
    
    point1x, point1y = space[0]
    
    for i in range(1, n + 1):
        point2x, point2y = space[i % n]
        
        # Check if the point is within the y-bounds of the current line segment
        if y > min(point1y, point2y) and y <= max(point1y, point2y) and x <= max(point1x, point2x):
            # Check if the point is to the left of the intersection with the line segment
            if point1y != point2y:
                intersection = (y - point1y) * (point2x - point1x) / (point2y - point1y) + point1x
                if point1x == point2x or x <= intersection:
                    is_inside = not is_inside
        
        point1x, point1y = point2x, point2y
    return is_inside

def distance_to_nearest_obstacle(pointX, pointY):
    # Iterate through each edge of the polygon
    for space in Obstacles_Blocks:
        for i in range(len(space)):
            point1, point2 = space[i], space[(i + 1) % len(space)]
            
            # Calculate the length of the line segment
            map_track = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
            
            # Calculate the unit vector along the line
            track_vector = [(point2[0] - point1[0]) / map_track, (point2[1] - point1[1]) / map_track]
            
            # Calculate the projected length of the point onto the line
            proj_length = max(0, min(map_track, (pointX - point1[0]) * track_vector[0] + (pointY - point1[1]) * track_vector[1]))
            
            # Calculate the nearest point on the line
            nearest = [point1[0] + track_vector[0] * proj_length, point1[1] + track_vector[1] * proj_length]
            
            # Calculate the distance between the point and the nearest point on the line
            distance_calc = ((pointX - nearest[0]) ** 2 + (pointY - nearest[1]) ** 2) ** 0.5
            
            # Check if the distance is within the tolerance
            if Tolerance_Space > distance_calc:
                return True
    
    # If no obstacle is too close, return False
    return False

# Will use the below function in Dijsktra
def obstacle_block_check(x, y):
    # Check if the point is outside the map boundaries
    if x < Tolerance_Space or y < Tolerance_Space or x > max_x - Tolerance_Space or y > max_y - Tolerance_Space:
        return True
    
    # Check if the point is too close to any obstacle using distance_to_nearest_obstacle function
    return distance_to_nearest_obstacle(x, y)


def Dijsktra_Algo(startNode, goalNode):
    # Initialize data structures for Dijkstra's algorithm
    predecessors = {}
    open_list = []
    closed_list = []
    actions = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.4), (-1, -1, 1.4), (1, -1, 1.4), (-1, 1, 1.4)]
    heapq.heappush(open_list, (0, startNode))
    updated_cost = {startNode: 0}

    # Run Dijkstra's algorithm
    while open_list:
        _, currentNode = heapq.heappop(open_list)
        closed_list.append(currentNode)

        if currentNode == goalNode:
            break

        # Explore adjacent cells and update costs
        for new_x, new_y, take_cost in actions:
            adjacentcells = (currentNode[0] + new_x, currentNode[1] + new_y)
            if max_y >= adjacentcells[1] >= 0 and 0 <= adjacentcells[0] <= max_x:
                costtocome_new = take_cost + updated_cost[currentNode] 
                if adjacentcells not in updated_cost or costtocome_new < updated_cost[adjacentcells]:
                    if not obstacle_block_check(adjacentcells[0], adjacentcells[1]):
                        updated_cost[adjacentcells] = costtocome_new
                        pref = costtocome_new
                        heapq.heappush(open_list, (pref, adjacentcells))
                        predecessors[adjacentcells] = currentNode

    # Reconstruct the path from goal to start
    currentNode = goalNode
    final_path = []
    while currentNode != startNode:
        final_path.append(currentNode)
        currentNode = predecessors.get(currentNode, startNode)
    final_path.append(startNode)
    final_path.reverse()

    return final_path, closed_list

def ani_explorationpath_path(closed_list, final_path, startNode, goalNode):
    # Create a figure with two subplots
    figure, (ani_point1, ani_point2) = plt.subplots(1, 2, figsize=(18, 5))

    # Set the xlim and ylim for both subplots
    for fpoint in [ani_point1, ani_point2]:
        fpoint.set_xlim(0, max_x)
        fpoint.set_ylim(0, max_y)

        # Add obstacle polygons to the subplots
        for space in Obstacles_Blocks:
            poly = plt.Polygon(space, facecolor="purple", edgecolor='black')
            fpoint.add_patch(poly)

    # Initialize scatter plot and line plot objects
    points = ani_point1.scatter([], [], s=1, color='red')
    track, = ani_point2.plot([], [], 'green', linewidth=1)

    # Set the initial and goal points to blue
    start_point, = ani_point1.plot([], [], 'bo', markersize=5)
    goal_point, = ani_point1.plot([], [], 'bo', markersize=5)

    # Initialization function for both animations
    def init():
        points.set_offsets(np.empty((0, 2)))
        track.set_data([], [])
        start_point.set_data([], [])
        goal_point.set_data([], [])
        return points, track, start_point, goal_point

# Update function for exploration animation
    def exploration(frame):
        iter = 10000
        frame = frame * iter
        closed_list_points = np.array(closed_list[:frame + 1])
        
        # Set the alpha (transparency) to 0.5 for the points
        points.set_offsets(closed_list_points)
        points.set_alpha(0.5)
        
        return points,


    # Update function for final path animation
    def final_path_ani(frame):
        iter = 100
        frame = frame * iter
        x, y = zip(*final_path[:frame + 1])
        track.set_data(x, y)
        return track,

    # Update function for initial and goal points
    def update_start_goal(frame):
        iter = 100
        start_point.set_data(*startNode)
        goal_point.set_data(*goalNode)
        return start_point, goal_point

    # Create FuncAnimation objects for all animations
    ExplorationAnimation = FuncAnimation(figure, exploration, frames=len(closed_list), init_func=init, blit=True, interval=10)
    final_pathAnimation = FuncAnimation(figure, final_path_ani, frames=len(final_path), init_func=init, blit=True, interval=20)

    # Save animation frames as GIFs
    imageio.mimsave('ExplorationAnimation.gif', [closed_list_frame for closed_list_frame in closed_list], duration=0.033)
    imageio.mimsave('final_pathAnimation.gif', [final_path_frame for final_path_frame in final_path], duration=0.033)

    # Display the plot
    plt.show()

# Get Start and Goal point
startNode, goalNode = add_values_pos()

# Run Dijkstra's Algorithm to find the final path and exploration path
final_path, closed_list = Dijsktra_Algo(startNode, goalNode)

# Visualize the exploration path, final path, and initial/goal points using animations
ani_explorationpath_path(closed_list, final_path, startNode, goalNode)

# Calculate the total time taken for the algorithm
endT = time.time()
Final_Time = endT - startT
print(f"Goal found in", Final_Time)
