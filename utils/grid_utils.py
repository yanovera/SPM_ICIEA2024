import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.spatial import distance
from utils.grid_search_algs import get_neighbors
from collections import deque


def load_decomposition(filename):
    """Load the decomposition from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def point_to_cell_coordinates(point, resolution, lower_limit=-np.pi):
    x, y = point[0], point[1]
    return (int((x - lower_limit) / resolution),
            int((y - lower_limit) / resolution))


def to_2d_coordinates(projected_point, u, v):
    """
    Convert the 3D projected point to 2D coordinates in the plane defined by vectors u and v.

    Parameters:
        projected_point (tuple): The coordinates of the projected point in 3D space as (x, y, z).
        u (np.array): The first basis vector of the plane.
        v (np.array): The second basis vector of the plane.

    Returns:
        tuple: The 2D coordinates of the projected point in the plane.
    """
    p = np.array(projected_point)
    alpha = np.dot(p, u) / np.dot(u, u)
    beta = np.dot(p, v) / np.dot(v, v)

    return (alpha, beta)


def project_point_to_2D(point):
    """
    Project a 3D point onto the plane with normal vector (1, 1, 1) and return its 2D coordinates.

    Parameters:
        point (tuple): The coordinates of the point in 3D space as (x, y, z).

    Returns:
        tuple: The 2D coordinates of the projected point in the plane.
    """
    # 3D coordinates of the projected point
    projected_point_3D = project_point_onto_plane(point)

    # Normal vector to the plane (1, 1, 1)
    n = np.array([1, 1, 1])

    # Calculate the unit normal vector
    n_hat = n / np.linalg.norm(n)

    # Take any vector not parallel to n (for example, (1, 0, 0))
    a = np.array([1, 0, 0])

    # Compute the first basis vector of the plane
    u = a - np.dot(a, n_hat) * n_hat

    # Compute the second basis vector of the plane
    v = np.cross(n_hat, u)

    # Convert the 3D projected point to 2D coordinates in the plane
    projected_point_2D = to_2d_coordinates(projected_point_3D, u, v)

    return projected_point_2D


def project_point_onto_plane(point):
    """
    Project a point onto the plane with normal vector (1, 1, 1).

    Parameters:
        point (tuple): The coordinates of the point in 3D space as (x, y, z).

    Returns:
        tuple: The coordinates of the projected point onto the plane as (x, y, z).
    """
    # Point P
    P = np.array(point)

    # Point P0 (a point on the plane, chosen to be the origin for simplicity)
    P0 = np.array([0, 0, 0])

    # Normal vector to the plane (1, 1, 1)
    n = np.array([1, 1, 1])

    # Calculate the unit normal vector
    n_hat = n / np.linalg.norm(n)

    # Calculate the projected point using the formula
    P_proj = P - np.dot(P - P0, n_hat) * n_hat

    return tuple(P_proj)


def save_decomposition(decomposition, filename):
    """Save the decomposition to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(decomposition, file)


def plot_2d_decomposition_with_borders(array_2d, xlabel, ylabel, show=True):
    """
    Plot a 2D decomposition grid using matplotlib, with black borders between cells.

    Parameters:
    - array_2d: 2D numpy array representing the decomposition

    Activated cells are those with a value larger than zero and will be colored in blue.
    Deactivated cells will be colored in red.
    """
    plt.figure(figsize=(6, 6))

    for i in range(array_2d.shape[0]):
        for j in range(array_2d.shape[1]):
            edgecolor = 'black'
            if array_2d[i, j] > 0:
                plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fc='blue', edgecolor=edgecolor))
            else:
                plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fc='red', edgecolor=edgecolor))

    plt.axis('scaled')
    plt.xlim(0, array_2d.shape[0])
    plt.ylim(0, array_2d.shape[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()  # Invert the y-axis to match the array layout
    if show:
        plt.show()


def create_2d_decomposition(resolution, lower_limit=-np.pi, upper_limit=np.pi):

    # Compute the number of cells in each dimension
    space_size = upper_limit - lower_limit
    number_of_cells = int(space_size / resolution)

    # Create a 3D array initialized with zeros (all cells deactivated)
    decomposition = np.zeros((number_of_cells, number_of_cells), dtype=float)

    return decomposition


def activate_cell(decomposition, x, y, z):
    decomposition[x, y, z] = 1


def deactivate_cell(decomposition, x, y, z):
    decomposition[x, y, z] = 0


def deactivate_cells_below_threshold(decomposition, threshold=0.05):
    # Find cells with values less than the threshold and not equal to zero
    condition = (decomposition < threshold) & (decomposition > 0)

    # Set those cells to zero (deactivate)
    decomposition[condition] = 0

    return decomposition


def deactivate_isolated_cells(grid):
    """
    Deactivate all cells that are isolated from the middle cell in a 2D grid.

    Parameters:
        grid (numpy.ndarray): 2D grid representing the decomposition.

    Returns:
        numpy.ndarray: Modified grid with isolated cells deactivated.
    """
    # Label connected components in the grid
    labeled_grid, num_features = label(grid)

    # Find the label of the middle cell
    middle_cell = (grid.shape[0] // 2, grid.shape[1] // 2)
    middle_label = labeled_grid[middle_cell]

    # Deactivate all cells that don't have the same label as the middle cell
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if labeled_grid[i, j] != middle_label:
                grid[i, j] = 0

    return grid


def cell_to_world_coordinates_centered(cell_coords, resolution, lower_limit=-np.pi):
    """Convert cell coordinates to world coordinates, referencing the center of the cell."""
    world_coords = [(c * resolution) + lower_limit + (resolution / 2) for c in cell_coords]
    return tuple(world_coords)


def path_to_world_coordinates_centered(path, resolution):
    """Convert a path in cell coordinates to world coordinates, referencing the center of each cell."""
    return [cell_to_world_coordinates_centered(cell, resolution) for cell in path]


def plot_path_on_grid(grid, path, show=False):
    """
    Plot a given path on a 2D grid.

    Parameters:
        grid (numpy.ndarray): 2D grid representing the decomposition.
        path (list): List of tuples representing the path.

    """
    fig, ax = plt.subplots()

    # Plot the grid
    ax.imshow(grid, cmap='gray_r', origin='upper')

    # Extract x and y coordinates from the path
    x_coords = [x for x, y in path]
    y_coords = [y for x, y in path]

    # Plot the path
    ax.plot(y_coords, x_coords, color='blue', marker='o')

    # Add grid lines
    ax.grid(which='both')

    # Add start and end markers
    if path:
        ax.text(y_coords[0], x_coords[0], 'S', color='green', fontsize=12, ha='center', va='center')
        ax.text(y_coords[-1], x_coords[-1], 'E', color='red', fontsize=12, ha='center', va='center')

    if show:
        plt.show()


def closest_activated_cell(grid, point, resolution, lower_limit=-np.pi):
    """
    Find the closest activated cell to a given point (in world coordinates) in a 2D grid.
    This optimized version starts searching from the neighbors of cell_point.

    Parameters:
        grid (numpy.ndarray): 2D grid representing the decomposition.
        point (tuple): World coordinates of the point (x, y).
        resolution (float): The resolution of the grid.
        lower_limit (float): The lower limit of the grid.

    Returns:
        tuple: The indices closest activated cell, or None if no activated cell is found.
    """
    # Convert the point to cell coordinates
    cell_point = point_to_cell_coordinates(point, resolution, lower_limit)

    # Initialize variables to store the closest cell and minimum distance
    closest_cell = None
    min_distance = float('inf')

    visited = set()
    queue = deque([cell_point])

    while queue:
        current = queue.popleft()
        visited.add(current)

        # Check if the current cell is within the grid boundaries
        if current[0] >= 0 and current[0] < grid.shape[0] and current[1] >= 0 and current[1] < grid.shape[1]:
            if grid[current] > 0:
                # Calculate the Euclidean distance to the point
                dist = distance.euclidean(cell_point, current)

                if dist < min_distance:
                    min_distance = dist
                    closest_cell = current
                    break  # Exit as soon as the closest activated cell is found

            neighbors = get_neighbors(current, grid.shape)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    return closest_cell


def project_point_to_3D(point_2D, lambda_value=0):
    """
    Project a point from the 2D plane back to 3D space.

    Parameters:
        point_2D (tuple): The coordinates of the point in 2D space as (alpha, beta).
        lambda_value (float): The scalar value to scale the normal vector (optional, default is 0).

    Returns:
        tuple: The coordinates of the point in 3D space as (x, y, z).
    """
    # Basis vectors and normal vector for the plane
    n = np.array([1, 1, 1])
    n_hat = n / np.linalg.norm(n)
    a = np.array([1, 0, 0])
    u = a - np.dot(a, n_hat) * n_hat
    v = np.cross(n_hat, u)

    # Convert 2D coordinates back to 3D coordinates in the plane
    alpha, beta = point_2D
    point_3D_in_plane = alpha * u + beta * v

    # Add a vector parallel to the normal vector to get the point in 3D space
    point_3D = point_3D_in_plane + lambda_value * n_hat

    return tuple(point_3D)

