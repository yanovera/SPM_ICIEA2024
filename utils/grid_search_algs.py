def get_neighbors(cell, shape, toroidal_neighbors = True):
    """Calculate the neighbors of a cell in a 3D grid."""
    x, y = cell
    delta = [-1, 0, 1]
    neighbors = [(x + dx, y + dy)
                 for dx in delta
                 for dy in delta
                 if dx != 0 or dy != 0]

    if toroidal_neighbors:
        # Wrap around the neighbors
        neighbors = [(nx % shape[0], ny % shape[1]) for nx, ny in neighbors]
    return neighbors

