from utils.grid_utils import load_decomposition, point_to_cell_coordinates, project_point_to_2D
from scipy.ndimage import label
from typing import Optional
import numpy as np
from environment.env_utils import SPMIK, Q321


class GridMap:
    def __init__(self, filename, joints_mode: bool = False, vast: Optional[np.array] = None, geopara: Optional[np.array] = None):
        if joints_mode:
            if vast is None or geopara is None:
                raise RuntimeError("vast and geopara must be specified for joints mode")
        self.labeled_grid, self.grid_resolution = self.load_grid_from_file(filename)
        self.joints_mode = joints_mode
        self.vast = vast
        self.geopara = geopara
        self.homing_label = self.get_label(roll=0.0, pitch=0.0)

    def get_label(self, roll: float, pitch: float) -> Optional[int]:
        if self.labeled_grid is not None:
            if self.joints_mode:
                tetavIK, _, SingFlag = SPMIK(Q321(roll, pitch, 0.0), self.vast, self.geopara)
                if SingFlag == 0:
                    return 0
                tetavIK_2d_projection = project_point_to_2D(tetavIK)
                cell_coord = point_to_cell_coordinates([tetavIK_2d_projection[0], tetavIK_2d_projection[1]],
                                                       resolution=self.grid_resolution)
            else:
                cell_coord = point_to_cell_coordinates([roll, pitch], resolution=self.grid_resolution)
            try:
                cell_label = self.labeled_grid[cell_coord[0], cell_coord[1]]
            except IndexError:
                cell_label = 0
            # if cell_label == 0:  # singularity
            #     cell_coord = closest_activated_cell(grid=self.labeled_grid, point=(roll, pitch), resolution=self.grid_resolution)
            #     cell_label = self.labeled_grid[cell_coord[0], cell_coord[1]]
            return cell_label
        else:
            return None

    @staticmethod
    def load_grid_from_file(filename: str):
        try:
            grid = load_decomposition(filename)
        except (FileNotFoundError, IsADirectoryError):
            try:
                grid = load_decomposition('../' + filename)
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                grid = None
        if grid is not None:
            print('grid loaded from ' + filename)
            labeled_grid, _ = label(grid)
            grid_resolution = 2 * np.pi / len(grid)
        else:
            raise RuntimeError('grid file not found')
        return labeled_grid, grid_resolution
