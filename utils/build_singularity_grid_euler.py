from environment import SPM_Env_ver4 as SPM_Env
import numpy as np
from grid_utils import plot_2d_decomposition_with_borders, create_2d_decomposition, deactivate_cells_below_threshold, save_decomposition, deactivate_isolated_cells, load_decomposition

CELL_SIZE = np.deg2rad(1)
FILENAME = '../grids/euler_grid_1.pkl'
DEACTIVATE_ISOLATED_CELLS = False
PLOT = True


def main():
    if PLOT:
        try:
            decomposition = load_decomposition(FILENAME)
            plot_2d_decomposition_with_borders(decomposition, xlabel='Phi', ylabel='Theta')
            exit()
        except IOError:
            print('file not found, building decomposition')

    env = SPM_Env.SpmEnv()

    num_cells = int(2*np.pi/CELL_SIZE)

    thetas = np.linspace(start=-np.pi, stop=np.pi, num=num_cells, endpoint=False)
    phis = np.linspace(start=-np.pi, stop=np.pi, num=num_cells, endpoint=False)

    decomposition = create_2d_decomposition(resolution=CELL_SIZE)

    for x_index, phi in enumerate(phis):
        print(100*x_index/len(phis),"%")
        for y_index, theta in enumerate(thetas):
            with np.errstate(divide='raise'), np.errstate(invalid='raise'):
                try:
                    decomposition[x_index, y_index] = min(env.get_singularities([phi, theta, 0])['singA'],
                                                          env.get_singularities([phi + CELL_SIZE, theta, 0])['singA'],
                                                          env.get_singularities([phi + CELL_SIZE, theta + CELL_SIZE, 0])['singA'],
                                                          env.get_singularities([phi, theta + CELL_SIZE, 0])['singA'],
                                                          )
                except FloatingPointError:
                    decomposition[x_index, y_index] = 0.0

    print('100.0%')

    deactivate_cells_below_threshold(decomposition, threshold=0.05)
    if DEACTIVATE_ISOLATED_CELLS:
        deactivate_isolated_cells(decomposition)
    save_decomposition(decomposition, FILENAME)







if __name__ == '__main__':
    main()
