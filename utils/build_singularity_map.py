from environment import SPM_Env_ver4 as SPM_Env
import numpy as np
import pickle
import matplotlib.pyplot as plt

PLOT = True
FILENAME = 'singularity_map.pickle'


def main():
    env = SPM_Env.SpmEnv()

    thetas = np.linspace(start=-np.pi/2.5, stop=np.pi/2.5, num=720)
    phis = np.linspace(start=-np.pi, stop=np.pi, num=720)

    sing_map = np.empty([len(thetas), len(phis)])

    for i, phi in enumerate(phis):
        for j, theta in enumerate(thetas):
            try:
                sing_map[i, j] = env.get_singularities([phi, theta, 0])['singA']
            except RuntimeError:
                sing_map[i, j] = 0

    sing_map_dict = {'thetas': thetas, 'phis': phis, 'singularity_map': sing_map}

    with open(FILENAME, 'wb') as handle:
        pickle.dump(sing_map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if PLOT:
    # Plot the singularity map
        plt.pcolormesh(thetas, phis, sing_map, cmap='magma')
        plt.colorbar()
        plt.xlabel(r'$\theta$ [rad]')
        plt.ylabel(r'$\phi$ [rad]')

        for angle in range(35,60,5):
            z = np.cos(np.deg2rad(angle))
            y = np.sqrt(1-z*z)
            multi_target_positions = SPM_Env.get_target_candidates(start_roll=0, start_pitch=0,
                                                                   lookv_goal=[0, y, z])
            target_thetas, target_phis = [], []
            for target_position in multi_target_positions:
                target_thetas.append(target_position[1])
                target_phis.append(target_position[0])
            target_thetas.append(multi_target_positions[0][1])
            target_phis.append(multi_target_positions[0][0])
            plt.plot(target_thetas, target_phis, linestyle='dashed', linewidth=0.8, label=rf'$\eta$= {angle}°')
        plt.legend()

        plt.show()


if __name__=='__main__':
    main()