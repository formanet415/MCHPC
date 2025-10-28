"""
Euler scheme solver for the 1D advection equation:
    ∂_t u + a0 ∂_x u = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


### Physical parameters ###
a0 = 2e5           # m/s
L = 1.5e11         # m (1 au)
sigma = 0.1 * L
nsigma = 0.1       # normalized width
numax = 1           # normalized max speed
ttravel = L / a0
ttravel_days = ttravel / 86400


class InitialProfile:
    """
    Class defining an initial profile for the advection equation.
    """
    def __init__(self, profile_type='gaussian', boundary_conditions='periodic'):
        self.profile_type = profile_type
        self.boundary_conditions = boundary_conditions

    def set_profile_parameters(self, **kwargs):
        if self.profile_type == 'gaussian':
            self.set_gaussian_profile(kwargs.get('grid', np.linspace(0, 1, 1000)),
                                      kwargs.get('x0', 0.1),
                                      kwargs.get('sigma', 0.1))
            
    def set_gaussian_profile(self, grid, x0, sigma):
        self.x0 = x0
        self.sigma = sigma
        self.grid = grid
        self.profile = self.Gaussian(grid, x0, sigma)
        if self.boundary_conditions == 'periodic':
            # Simple periodic extension by one period
            self.profile += self.Gaussian(grid + 1, x0, sigma) + self.Gaussian(grid - 1, x0, sigma)

    def Gaussian(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def get_physical_profile(self):
        """
        Converts the normalized profile to physical units.
        """
        self.x_phys = self.grid * L
        self.u_phys = self.profile * numax


    def set_profile(self, new_profile):
        self.profile = new_profile

    def plot_profile(self, filename=None):
        """
        Plots the current profile in physical units, optionally saving to file.
        """
        self.get_physical_profile()
        plt.figure()
        plt.plot(self.x_phys, self.u_phys)
        plt.xlabel('Distance (m)')
        plt.ylabel('Normalized Speed')
        plt.title(f'Initial Profile: {self.profile_type}')
        plt.grid()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_profile_norm(self, filename=None):
        """
        Plots the current normalized profile, optionally saving to file.
        """
        plt.figure()
        plt.plot(self.grid, self.profile)
        plt.xlabel('Normalized Distance x/L')
        plt.ylabel('Normalized Speed (u/umax)')
        plt.title(f'Initial Profile: {self.profile_type}')
        plt.grid()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class TransportEquation1D:
    """
    1D advection equation solver using an explicit Euler scheme.
    """
    def __init__(self, a0, L, Nx, Ccond, initial_profile):
        self.a0 = a0
        self.L = L
        self.na0 = a0 / L          # Normalized velocity (1/s)
        self.Nx = Nx
        self.Ccond = Ccond
        self.profile = initial_profile
        self.grid = initial_profile.grid

    def determine_time_step(self):
        """
        Computes a time step satisfying the CFL condition.
        """
        if self.Ccond > 1:
            print("WARNING: CFL condition violated: Ccond should be chosen at least ≤ 1 for stability.")
        dxphys = self.L / self.Nx
        dt = self.Ccond * dxphys / self.a0 # a0 also in physical units
        return dt

    def du_old(self, u, dx, dt):
        """
        Implementation using central difference (not stable for advection equation).
        """
        dudx = np.zeros(np.shape(u)[0] - 2)
        dudx = (u[2:] - u[:-2]) / (2*dx)
        return -dt * self.na0 * dudx

    def du(self, u, dx, dt):
        """
        Implementation using upwind scheme. This is stable, but why? The information used to update u[i] only comes from u[i] and u[i-1], meaning along the direction of propagation. 
        This avoids introducing spurious oscillations that can occur when using central differences, which use information from both sides of the point being updated.
        """
        C = self.na0 * dt / dx
        dudx = u[1:-1] - u[:-2]  # upwind
        return -C * dudx

    def run_simulation(self, tmax, dtplot=100):
        """
        Runs the advection simulation and saves an animation if requested.
        """
        dt = self.determine_time_step()
        plotevery = int(dtplot / dt)
        nsteps = int(tmax / dt)
        dx = 1 / self.Nx
        u = self.profile.profile.copy()


        for n in range(nsteps):
            du = self.du(u, dx, dt)
            u[1:-1] += du

            if self.profile.boundary_conditions == 'periodic':
                u[0] = u[-2]
                u[-1] = u[1]

            if n%plotevery == 0:
                self.profile.set_profile(u)
                self.profile.plot_profile(filename=f'profile_{n:05d}.png')

        # create an animation by stacking the saved plots
        plot_files = [f'profile_{n:05d}.png' for n in range(0, nsteps, plotevery)]
        fig, ax = plt.subplots()
        img = plt.imread(plot_files[0])
        im = ax.imshow(img)
        ax.axis('off')
        def update(frame):
            img = plt.imread(plot_files[frame])
            im.set_array(img)
            ax.axis('off')
            return [im]
        
        ani = FuncAnimation(fig, update, frames=len(plot_files), blit=True)
        ani.save('advection_simulation.gif', writer='imagemagick', fps=5)

        # delete temporary plot files
        for file in plot_files:
            os.remove(file)

    


if __name__ == "__main__":
    nx = 1000
    Ccond = 0.5
    boundary_conditions = 'periodic'

    initial_profile = InitialProfile(profile_type='gaussian', boundary_conditions=boundary_conditions)
    initial_profile.set_profile_parameters(grid=np.linspace(0, 1, nx), x0=0.1, sigma=nsigma)
    initial_profile.plot_profile(filename='initial_profile.png')

    simulator = TransportEquation1D(a0, L, nx, Ccond, initial_profile)
    simulator.run_simulation(tmax=ttravel, dtplot=10000)