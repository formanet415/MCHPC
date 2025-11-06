"""
Euler scheme solver for the 1D advection equation:
    ∂_t u + a0 ∂_x u = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

### Physical parameters ###
a0 = 2e5 # m/s
L = 1.5e11 # m (1 au)
sigma = 0.1 * L
nsigma = 0.1 # normalized width
numax = 1 # normalized max speed
ttravel = L / a0
ttravel_days = ttravel / 86400

class InitialProfile:
    """
    Class defining an initial profile for the advection equation.
    """
    def __init__(self, profile_type='gaussian', boundary_conditions='periodic'):
        self.profile_type = profile_type
        self.boundary_conditions = boundary_conditions
        self.usedscheme = 'unknown'

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
        elif self.boundary_conditions == 'dirichlet':
            # Dirichlet BCs: profile goes to zero at boundaries
            self.profile[0] = 0
            self.profile[-1] = 0
        elif self.boundary_conditions == 'neumann':
            # Neumann BCs: zero gradient at boundaries
            self.profile[0] = self.profile[1]
            self.profile[-1] = self.profile[-2]

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
        plt.title(f'Initial Profile: {self.profile_type}, BCs: {self.boundary_conditions}, Scheme: {self.usedscheme}')
        plt.grid()
        plt.xlim(0, L)
        plt.ylim(0, numax*1.1)
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
    def __init__(self, a0, L, Nx, Ccond, initial_profile, scheme = "LF", limiter = None):
        self.a0 = a0
        self.L = L
        self.na0 = a0 / L # Normalized velocity (1/s)
        self.Nx = Nx
        self.Ccond = Ccond
        self.profile = initial_profile
        self.grid = initial_profile.grid
        if limiter:
            duindex = 1 # use slope-limited version
        else:
            duindex = 0 # use basic version
        if scheme == "cent":
            self.du = [self.du_cent, None][duindex]
        elif scheme == "upwind":
            self.du = [self.du_upwind, self.du_upwind_limited][duindex]
        elif scheme == "LF":
            self.du = [self.du_LaxFriedrich, self.du_LaxFriedrich_limited][duindex]
        elif scheme == "LW":
            self.du = [self.du_LaxWendroff, None][duindex]
        self.profile.usedscheme = scheme
        self.limiter = limiter

    def determine_time_step(self):
        """
        Computes a time step satisfying the CFL condition.
        """
        if self.Ccond > 1:
            print("WARNING: CFL condition violated: Ccond should be chosen at least ≤ 1 for stability.")
        dxphys = self.L / self.Nx
        dt = self.Ccond * dxphys / self.a0 # a0 also in physical units
        return dt
    
    def du_cent(self, u, dx, dt):
        """
        Implementation using central difference (not stable for advection equation).
        """
        dudx = (u[2:] - u[:-2]) / (2*dx)
        return -dt * self.na0 * dudx
    
    def du_upwind(self, u, dx, dt):
        """
        Implementation using upwind scheme. This is stable, but why? The information used to update u[i] only comes from u[i] and u[i-1], meaning along the direction of propagation.
        This avoids introducing spurious oscillations that can occur when using central differences, which use information from both sides of the point being updated.
        """
        C = self.na0 * dt / dx
        dudx = u[1:-1] - u[:-2] # upwind
        return -C * dudx
   
    def du_LaxFriedrich(self, u, dx, dt):
        """
        Implementation using Lax-Friedrichs scheme.
        """
        C = self.na0 * dt / dx
        dudx = (u[2:] - u[:-2]) / 2
        return -C * dudx + 0.5 * (u[2:] - 2*u[1:-1] + u[:-2])
   
    def du_LaxWendroff(self, u, dx, dt):
        """
        Implementation using Lax-Wendroff scheme.
        """
        C = self.na0 * dt / dx
        # predictor: compute u at half time step
        u_half = 0.5 * (u[1:] + u[:-1]) - 0.5 * C * (u[1:] - u[:-1])
        # corrector: compute du using u_half
        return -C*(u_half[1:] - u_half[:-1]) # the size matches u[1:-1]
    
    def compute_slope(self, u, limiter='minmod'):
        """
        Computes the slope using a slope limiter for higher-order schemes.
        """
        if self.profile.boundary_conditions == 'periodic':
            u_left = np.roll(u, 1)
            u_right = np.roll(u, -1)
            duminus = u - u_left
            duplus = u_right - u
        else:
            duminus = u[1:-1] - u[:-2]
            duplus = u[2:] - u[1:-1]

        if limiter == 'minmod':
            slope_inner = self.minmod(duminus, duplus)
        elif limiter == 'MC':
            slope_inner = self.MC(duminus, duplus)
        elif limiter == 'VanLeer':
            slope_inner = self.VanLeer(duminus, duplus)
        elif limiter == 'Koren':
            slope_inner = self.Koren(duminus, duplus)
        else:
            raise ValueError(f"Unknown limiter: {limiter}")

        if self.profile.boundary_conditions == 'periodic':
            return slope_inner
        else:
            slope = np.zeros_like(u)
            slope[1:-1] = slope_inner
            return slope
    
    def minmod(self, a, b):
        """
        Minmod slope limiter.
        """
        return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)
   
    def MC(self, a, b):
        """
        Monotonized Central slope limiter.
        """
        return np.where(a*b > 0, np.sign(a) * np.minimum( np.minimum(2* np.abs(a), 2* np.abs(b)), 0.5* np.abs(a + b) ), 0.0)
   
    def VanLeer(self, a, b):
        """
        Van Leer slope limiter.
        """
        return np.where(a*b > 0, 2*a*b / (a + b + 1e-12), 0.0)
    def Koren(self, a, b):
        """
        Koren slope limiter.
        """
        r = a / (b + 1e-12)
        phi = np.minimum(2 * np.minimum(r, 1), (1 + 2 * r) / 3)
        return phi * b
   
    def du_upwind_limited(self, u, uL, uR, dx, dt):
        """
        Upwind scheme with slope limiter.
        """
        flux = np.zeros_like(u)
        if self.na0 > 0:
            flux[1:] = self.na0 * uR[:-1]
        else:
            flux[1:] = self.na0 * uL[1:]
        return -dt/dx * (flux[2:] - flux[1:-1])
    
    def du_LaxFriedrich_limited(self, u, uL, uR, dx, dt):
        """
        Lax-Friedrichs scheme with slope limiter.
        """
        C = self.na0 * dt / dx
        flux = np.zeros_like(u)
        flux[1:] = 0.5 * self.na0 * (uR[:-1] + uL[1:]) - 0.5 * (self.na0 / C) * (uL[1:] - uR[:-1])
        return -dt/dx * (flux[2:] - flux[1:-1])

    def advection_step(self, u, dx, dt, limiter=None):
        if not limiter:
            return self.du(u, dx, dt)
        # if limiter is defined, we calculate the slope.
       
        slope = self.compute_slope(u, limiter=limiter)
        uL = u - 0.5 * slope
        uR = u + 0.5 * slope
        return self.du(u, uL, uR, dx, dt)
    
    def run_simulation(self, tmax, dtplot=100, aniname = 'animation.gif'):
        """
        Runs the advection simulation and saves an animation if requested.
        """
        dt = self.determine_time_step()
        plotevery = int(dtplot / dt)
        nsteps = int(tmax / dt)
        dx = 1 / self.Nx
        u = self.profile.profile.copy()
        for n in range(nsteps):
            du = self.advection_step(u, dx, dt, self.limiter)
            u[1:-1] += du
            if self.profile.boundary_conditions == 'periodic':
                u[0] = u[-2]
                u[-1] = u[1]
            elif self.profile.boundary_conditions == 'dirichlet':
                u[0] = 0
                u[-1] = 0
            elif self.profile.boundary_conditions == 'neumann':
                u[0] = u[1]
                u[-1] = u[-2]
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
        ani.save(aniname, writer='imagemagick', fps=20)
        # delete temporary plot files
        for file in plot_files:
            os.remove(file)
   
if __name__ == "__main__":
    nx = 1000
    Ccond = 0.5
    bcs = ['periodic', 'dirichlet', 'neumann']
    schemes = ['cent', 'upwind', 'LF', 'LW']
    limiters = [None, 'minmod', 'MC', 'VanLeer'] 
    for bc in bcs:
        for scheme in schemes:
                for limiter in limiters:
                    # Use a copy of the initial profile for each scheme
                    sim_profile = InitialProfile(profile_type='gaussian', boundary_conditions=bc)
                    sim_profile.set_profile_parameters(grid=np.linspace(0, 1, nx), x0=0.1, sigma=nsigma)
                    sim_profile.usedscheme = scheme # Set scheme for title
                    tmax = ttravel if bc == 'periodic' else ttravel * 1.5
                    simulator = TransportEquation1D(a0, L, nx, Ccond, sim_profile, scheme=scheme, limiter=limiter)
                    #if not limiter:
                    # continue
                    #if limiter=='minmod':
                    # continue
                   
                    if simulator.du:
                        print(f"Running simulation: BC={bc}, Scheme={scheme}, Limiter={limiter}")
                        simulator.run_simulation(tmax=tmax, dtplot=6000, aniname=f'{bc}_{scheme}_{limiter if limiter else "no_limiter"}_advection_animation.gif')
                    else:
                        print(f"Skipping combination: BC={bc}, Scheme={scheme}, Limiter={limiter} (not implemented)")