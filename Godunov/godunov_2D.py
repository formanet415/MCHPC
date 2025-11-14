import numpy as np
import matplotlib.pyplot as plt

# ... (All helper functions from prim_to_cons to get_W_and_flux remain the same) ...
# W is a vector of primitive variables: [rho, u, v, P]
# u and v are velocity components in x and y directions
def prim_to_cons(W, gamma):
    """
    Convert primitive variables to conserved variables.
    """
    rho, u, v, P = W
    E = P / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2) 
    return np.array([rho, rho * u, rho * v, E]) 

# U is a vector of conserved variables: [rho, rhou, rhov, E]
# rhou and rhov are momentum components in x and y directions
def cons_to_prim(U, gamma):
    """
    Convert conserved variables to primitive variables.
    """
    rho, rhou, rhov, E = U
    rho = np.maximum(rho, 1e-12) 
    
    u = rhou / rho 
    v = rhov / rho 
    
    P = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2)) 
    P = np.maximum(P, 1e-12) 
    
    return np.array([rho, u, v, P]) 

def sound_speed(P, rho, gamma):
    """
    Compute sound speed.
    """
    return np.sqrt(gamma * P / rho)

def minmod(a, b):
    """
    Minmod slope limiter.
    """
    return np.where(a*b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a,b))

def reconstruct(W, idim):
    """
    what is meant by reconstruct here is to get the left and right states at each cell interface
    """
    axis = idim + 1  
    dL = W - np.roll(W, 1, axis=axis)
    dR = np.roll(W, -1, axis=axis) - W
    slope = minmod(dL, dR)
    WL = W + 0.5 * slope
    WR = np.roll(W - 0.5 * slope, -1, axis=axis)
    return WL, WR

def hll_flux(U_L, U_R, idir, gamma=1.4):
    """
    calculates the hll flux along idir
    """
    WL, FL = get_W_and_flux(U_L, gamma, idir)
    WR, FR = get_W_and_flux(U_R, gamma, idir)

    # to decide which flux to use we need the wave speeds
    rhoL, uL, vL, PL = WL
    rhoR, uR, vR, PR = WR

    cL = sound_speed(PL, rhoL, gamma)
    cR = sound_speed(PR, rhoR, gamma)
    
    vL = [uL, vL][idir]
    vR = [uR, vR][idir]

    SL = np.minimum(vL - cL, vR - cR)
    SR = np.maximum(vL + cL, vR + cR)
    
    SL_b = SL[np.newaxis, :, :]
    SR_b = SR[np.newaxis, :, :]
    denom = SR_b - SL_b + 1e-12
    F_HLL = (SR_b * FL - SL_b * FR + SL_b * SR_b * (U_R - U_L)) / denom

    F = np.where(SL_b >= 0, FL, np.where(SR_b <= 0, FR, F_HLL))
    return F
    
def get_W_and_flux(U, gamma, idir):
    """
    Helper - returns the flux in idir direction and also the primitive variables since they are used in the calcualtio nand we need them anyway
    """
    W = cons_to_prim(U, gamma)
    rho, u, v, P = W
    E = U[3] 
    
    F = np.zeros_like(U) # fluxes
    if idir == 0: # x-direction flux
        F[0] = rho * u
        F[1] = rho * u**2 + P
        F[2] = rho * u * v
        F[3] = u * (E + P)
    elif idir == 1: # y-direction flux
        F[0] = rho * v
        F[1] = rho * u * v
        F[2] = rho * v**2 + P
        F[3] = v * (E + P)
        
    return W, F

class Godunov2D:
    """
    Simulation class for 2D Godunov code.
    """
    def __init__(self, nx, ny, xmin, xmax, ymin, ymax, gamma=1.4):
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.gamma = gamma
        
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny

        # the entire boundary will be ghost cells
        
        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
       
        # Conserved variables array U: shape (nvar, nx, ny)
        self.U = np.zeros((4, nx, ny))
        self.W = np.zeros((4, nx, ny))
        self.t = 0.

        self._initial_conditions()
        self.fignames = []

    def _initial_conditions(self):
        """
        Set initial conditions for the simulation.
        """
        W_ambient = np.array([0.01, 0.0, 0.0, 0.01])
        # where the components are [rho, u, v, P]
        U_ambient = prim_to_cons(W_ambient, self.gamma)
        self.W[...] = W_ambient[:, np.newaxis, np.newaxis]
        self.U[...] = U_ambient[:, np.newaxis, np.newaxis]

        # left cloud 
        W_L = np.array([1.0, 1.0, 0.0, 1.0])
        U_L = prim_to_cons(W_L, self.gamma)
        centerL = (2.5, 2.5)
        radiusL = 1.

        # right cloud
        W_R = np.array([0.125, -1.0, 0.0, 0.1])
        U_R = prim_to_cons(W_R, self.gamma)
        centerR = (7.5, 2.5)
        radiusR = 1.

        mask_L = (self.xx - centerL[0])**2 + (self.yy - centerL[1])**2 < radiusL**2
        mask_R = (self.xx - centerR[0])**2 + (self.yy - centerR[1])**2 < radiusR**2

        self.U[:, mask_L] = U_L[:, np.newaxis]
        self.U[:, mask_R] = U_R[:, np.newaxis]
        
        self.W[:, mask_L] = W_L[:, np.newaxis]
        self.W[:, mask_R] = W_R[:, np.newaxis]

    def plot_density(self):
        """
        Density plotter
        """
        rho = self.U[0, :, :]

        plt.figure(figsize=(12, 6))
        # Use a reasonable vmin/vmax to handle potential outliers
        plt.imshow(rho.T, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                   origin='lower', cmap='jet', vmin=0.0, vmax=1.2)
        plt.colorbar(label='Density')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Density at t={self.t:.2f}')
        figname = f'density_t{self.t:.2f}.png'
        plt.savefig(figname, dpi=150, bbox_inches='tight')
        plt.close()
        self.fignames.append(figname)

    def compute_dt(self, CFL=0.8):
        """
        Compute time step based on CFL condition.
        """
        rho, u, v, P = self.W
        c = sound_speed(P, rho, self.gamma)
        
        # Calculate max wave speeds in each direction
        max_speed_x = np.max(np.abs(u) + c)
        max_speed_y = np.max(np.abs(v) + c)
        
        # Calculate dt based on the most restrictive condition
        dt_x = self.dx / max_speed_x
        dt_y = self.dy / max_speed_y
        
        dt = CFL * min(dt_x, dt_y)
        return dt
    
    def apply_bc(self, U):
        """
        Applies transmissive/outflow boundary conditions (zero-order extrapolation).
        """
        # Note: This modifies U in-place
        U[:, 0, :] = U[:, 1, :]
        U[:, -1, :] = U[:, -2, :]
        U[:, :, 0] = U[:, :, 1]
        U[:, :, -1] = U[:, :, -2]
        return U
   
    def get_rhs(self, U_in):
        """
        Calculates the Right-Hand-Side (RHS) of the finite-volume update:
        RHS = - ( (F_i+1/2 - F_i-1/2)/dx + (G_j+1/2 - G_j-1/2)/dy )
        """
        # Create a copy to apply BCs without modifying the original
        U = U_in.copy()
        U = self.apply_bc(U)
        
        # Convert to primitive variables (W now includes ghost cells)
        W = cons_to_prim(U, self.gamma)

        # --- x-direction fluxes ---
        WLx, WRx = reconstruct(W, idim=0)
        ULx = prim_to_cons(WLx, self.gamma)
        URx = prim_to_cons(WRx, self.gamma)
        Fx = hll_flux(ULx, URx, idir=0, gamma=self.gamma)
        div_Fx = Fx - np.roll(Fx, 1, axis=1)  # This is (F_i+1/2 - F_i-1/2)

        # --- y-direction fluxes ---
        WLy, WRy = reconstruct(W, idim=1)
        ULy = prim_to_cons(WLy, self.gamma)
        URy = prim_to_cons(WRy, self.gamma)
        Fy = hll_flux(ULy, URy, idir=1, gamma=self.gamma)
        div_Fy = Fy - np.roll(Fy, 1, axis=2)  # This is (G_j+1/2 - G_j-1/2)
        
        # --- Total RHS ---
        rhs = - (div_Fx / self.dx + div_Fy / self.dy)
        
        return rhs

    def advance(self, dt):
        """
        Advances the solution by dt using a 2-stage Runge-Kutta (MOL)
        This is equivalent to the MUSCL-Hancock predictor-corrector.
        
        U_star = U^n + 0.5 * dt * RHS(U^n)
        U^{n+1} = U^n + dt * RHS(U_star)
        """
        
        # --- Predictor step ---
        # Get RHS based on U^n
        rhs_n = self.get_rhs(self.U)
        
        # Predict U_star at t = n + dt/2
        # U_star is a cell-centered, half-time-step-evolved state
        U_star = self.U + 0.5 * dt * rhs_n
        
        # --- Corrector step ---
        # Get RHS based on the predicted state U_star
        # This gives us fluxes at t = n + dt/2
        rhs_star = self.get_rhs(U_star)
        
        # Correct to U^{n+1} using the half-step fluxes
        U_new = self.U + dt * rhs_star
        
        return U_new

    def step(self, dt):
        self.t += dt
        
        # Advance the conserved variables
        self.U = self.advance(dt)
        
        # Apply BCs to the new state
        self.U = self.apply_bc(self.U)
        
        # Update primitive variables (W) from the new conserved variables (U)
        self.W = cons_to_prim(self.U, self.gamma)
        

if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 200, 100
    xmin, xmax = 0.0, 10.0
    ymin, ymax = 0.0, 5.0
    gamma = 1.4
    final_time = 3.
    plot_interval = 0.05 # Increased interval for fewer plots

    sim = Godunov2D(nx, ny, xmin, xmax, ymin, ymax, gamma)
    sim.plot_density()

    # Use a counter for plotting to avoid floating point issues
    plot_counter = 1
    
    while sim.t < final_time:
        dt = sim.compute_dt(CFL=0.5) # A CFL of 0.5 is safer
        
        # Ensure we don't step over the final time
        if sim.t + dt > final_time:
            dt = final_time - sim.t
            
        sim.step(dt)
        
        print(f"Time: {sim.t:.2f} / {final_time:.2f}, dt: {dt:.4e}")

        # Check if it's time to plot
        if sim.t >= plot_counter * plot_interval:
            sim.plot_density()
            plot_counter += 1
            
    # Always plot the final state
    sim.plot_density()
    fignames = sim.fignames
    
    print("Simulation complete.")

    print("Generating movie...")
    import imageio
    import imageio_ffmpeg  

    with imageio.get_writer(
        'Godunov/godunov_2D_simulation.mp4', 
        fps=10, 
        codec='libx264'
    ) as writer:
        for filename in fignames:
            writer.append_data(imageio.imread(filename))
    print("Movie saved as godunov_2D_simulation.mp4")
    import os
    for filename in fignames[:-2]:
        os.remove(filename)
    print("Temporary figure files removed.")

    print("Code execution finished.")