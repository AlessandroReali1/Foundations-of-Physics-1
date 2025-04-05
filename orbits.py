# 30537 - Foundations of Physics 1

# Simulation of orbit of planets
# Application of kepler's laws and Newton's gravitational law.

import numpy as np
import matplotlib.pyplot as plt


G = 1. # arbitrary!
dt = 0.001 # change this to see the effect of the approximation error
dims = 2 # you can switch to a 3D problem, but to plot it you will have to change the code (look online). Alternatively you can just plot in 2D.
cmap = {0:'k', 1:'b', 2:'y', 3:'g', 4:'r'} # we assign a color to each planet, so that the plot is more understandable.

def norm(x):
    return np.sqrt((x**2).sum())    

def initial_condition_generic(ms, v=1.5):
    # you can play with this!
    # the current implementation initializes the masses on a unit circle, at equal distances
    # and the velocities are initialized tangential to the circle, making sure that the total momentum is zero
    # the parameter v sets the scale of the velocities (basically controlling the total kinetic energy)
    p = len(ms)
    rs = np.array([[np.cos(i * (2*np.pi/p)), np.sin(i * (2*np.pi/p))] for i in range(p)])
    # initial condition with zero momentum for the CM
    vs = v*np.array([[-np.sin(i * (2*np.pi/p))/ms[i], np.cos(i * (2*np.pi/p))/ms[i]] for i in range(p)])
    return rs, vs
    
def acceleration(r1, r2, m2):
    r_12 = r2 - r1
    d_12 = norm(r_12)
    a = G * m2 / d_12**3 * r_12  
    return a

def accelerations(rs, ms):
    p = len(ms)
    a = np.zeros((p, dims))
    for i in range(p):
        for j in range(p):
            if j == i: continue
            a[i] += acceleration(rs[i], rs[j], ms[j])
    return a

def integration_step_Euler(rs, vs, ms):   
    dvs = dt * accelerations(rs, ms)
    drs = dt * vs
    return drs, dvs

def integration_step_RK4(rs, vs, ms): 
    k1vs = accelerations(rs, ms); k1rs = vs;
    k2vs = accelerations(rs + k1rs * dt/2, ms); k2rs = vs + k1vs * dt/2;
    k3vs = accelerations(rs + k2rs * dt/2, ms); k3rs = vs + k2vs * dt/2;
    k4vs = accelerations(rs + k3rs * dt, ms); k4rs = vs + k3vs * dt;
    
    drs = dt/6 * (k1rs + 2*k2rs + 2*k3rs + k4rs)
    dvs = dt/6 * (k1vs + 2*k2vs + 2*k3vs + k4vs) 
    return drs, dvs    

def total_energy(rs, vs, ms): # this should be conserved! (is it?)
    p = len(ms)
    K = 0. # kinetic energy
    for i in range(p):
        K += 1/2 * ms[i] * norm(vs[i])**2
    U = 0. # potential energy
    for i in range(p):
        for j in range(i+1, p):
            U += - G * ms[i] * ms[j] / norm(rs[i] - rs[j])
    return K + U # total mechanical energy

def cm_momentum(vs, ms): # this should be conserved!
    return np.sum(np.array(ms).reshape(-1,1) * vs, axis=0)

def cm_position(rs, ms): # if you want you can plot this as well!
    return 1/sum(ms) * np.sum(ms * rs, axis=1)

    
def trajectory(ms, T=100000, plottime=100, seed=None):
    if seed is not None: # this is to fix the seed, so that the generated random numbers are repeatable and the simulation can be run identically
        np.random.seed(seed)
    
    # INITIALIZE POSITIONS AND VELOCITIES
    p = len(ms)
    rs, vs = initial_condition_generic(ms)
    rs_traj = np.zeros((T, p, dims)) # stores the whole trajectory
    
    # uncomment below to run two simulations in parallel (e.g. to check the divergence of the trajectories in time)
    # rs2, vs2 = np.copy(rs), np.copy(vs) # same initial condition
    # rs2[0] += 0.001 * np.random.randn(dims) # small gaussian perturbation of the position of one planet
    # rs_traj2 = np.copy(rs_traj) # to store the trajectory of the second simulation
    for t in range(T):
        # COMPUTE UPDATE
        drs, dvs = integration_step_RK4(rs, vs, ms) 
        # uncomment below if you are comparing simulations
        # drs2, dvs2 = integration_step_RK4(rs2, vs2, ms) # (EXP 1) check divergence of trajectories
        # drs2, dvs2 = integration_step_Euler(rs2, vs2, ms) # (EXP 2) check approximation error
        
        # APPLY UPDATE
        rs = rs + drs; vs = vs + dvs 
        rs_traj[t, :, :] = rs # and store away the current configuration
        # uncomment below if you are comparing simulations
        # rs2 = rs2 + drs2; vs2 = vs2 + dvs2
        # rs_traj2[t, :, :] = rs2
        
        # SHOW THE TRAJECTORY
        if t % plottime == 0:
            print(f"t={t} E={total_energy(rs, vs, ms)} p_cm={cm_momentum(vs, ms)}")
            plt.pause(0.001)
            plt.clf()
            for i in range(p):
                plt.plot(rs_traj[:t+1,i,0], rs_traj[:t+1,i,1], c=cmap[i])
                plt.plot(rs_traj[t,i,0], rs_traj[t,i,1], 'o', c=cmap[i])
                # uncomment below if you are comparing simulations
                # plt.plot(rs_traj2[:t+1,i,0], rs_traj2[:t+1,i,1], '--', c=cmap[i]) # the second simulation is displayed with a dashed trajectory
                # plt.plot(rs_traj2[t,i,0], rs_traj2[t,i,1], 'o', c=cmap[i])
        
    return rs_traj

# example for running the code:
# input a list of masses, the total number of steps (you can interrupt from the console), and how often you plot.
trajectory([3., 2., 1.], T=1000000, plottime=100, seed=12345)

# POSSIBLE EXPERIMENTS:
# 1- With two planets of equal masses, try to find the parameter 'v' in the intial condition that puts the planets on a circular orbit. You can compute analytically!
# 2- Try to initialize two simulations from very similar initial condiitons and track the evolution of the distance of the position of one planet in the two simulations. 
#    Is it diverging polinomially or exponentially? Can you fit the curve to find the Lyapunov exponent? 
# 3- compare Euler vs RK4, see that after some time, even if the initial condition is the same, the two simulations eventually diverge.
#    You can see that this happens more quickly if your 'dt' is larger.
# 4- Simulate the "reduced" problem, initializing two planets on a circular orbit, and 
#    then adding an additional very small mass (even m=0. can work), to the system.
#    If you compare the kinetic energy and the potential energy, you can even set the two larger masses on a circular orbit. 
#    When you plot, you could even rotate the frame of reference so that the two large masses appear to sit still and only the third small mass moves around.
# 4- Assuming you have 3 equal masses, can you find some periodic orbits besides the "trivial" one we are setting with our initialization?
#    You should see that a lot of initial conditions lead to a planet getting ejected from the system.
# 5- Can you find an initial condition that yields "Moore's chasing solution", where the shape of the orbit is an infinity symbol? 