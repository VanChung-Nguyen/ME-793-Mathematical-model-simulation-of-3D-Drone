import numpy as np
# (plt is unused below; keep if you'll plot)
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate

import pybounds

############################################################################################
# Physical parameters
############################################################################################
m  = 1.82
g  = 9.81
lx = 0.159          # arm length (square frame)
Ixx = 0.075
Iyy = 0.075
Izz = 0.105
kt  = 0.00025       # thrust coefficient (thrust = kt * u_i) with u_i ∝ Ω_i^2
kd  = 0.000004      # yaw drag coefficient (yaw torque = ± kd * u_i)

############################################################################################
# Continuous-time dynamics (full 3D)
############################################################################################
class F(object):
    def f(self, x_vec, u_vec,
          m=m, g=g, lx=lx, Ixx=Ixx, Iyy=Iyy, Izz=Izz, kt=kt, kd=kd,
          return_state_names=False):

        if return_state_names:
            return ['x','y','z','phi','theta','psi',
                    'xdot','ydot','zdot','phidot','thetadot','psidot']

        # --- ensure scalars when indexing ---
        x, y, z, phi, theta, psi = x_vec[0:6]
        xdot, ydot, zdot = x_vec[6:9]
        phidot, thetadot, psidot = x_vec[9:12]
    
        # Inputs
        u1, u2, u3, u4 = u_vec
    
        # Shorthands
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
    
        # Body z-axis in world frame
        ez_w_x = spsi * sphi + cpsi * sth * cphi
        ez_w_y = spsi * sth * cphi - cpsi * sphi
        ez_w_z = cphi * cth
    
        # Angular rates
        p, q, r = phidot, thetadot, psidot
    
        # --- f0: drift (no control) ---
        f0_contribution = np.array([
            xdot,
            ydot,
            zdot,
            phidot,
            thetadot,
            psidot,
            0.0,
            0.0,
            -g,
            ((Iyy - Izz)/Ixx)*q*r,
            ((Izz - Ixx)/Iyy)*r*p,
            ((Ixx - Iyy)/Izz)*p*q
        ])
    
        # --- f1: contribution for u1 ---
        f1_contribution = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (kt/m) * ez_w_x * u1,
            (kt/m) * ez_w_y * u1,
            (kt/m) * ez_w_z * u1,
            (-lx * kt * u1) / Ixx,
            (-lx * kt * u1) / Iyy,
            ( kd * u1) / Izz
        ])
    
        # --- f2: contribution for u2 ---
        f2_contribution = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (kt/m) * ez_w_x * u2,
            (kt/m) * ez_w_y * u2,
            (kt/m) * ez_w_z * u2,
            ( lx * kt * u2) / Ixx,
            (-lx * kt * u2) / Iyy,
            (-kd * u2) / Izz
        ])
    
        # --- f3: contribution for u3 ---
        f3_contribution = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (kt/m) * ez_w_x * u3,
            (kt/m) * ez_w_y * u3,
            (kt/m) * ez_w_z * u3,
            (-lx * kt * u3) / Ixx,
            ( lx * kt * u3) / Iyy,
            (-kd * u3) / Izz
        ])
    
        # --- f4: contribution for u4 ---
        f4_contribution = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (kt/m) * ez_w_x * u4,
            (kt/m) * ez_w_y * u4,
            (kt/m) * ez_w_z * u4,
            ( lx * kt * u4) / Ixx,
            ( lx * kt * u4) / Iyy,
            ( kd * u4) / Izz
        ])
    
        # --- Combine dynamics ---
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution + f3_contribution + f4_contribution
        return x_dot_vec

############################################################################################
# Continuous-time measurement functions
############################################################################################
class H(object):
    """
    Measurements:
      - x, y, z
      - optical flow-like terms xdot/z, ydot/z (safe division)
      - Euler angles phi, theta, psi
      - angular rates phidot, thetadot, psidot
      - accelerations (world-frame ax, ay, az) from dynamics
    """
    def __init__(self, measurement_option='h_all'):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        if not hasattr(self, self.measurement_option):
            raise AttributeError(f"Unknown measurement option: {self.measurement_option}")
        return getattr(self, self.measurement_option)(
            x_vec, u_vec, return_measurement_names=return_measurement_names
        )

    def h_all(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return [
                'x','y','z',
                'optic_flow_x','optic_flow_y',
                'phi','theta','psi',
                'phidot','thetadot','psidot',
                'ax','ay','az'
            ]

        x, y, z, phi, theta, psi = x_vec[0:6]
        xdot, ydot, zdot         = x_vec[6:9]
        p, q, r                  = x_vec[9:12]
        u1, u2, u3, u4 = u_vec
        
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
    
        # Body z-axis in world frame
        ez_w_x = spsi * sphi + cpsi * sth * cphi
        ez_w_y = spsi * sth * cphi - cpsi * sphi
        ez_w_z = cphi * cth
        
        T = kt * (u1 + u2 + u3 + u4)
        ax = (T / m) * ez_w_x
        ay = (T / m) * ez_w_y
        az = (T / m) * ez_w_z - g
    
        z_safe = z if abs(z) > 1e-3 else 1e-3
    
        y_vec = np.array([
            x,
            y,
            z,
            xdot / z_safe,
            ydot / z_safe,
            phi,
            theta,
            psi,
            phidot,
            thetadot,
            psidot,
            ax,
            ay,
            az
        ])
        return y_vec

############################################################################################
# Drone simulation with MPC (3D set-points)
############################################################################################
def simulate_drone(f, h, tsim_length=20.0, dt=0.1, measurement_names=None,
                   trajectory_shape='squiggle', setpoint=None, rterm=1e-10):
    """
    trajectory_shape ∈ {'circle','lemniscate','alternating','squiggle','random','constant_psidot'}
    If setpoint is given, it should contain arrays over time for any of:
      'x','y','z','psi','x_dot','y_dot','z_dot','psi_dot' (others allowed, ignored by MPC if unused)
    """
    # Names
    state_names = F().f(None, None, return_state_names=True)
    input_names = ['u1','u2','u3','u4']

    if measurement_names is None:
        measurement_names = H().h(None, None, return_measurement_names=True)

    # Initialize simulator
    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=10
    )

    # Time base
    tsim = np.arange(0.0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    # Build default 3D setpoints if none provided
    if setpoint is None:
        assert trajectory_shape in ['circle','lemniscate','alternating','squiggle','random','constant_psidot']

        if trajectory_shape == 'circle':
            R = 0.1
            w = 0.3 * 2*np.pi
            x = R*np.cos(w*tsim)
            y = R*np.sin(w*tsim)
            z = 1.0 + 0.3*np.sin(0.5*w*tsim)
            psi = NA
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

        elif trajectory_shape == 'lemniscate':
            a = 2.0
            w = 0.25 * 2*np.pi
            x = a * np.sin(w*tsim)
            y = a * np.sin(w*tsim) * np.cos(w*tsim)
            z = 1.2 + 0.4*np.sin(0.5*w*tsim + 0.7)
            psi = NA
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

        elif trajectory_shape == 'alternating':
            # piecewise accel to get lateral & vertical changes
            a = 0
            b = int(len(tsim)/4.)
            c = int(len(tsim)*2/4.)
            d = int(len(tsim)*3/4.)
            e = -1

            ax = np.hstack(( 2.0*np.cos(0.3*2*np.pi*tsim)[a:b],
                              0*tsim[b:c],
                              2.0*np.cos(0.3*2*np.pi*tsim)[c:d],
                              0*tsim[d:e]))
            vx = np.cumsum(ax)*dt
            x  = 5*np.cumsum(vx)*dt

            ay = np.hstack(( 1.6*np.sin(0.27*2*np.pi*tsim)[a:b],
                              0*tsim[b:c],
                             -1.6*np.sin(0.27*2*np.pi*tsim)[c:d],
                              0*tsim[d:e]))
            vy = np.cumsum(ay)*dt
            y  = 5*np.cumsum(vy)*dt

            az = np.hstack(( 0.1*np.sin(0.2*2*np.pi*tsim)[a:b],
                              0*tsim[b:c],
                             -0.1*np.sin(0.2*2*np.pi*tsim)[c:d],
                              0*tsim[d:e]))
            vz = np.cumsum(az)*dt
            z  = 5*np.cumsum(vz)*dt + 1.0

            for arr in (x,y,z):
                if len(arr) > len(tsim):
                    arr[:] = arr[:len(tsim)]
            psi = NA
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

        elif trajectory_shape == 'squiggle':
            x = 2.0*np.cos(2*np.pi*0.30*tsim)
            y = 1.5*np.sin(2*np.pi*0.22*tsim + 0.4)
            z = 0.8 + 0.4*np.sin(2*np.pi*0.18*tsim + 0.9)
            psi = NA
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

        elif trajectory_shape == 'random':
            def generate_smooth_curve(t_points, method='spline', smoothness=0.1, amplitude=1.0, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                t_points = np.array(t_points)
                # cubic spline over random control points
                n_control = max(5, int(len(t_points) * smoothness))
                control_t = np.linspace(t_points[0], t_points[-1], n_control)
                control_y = np.random.normal(0, amplitude/3, n_control)
                spline = interpolate.CubicSpline(control_t, control_y)
                return spline(t_points)

            x = generate_smooth_curve(tsim, smoothness=0.15, amplitude=3.0, seed=42)
            y = generate_smooth_curve(tsim, smoothness=0.12, amplitude=2.6, seed=7)
            z = 1.5 + 0.6*generate_smooth_curve(tsim, smoothness=0.10, amplitude=2.0, seed=24)
            psi = NA
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

        elif trajectory_shape == 'constant_psidot':
            psidot = 0.15*np.sign(np.cos(tsim*2*np.pi*0.15))
            psi = np.cumsum(psidot)*dt
            x = np.zeros_like(tsim)
            y = np.zeros_like(tsim)
            z = np.ones_like(tsim)*1.5
            setpoint = {'x': x, 'y': y, 'z': z, 'psi': psi}

    # Feed setpoints into simulator's TVPs
    # Convention: simulator.model.tvp has fields 'x_set','y_set','z_set','psi_set' if used in your pybounds model.
    simulator.update_dict(setpoint, name='setpoint')

    # --- Objective (track position + altitude + yaw) ---
    # Use tvp keys exactly as defined above (no *_set suffixes)
    cost_x   = (simulator.model.x['x']   - simulator.model.tvp['x_set'])   ** 2
    cost_y   = (simulator.model.x['y']   - simulator.model.tvp['y_set'])   ** 2
    cost_z   = (simulator.model.x['z']   - simulator.model.tvp['z_set'])   ** 2
    cost_psi = (simulator.model.x['psi'] - simulator.model.tvp['psi_set']) ** 2
    cost = cost_x + cost_y + cost_z + cost_psi

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=rterm, u2=rterm, u3=rterm, u4=rterm)

    # === Bounds ===
    simulator.mpc.bounds['lower', '_x', 'phi']   = -np.pi/4
    simulator.mpc.bounds['upper', '_x', 'phi']   =  np.pi/4
    simulator.mpc.bounds['lower', '_x', 'theta'] = -np.pi/4
    simulator.mpc.bounds['upper', '_x', 'theta'] =  np.pi/4
    simulator.mpc.bounds['lower', '_x', 'z']     =  0.0  # no underground

    # inputs nonnegative (no reverse thrust)
    for ui in ['u1','u2','u3','u4']:
        simulator.mpc.bounds['lower', '_u', ui] = 0.0
        # Optionally cap: simulator.mpc.bounds['upper','_u',ui] = u_max

    # === Simulate (MPC) ===
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=None, u=None, mpc=True, return_full_output=True
    )
    return t_sim, x_sim, u_sim, y_sim, simulator

###############################################################################################
# (Optional) smooth curve generator if you want to use standalone elsewhere
###############################################################################################
def generate_smooth_curve(t_points, method='spline', smoothness=0.1, amplitude=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t_points = np.array(t_points)
    if method == 'spline':
        n_control = max(5, int(len(t_points) * smoothness))
        control_t = np.linspace(t_points[0], t_points[-1], n_control)
        control_y = np.random.normal(0, amplitude/3, n_control)
        spline = interpolate.CubicSpline(control_t, control_y)
        return spline(t_points)
    elif method == 'sine_sum':
        n_harmonics = max(3, int(20 * smoothness))
        result = np.zeros_like(t_points, dtype=float)
        for i in range(n_harmonics):
            freq  = np.random.exponential(1.0 / smoothness)
            phase = np.random.uniform(0, 2*np.pi)
            amp   = np.random.uniform(0, amplitude) / (i + 1)
            result += amp * np.sin(2*np.pi*freq*t_points + phase)
        return result
    elif method == 'noise_filter':
        from scipy.signal import butter, filtfilt
        noise = np.random.normal(0, amplitude, len(t_points))
        nyquist = 0.5 * len(t_points) / (t_points[-1] - t_points[0])
        cutoff = nyquist * smoothness
        b, a = butter(3, cutoff / nyquist, btype='low')
        return filtfilt(b, a, noise)
    else:
        raise ValueError("Method must be 'spline', 'sine_sum', or 'noise_filter'")
