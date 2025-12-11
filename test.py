import numpy as np
import json
import os
from datetime import datetime
from Particle import Particle
from Data import Sun
from math import pi

class Tests:
    """
    Class for testing conservation laws in the simulation.
    """
    @staticmethod
    def conservation_of_linear_momentum(particles, history):
        """
        Tests conservation of total linear momentum (P) for the system.
        
        """
        # Calculate initial linear momentum (first time step)
        P_initial = np.array([0.0, 0.0, 0.0])
        for particle in particles:
            if len(history[particle.name]['vx']) > 0:
                v = np.array([history[particle.name]['vx'][0],
                             history[particle.name]['vy'][0],
                             history[particle.name]['vz'][0]])
                P_initial += particle.mass * v
        
        # Calculate final linear momentum (last time step)
        P_final = np.array([0.0, 0.0, 0.0])
        for particle in particles:
            if len(history[particle.name]['vx']) > 0:
                idx = len(history[particle.name]['vx']) - 1
                v = np.array([history[particle.name]['vx'][idx],
                             history[particle.name]['vy'][idx],
                             history[particle.name]['vz'][idx]])
                P_final += particle.mass * v
        
        # Check conservation
        difference = np.linalg.norm(P_final - P_initial)
        P_magnitude = np.linalg.norm(P_initial)
        
   
        tolerance = 1e6  # kg*m/s - reasonable tolerance for numerical errors
        
        # Calculate error percentage relative to a reference scale
        max_particle_momentum = 0.0
        for particle in particles:
            if len(history[particle.name]['vx']) > 0:
                v = np.array([history[particle.name]['vx'][0],
                             history[particle.name]['vy'][0],
                             history[particle.name]['vz'][0]])
                p_mag = np.linalg.norm(particle.mass * v)
                max_particle_momentum = max(max_particle_momentum, p_mag)
        
        # Use max particle momentum as reference scale, or absolute difference 
        if max_particle_momentum > 1e-10:
            error_percent = (difference / max_particle_momentum * 100)
  
       
        
        if  difference <= tolerance:
            print(f"Linear momentum conserved. Difference: {difference:.6e} kg*m/s")
            return True, error_percent
        else:
            print(f"Linear momentum NOT conserved. Difference: {difference:.6e} kg*m/s")
            return False, error_percent

    
    @staticmethod
    def conservation_of_angular_momentum(particles, history):
        """
        Tests conservation of angular momentum for the system. L = r × (m*v)
        
        """    
        # Calculate initial angular momentum (first time step)
        L_initial = np.array([0.0, 0.0, 0.0])
        for particle in particles:
            if len(history[particle.name]['x']) > 0:
                r = np.array([history[particle.name]['x'][0], history[particle.name]['y'][0], history[particle.name]['z'][0]])
                v = np.array([history[particle.name]['vx'][0], history[particle.name]['vy'][0], history[particle.name]['vz'][0]])
                p = particle.mass * v
                L_initial += np.cross(r, p)
        
        # Calculate final angular momentum (last time step)
        L_final = np.array([0.0, 0.0, 0.0])
        for particle in particles:
            if len(history[particle.name]['x']) > 0:
                idx = len(history[particle.name]['x']) - 1
                r = np.array([history[particle.name]['x'][idx], history[particle.name]['y'][idx], history[particle.name]['z'][idx]])
                v = np.array([history[particle.name]['vx'][idx], history[particle.name]['vy'][idx], history[particle.name]['vz'][idx]])
                p = particle.mass * v
                L_final += np.cross(r, p)
        
        # Check conservation
        difference = np.linalg.norm(L_final - L_initial)
        L_magnitude = np.linalg.norm(L_initial)
        tolerance =  1e-4  * L_magnitude if L_magnitude > 0 else 1e-6
        
        error_percent = (difference / L_magnitude * 100) if L_magnitude > 0 else 0.0 
        
        if difference > tolerance:
            print(f"Angular momentum NOT conserved. Difference: {difference:.6e}, Error: {error_percent:.6f}%")
            return False, error_percent
        else:
            print(f"Angular momentum conserved. Difference: {difference:.6e}, Error: {error_percent:.6f}%")
            return True, error_percent
    
    @staticmethod
    def conservation_of_energy(particles, history):
        """
        Tests conservation of total energy for the system.
        
        Returns:
            tuple (bool, float): True if energy is conserved, and the percentage error
        """
        if not history or len(history[list(history.keys())[0]]['x']) < 2:
            print("Not enough history data for energy test")
            return False, 0.0
        
        G = Particle.G
        num_steps = len(history[list(history.keys())[0]]['x'])
        energy_history = np.zeros(num_steps)
        
        for t in range(num_steps):
            E_total = 0.0

            # --- Kinetic Energy ---
            for p in particles:
                v_vec = np.array([
                    history[p.name]['vx'][t],
                    history[p.name]['vy'][t],
                    history[p.name]['vz'][t]
                ])
                E_total += 0.5 * p.mass * np.linalg.norm(v_vec)**2

            # --- Potential Energy ---
            for i, p1 in enumerate(particles):
                r1 = np.array([
                    history[p1.name]['x'][t],
                    history[p1.name]['y'][t],
                    history[p1.name]['z'][t]
                ])
                for j, p2 in enumerate(particles):
                    if i < j:  # avoid double counting
                        r2 = np.array([
                            history[p2.name]['x'][t],
                            history[p2.name]['y'][t],
                            history[p2.name]['z'][t]
                        ])
                        r = np.linalg.norm(r2 - r1)
                        if r > 1e-10:
                            E_total -= G * p1.mass * p2.mass / r

            energy_history[t] = E_total

      
        E_initial = energy_history[0]
        E_final = energy_history[-1]

        diff = abs(E_final - E_initial)
        magnitude = abs(E_initial)
        tolerance = 1e-3 * magnitude if magnitude > 0 else 1e-6

        error_percent = (diff / magnitude * 100) if magnitude > 0 else 0.0
        is_conserved = diff <= tolerance

        if is_conserved:
            print(f"Energy conserved. Difference: {diff:.6e}, Error: {error_percent:.6f}%")
        else:
            print(f"Energy NOT conserved. Difference: {diff:.6e}, Error: {error_percent:.6f}%")
            print(f"Initial E: {E_initial:.6e}, Final E: {E_final:.6e}")

        return is_conserved, error_percent, energy_history
    
    # For plotting Jupiter's energy over time
    @staticmethod
    def calculate_planet_energy_history(particle, particles, history):
        """
        Calculates a planet's kinetic energy, potential energy, and total energy over time.
        General method that works for any particle.
        
        Args:
            particle: Particle object for the planet
            particles: List of all Particle objects
            history: Dictionary containing position and velocity history
            
        Returns:
            tuple: (kinetic_energy, potential_energy, total_energy) arrays
                   All arrays are in the same units as the simulation
        """
        if not history or len(history[list(history.keys())[0]]['x']) < 2:
            return None, None, None
        
        G = Particle.G
        
        if particle.name not in history or len(history[particle.name]['x']) == 0:
            return None, None, None
        
        num_steps = len(history[particle.name]['x'])
        
        # Initialize energy arrays
        kinetic_energy = np.zeros(num_steps)
        potential_energy = np.zeros(num_steps)
        total_energy = np.zeros(num_steps)
        
        # Calculate energies at each time step
        for t in range(num_steps):
            # Kinetic energy: KE = 0.5 * m * v^2
            v_vec = np.array([
                history[particle.name]['vx'][t],
                history[particle.name]['vy'][t],
                history[particle.name]['vz'][t]
            ])
            kinetic_energy[t] = 0.5 * particle.mass * np.linalg.norm(v_vec)**2
            
            # Potential energy: PE = -G * m1 * m2 / r (sum over all other particles)
            r_particle = np.array([
                history[particle.name]['x'][t],
                history[particle.name]['y'][t],
                history[particle.name]['z'][t]
            ])
            
            pot_energy = 0.0
            for other_particle in particles:
                if other_particle.name != particle.name:
                    if other_particle.name in history and len(history[other_particle.name]['x']) > t:
                        r_other = np.array([
                            history[other_particle.name]['x'][t],
                            history[other_particle.name]['y'][t],
                            history[other_particle.name]['z'][t]
                        ])
                        r_distance = np.linalg.norm(r_other - r_particle)
                        if r_distance > 1e-10:
                            pot_energy -= G * particle.mass * other_particle.mass / r_distance
            
            potential_energy[t] = pot_energy
            total_energy[t] = kinetic_energy[t] + potential_energy[t]
        
        return kinetic_energy, potential_energy, total_energy
    
    @staticmethod
    def calculate_jupiter_energy_history(particles, history):
        """
        Calculates Jupiter's kinetic energy, potential energy, and total energy over time.
         
        """
        # Find Jupiter particle
        jupiter_particle = next((p for p in particles if p.name == "Jupiter"), None)
        if jupiter_particle is None:
            print("Error: Jupiter not found in particles list")
            return None, None, None
        
        return Tests.calculate_planet_energy_history(jupiter_particle, particles, history)
    
    @staticmethod
    def escape_velocity(particles, history):
        """
        Calculates the escape velocity from each planet's surface and compares with known values.
      
        """
        G = Particle.G
        
        # Known planetary radii in meters
        planetary_radii = {
            "Mercury": 2.4397e6,      # meters
            "Venus": 6.0518e6,
            "Earth": 6.371e6,
            "Mars": 3.3895e6,
            "Jupiter": 6.9911e7,
            "Saturn": 5.8232e7,
            "Uranus": 2.5362e7,
            "Neptune": 2.4622e7
        }
        
        # Known escape velocities from planet surfaces in m/s
        known_escape_velocities = {
            "Mercury": 4250.0,        
            "Venus": 10360.0,         
            "Earth": 11186.0,        
            "Mars": 5030.0,            
            "Jupiter": 59500.0,     
            "Saturn": 35500.0,       
            "Uranus": 21300.0,        
            "Neptune": 23500.0        
        }
        
        escape_velocities = {}
        
        # Print header
        print("\nEscape Velocities (from planet surface):")
        print(f"{'Planet':<12} {'Calculated':<15} {'Known':<15} {'Error %':<12} {'Status':<10}")
        print("-" * 70)
        
        for particle in particles:
            if particle.name.lower() == "sun":
                continue
            
            planet_name = particle.name
            if planet_name in planetary_radii:
                radius = planetary_radii[planet_name]
                mass = particle.mass
                
                # Calculate escape velocity from planet surface: v_esc = sqrt(2 * G * M / R)
                v_esc_calculated = np.sqrt(2 * G * mass / radius)
                
                # Get known value
                v_esc_known = known_escape_velocities.get(planet_name, None)
                
                if v_esc_known is not None:
                    error_percent = abs(v_esc_calculated - v_esc_known) / v_esc_known * 100
                    status = "✓ Good" if error_percent < 5.0 else "✗ High Error"
                    
                    print(f"{planet_name:<12} {v_esc_calculated/1000:>6.2f} km/s    {v_esc_known/1000:>6.2f} km/s    {error_percent:>6.2f}%      {status:<10}")
                    
                    escape_velocities[planet_name] = {
                        "calculated_m_per_s": float(v_esc_calculated),
                        "calculated_km_per_s": float(v_esc_calculated / 1000),
                        "known_m_per_s": float(v_esc_known),
                        "known_km_per_s": float(v_esc_known / 1000),
                        "error_percent": float(error_percent)
                    }
                else:
                    print(f"{planet_name:<12} {v_esc_calculated/1000:>6.2f} km/s    {'N/A':<15} {'N/A':<12} {'N/A':<10}")
                    escape_velocities[planet_name] = {
                        "calculated_m_per_s": float(v_esc_calculated),
                        "calculated_km_per_s": float(v_esc_calculated / 1000),
                        "known_m_per_s": None,
                        "known_km_per_s": None,
                        "error_percent": None
                    }
        
        return escape_velocities
    

    @staticmethod
    def period_error(angles, delta_T, particles):
        """
        Calculates orbital period and percent error for every planet 
        that has angle history stored in `angles`.

        """

        from math import pi

        # Known orbital periods in seconds
        known_periods = {
            "Mercury": 7603200,
            "Venus":   19414080,
            "Earth":   31557600,
            "Mars":    59356800,
            "Jupiter": 374335776,
            "Saturn":  929596608,
            "Uranus":  2661041808,
            "Neptune": 5200418560,
        }

        results = {}

        for particle in particles:
            name = particle.name

        
            if name not in angles:
                continue

            theta = angles[name]

            accumulated_angle = 0.0
            orbit_completion_step = None

            #  Detect orbit 
            for i in range(1, len(theta)):
                angle_change = theta[i] - theta[i - 1]

                # Handle wrap-around
                if angle_change > pi:
                    angle_change -= 2 * pi
                elif angle_change < -pi:
                    angle_change += 2 * pi

                accumulated_angle += angle_change

                # Check if 2π reached
                if accumulated_angle >= 2 * pi:

                    # prevent false early detection
                    if i < len(theta) * 0.05:
                        continue

                    remaining_angle = accumulated_angle - 2 * pi
                    prev_angle_change = angle_change if angle_change != 0 else 1e-12

                    fractional_step = remaining_angle / prev_angle_change
                    orbit_completion_step = i - fractional_step
                    break

            # Could not detect a full orbit → skip
            if orbit_completion_step is None:
                results[name] = (None, None)
                continue

            # --- Calculate period and error ---
            calculated_period = orbit_completion_step * delta_T
            target_period_s = known_periods.get(name)

            if target_period_s:
                error = abs(calculated_period - target_period_s) / target_period_s * 100
            else:
                error = None

            results[name] = (calculated_period, error)

            print(f"{name}: Detected Period = {calculated_period:.3e} s "
                f"({calculated_period/86400:.2f} days), Error = {error:.2f}%")

        return results
    
    
   
    @staticmethod
    def save_test_results(particles, history, delta_T, duration, method, filename=None):
        """
        Saves test results to a JSON file.
        
        """
        # Get desktop path
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_results_{method}_{timestamp}.json"
        
        # Create full path to desktop
        full_path = os.path.join(desktop_path, filename)
        
        # Run all tests
        linear_mom_result, linear_mom_error = Tests.conservation_of_linear_momentum(particles, history)
        ang_mom_result, ang_mom_error = Tests.conservation_of_angular_momentum(particles, history)
        energy_result, energy_error, energy_history = Tests.conservation_of_energy(particles, history)
        
        # Calculate angles for all planets relative to the Sun
        sun_particle = next((p for p in particles if p.name.lower() == "sun"), None)
        angles_dict = {}
        
        if sun_particle and sun_particle.name in history:
            for particle in particles:
                # Skip the Sun itself
                if particle.name.lower() == "sun":
                    continue
                
                # Only calculate angles if we have history for this particle
                if particle.name in history and len(history[particle.name]['x']) > 0:
                    angles = []
                    for i in range(len(history[particle.name]['x'])):
                        # Calculate position vector relative to Sun
                        r = np.array([
                            history[particle.name]['x'][i] - history[sun_particle.name]['x'][i],
                            history[particle.name]['y'][i] - history[sun_particle.name]['y'][i],
                            history[particle.name]['z'][i] - history[sun_particle.name]['z'][i]
                        ])
                        # Calculate angular position (arctan2 gives angle in range [-pi, pi])
                        angles.append(np.arctan2(r[1], r[0]))
                    angles_dict[particle.name] = angles
        
        # Calculate periods for all planets
        period_results = {}
        if angles_dict:
            period_results = Tests.period_error(angles_dict, delta_T, particles)
        
        # Calculate escape velocities
        escape_velocities = Tests.escape_velocity(particles, history)
        
        # Prepare period_errors dictionary for JSON
        period_errors_dict = {}
        for planet_name, (calculated_period, error_percent) in period_results.items():
            period_errors_dict[planet_name] = {
                "calculated_period_s": float(calculated_period) if calculated_period is not None else None,
                "calculated_period_days": float(calculated_period / 86400) if calculated_period is not None else None,
                "error_percent": float(error_percent) if error_percent is not None else None
            }
        # Prepare data dictionary
        results = {
            "simulation_parameters": {
                "time_step": float(delta_T),
                "duration": float(duration),
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "num_particles": len(particles),
                "particle_names": [p.name for p in particles]
            },
            "conservation_tests": {
                "linear_momentum": {
                    "conserved": bool(linear_mom_result),
                    "error_percent": float(linear_mom_error)
                },
                "angular_momentum": {
                    "conserved": bool(ang_mom_result),
                    "error_percent": float(ang_mom_error)
                },
                "energy": {
                    "conserved": bool(energy_result),
                    "error_percent": float(energy_error)
                }
            },
            "period_analysis": {
                "period_errors": period_errors_dict
            },
            "escape_velocity": {
                "velocities_m_per_s": escape_velocities
            }
        }
        
        # Save to JSON file on desktop
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to: {full_path}")
        return full_path

