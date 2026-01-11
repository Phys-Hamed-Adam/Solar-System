from matplotlib import pyplot as plt
import numpy as np
from tests import Tests

# Time step values (in seconds)
time_step = [28800, 86400, 172800]


rk4_Angular_Momentum_Error = [0.00000003, 0.00000010,0.00000018]
verlet_Angular_Momentum_Error = [0.00000178, 0.00000533 , 0.00001068]
euler_cromer_Angular_Momentum_Error = [0.00000171,0.00000514, 0.00001028]
euler_richardson_Angular_Momentum_Error = [0.00000004,0.00000041,0.00000248]


def plot(particles, history, delta_T):
    """
    Sums total energy of all planets and plots it as a function of time.
    """
    # Get all planets (exclude Sun)
    planet_particles = [p for p in particles if p.name.lower() != "sun"]
    
    # Get number of steps from first planet
    first_planet = planet_particles[0]
    num_steps = len(history[first_planet.name]['x'])
    time_array = np.arange(num_steps) * delta_T / 86400  # Convert to days
    
    # Initialize summed energy arrays
    summed_total_energy = np.zeros(num_steps)
    
    # Calculate and sum energies for each planet
    for planet in planet_particles:
        # Use Tests class to calculate energy history for each planet
        kinetic_energy, potential_energy, total_energy = Tests.calculate_planet_energy_history(planet, particles, history)
        
        # Check if energy calculation was successful before plotting
        if kinetic_energy is None:
            continue
        
        plt.figure(2, figsize=(10, 6))
        plt.plot(time_array, kinetic_energy, label='Kinetic Energy', linewidth=2, color='red')
        plt.plot(time_array, potential_energy, label='Potential Energy', linewidth=2, color='green')
        plt.plot(time_array, total_energy, label='Total Energy', linewidth=2, color='blue')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Energy (J)', fontsize=12)
        plt.title(planet.name + ' Energy vs Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Sum the total energy
        summed_total_energy += total_energy
    
    # Create the plot
    plt.figure(1, figsize=(14, 8))
    
    # Plot summed total energy
    plt.plot(time_array, summed_total_energy / 1e32, label='Total Energy (All Planets)', linewidth=2, color='blue')
    
    # Set y-axis limits to zoom in on the data range
    energy_values = summed_total_energy / 1e32
    if len(energy_values) > 0:
        energy_min = np.min(energy_values)
        energy_max = np.max(energy_values)
        energy_range = energy_max - energy_min
        # Add minimal padding (1%) for tighter zoom
        if energy_range > 0:
            plt.ylim(energy_min - 0.01 * energy_range, energy_max + 0.01 * energy_range)
        else:
            plt.ylim(energy_min - abs(energy_min) * 0.01, energy_max + abs(energy_max) * 0.01)
    
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Total Energy (×10³² J)', fontsize=12)
    plt.title("Summed Planetary Energy vs Time", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add summary text showing energy conservation
    initial_total = summed_total_energy[0] / 1e32
    final_total = summed_total_energy[-1] / 1e32
    energy_change_percent = abs((final_total - initial_total) / abs(initial_total) * 100) if abs(initial_total) > 1e-10 else 0
    
    summary_text = f'Initial Total Energy: {initial_total:.6f} ×10³² J\n'
    summary_text += f'Final Total Energy: {final_total:.6f} ×10³² J\n'
    summary_text += f'Energy Change: {energy_change_percent:.6f}%'
    
    plt.text(0.02, 0.98, summary_text,
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    plt.figure(3, figsize=(10, 6))
    plt.plot(time_step, rk4_Angular_Momentum_Error, marker='o', label='RK4', linewidth=2)
    plt.plot(time_step, verlet_Angular_Momentum_Error, marker='s', label='Verlet', linewidth=2)
    plt.plot(time_step, euler_cromer_Angular_Momentum_Error, marker='^', label='Euler-Cromer', linewidth=2)
    plt.plot(time_step, euler_richardson_Angular_Momentum_Error, marker='d', label='Euler-Richardson', linewidth=2)
    plt.xlabel('Time Step (seconds)', fontsize=12)
    plt.ylabel('Angular Momentum Error (%)', fontsize=12)
    plt.title('Angular Momentum Error vs Time Step', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    plt.show()
    
    return time_array, {'Summed': summed_total_energy}, None
