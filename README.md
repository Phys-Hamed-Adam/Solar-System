# Gravitational N-Body Simulation

Python-based N-body simulation of the Solar System with multiple numerical integration methods.

## Installation

```bash
pip install numpy matplotlib pygame customtkinter astropy poliastro spiceypy
```

## Usage

Run the GUI:
```bash
python Simulation.py
```

Set time step, duration, and integration method, then click "Run Simulation".

## Features

- **Integration Methods**: Euler-Cromer, RK4, Verlet, Euler-Richardson
- **Real Data**: JPL ephemeris data via astropy
- **Visualization**: Real-time Pygame visualization
- **Analysis**: Energy conservation, orbital periods, escape velocities
- **Auto-save**: Results saved to Desktop JSON files

## Files & Classes

### `Data.py`
- **`get_data(name, time_obj)`**: Retrieves position/velocity from JPL ephemeris and transforms to ecliptic coordinates
- Creates `Particle` objects for Sun and all 8 planets with real initial conditions

### `Particle.py`
- **`Particle` class**: Represents a celestial body
  - **`__init__()`**: Initializes position, velocity, acceleration, mass, name
  - **`calculate_acceleration_from_particles()`**: Calculates gravitational acceleration from all other particles

### `Integration.py`
- **`IntegrationMethods` class**: Static methods for numerical integration
  - **`euler_cromer()`**: First-order method, updates velocity then position
  - **`runge_kutta()`**: 4th-order RK4 method with weighted averages
  - **`euler_richardson()`**: Second-order method using midpoint evaluation
  - **`verlet()`**: Symplectic method using Taylor expansion

### `Simulation.py`
- **`Simulation` class**:
  - **`simulate()`**: Main simulation loop, tracks orbits, handles visualization, saves results
- **`draw(history, particles)`**: Replays simulation from stored history data
- **`run_simulation()`**: GUI callback function that runs simulation

### `tests.py`
- **`Tests` class**: Analysis and validation methods
  - **`conservation_of_linear_momentum()`**: Tests if total momentum is conserved
  - **`conservation_of_angular_momentum()`**: Tests if angular momentum is conserved
  - **`conservation_of_energy()`**: Tests if total energy is conserved, returns energy history
  - **`calculate_planet_energy_history()`**: Calculates kinetic, potential, total energy over time for any planet
  - **`calculate_jupiter_energy_history()`**: Wrapper for Jupiter's energy history
  - **`escape_velocity()`**: Calculates and compares planetary escape velocities
  - **`period_error()`**: Calculates orbital periods and compares with known values
  - **`save_test_results()`**: Saves all test results to JSON file on Desktop

### `plot.py`
- **`plot(particles, history, delta_T)`**: Creates energy-time graphs for each planet

## Simulation Behavior

Simulation ends when Uranus completes one orbit (~30,660 days) or reaches 120,000 days. Results are then automatically saved and plotted.
