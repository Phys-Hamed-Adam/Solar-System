# Import required libraries 
import numpy as np
from astropy.constants import G  # Gravitational constant G

class Particle:
    """
    Represents a physical object in the simulation.
    Stores position, velocity, acceleration, mass, and other properties for numerical integration.
    """

    G = G.value  # Gravitational constant (m^3 kg^-1 s^-2) from astropy.constants
    
    def __init__(
        self,
        position=None,
        velocity=None,
        acceleration=None,
        name=None,
        mass=None):
        
        """
        Initialize a Particle with position, velocity, acceleration, name, and mass.
        Arrays are stored as copies and converted to float.
        """
        
        self.position = np.array(position, dtype=float) if position is not None else np.zeros(3)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)
        self.acceleration = np.array(acceleration, dtype=float) if acceleration is not None else np.zeros(3)

        # Store particle name and mass 
        self.name = name
        self.mass = float(mass)
    
    def calculate_acceleration_from_particles(self, position, particles):
        """
        Calculates total gravitational acceleration at a given position due to all other particles.
        """
        total_acceleration = np.zeros(3)

        for other in particles:
            if other is not self:
                # Calculate vector from given position to other particle
                r = other.position - position
                r_mag = np.linalg.norm(r)

                if r_mag > 1e-10:
                    # Calculate gravitational acceleration: a = G * m / r^2 * r_hat
                    # where r_hat = r / r_mag, so: a = G * m / r^3 * r
                    total_acceleration += (
                        self.G * other.mass / r_mag**3
                    ) * r

        return total_acceleration
        
    def __str__(self):
        """
        Returns a formatted string representation of the particle,
        showing its name, mass, position, velocity, and acceleration.
        """
        return "Particle: {0}, Mass: {1:.3e}, Position: {2}, Velocity: {3}, Acceleration: {4}".format(
        self.name, self.mass,self.position, self.velocity, self.acceleration
    ) 
