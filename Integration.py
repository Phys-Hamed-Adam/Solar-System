from Particle import Particle

class IntegrationMethods:
    """
    A class containing numerical integration methods for updating the particles' positions and velocities.

    """
    
    @staticmethod
    def euler_cromer(particle, delta_T, particles=None):
        """
        
        The Euler-Cromer method updates velocity first, then uses the new velocity to update position
      
        """
        # Calculate acceleration at current position from all other particles
        particle.acceleration = particle.calculate_acceleration_from_particles(particle.position, particles)
        # Update velocity first: current velocity plus acceleration times time step
        particle.velocity += particle.acceleration * delta_T
        # Update position using the newly updated velocity
        particle.position += particle.velocity * delta_T

    @staticmethod
    def runge_kutta(particle, delta_T, particles):
        """

        The Runge-Kutta method provides better accuracy by calculating the acceleration at multiple points within the time step and then taking the weighted average.
     
        """
        # k1 calculates initial slope estimates at the current position and velocity

        a1 = particle.calculate_acceleration_from_particles(particle.position, particles)
        k1v = delta_T * a1
        k1r = delta_T * particle.velocity

        # k2 calculates slope estimates at the midpoint using k1 estimates
        
        p2 = particle.position + 0.5 * k1r
        v2 = particle.velocity + 0.5 * k1v
        # Calculate acceleration and velocity changes at this midpoint
        a2 = particle.calculate_acceleration_from_particles(p2, particles)
        k2v = delta_T * a2
        k2r = delta_T * v2

        # k3 calculates slope estimates at another midpoint using k2 estimates
     
        p3 = particle.position + 0.5 * k2r
        v3 = particle.velocity + 0.5 * k2v
        # Calculate acceleration and velocity changes at this refined midpoint
        a3 = particle.calculate_acceleration_from_particles(p3, particles)
        k3v = delta_T * a3
        k3r = delta_T * v3

        # k4 calculates slope estimates at the end of the time step using k3 estimates
       
        p4 = particle.position + k3r
        v4 = particle.velocity + k3v
        # Calculate acceleration and velocity changes at the end point
        a4 = particle.calculate_acceleration_from_particles(p4, particles)
        k4v = delta_T * a4
        k4r = delta_T * v4

        # Calculate the weighted average of all four estimates
        dp = (k1r + 2*k2r + 2*k3r + k4r) / 6
        dv = (k1v + 2*k2v + 2*k3v + k4v) / 6
        
        # Update position and velocity
        particle.position += dp
        particle.velocity += dv
        # Update the stored acceleration 
        particle.acceleration = a4

    @staticmethod
    def euler_richardson(particle, delta_T, particles):
        """
        
        This method uses a midpoint evaluation to achieve second-order accuracy while maintaining the simplicity of Euler.
    
    
        """
        # Calculate acceleration at the current position
        a1 = particle.calculate_acceleration_from_particles(particle.position, particles)
        v1 = particle.velocity

        # Estimate position and velocity at midpoint of the time step
        mid_pos = particle.position + 0.5 * delta_T * v1
        mid_vel = particle.velocity + 0.5 * delta_T * a1

        # Calculate acceleration at the midpoint position
        a2 = particle.calculate_acceleration_from_particles(mid_pos, particles)

        # Update position and velocity using the the midpoint acceleration
        particle.position += delta_T * mid_vel
        particle.velocity += delta_T * a2
        # Update stored the acceleration
        particle.acceleration = a2
        
    @staticmethod
    def verlet(particle, delta_T, particles, a_next=None):

        # Store current acceleration
        a_current = particle.acceleration.copy()
        
        # Update position using Taylor expansion
        particle.position = particle.position + particle.velocity * delta_T + 0.5 * particle.acceleration * delta_T**2
        # Calculate acceleration at the new position 
        a_next = particle.calculate_acceleration_from_particles(particle.position, particles)
        
        # Update velocity using average of current and next acceleration
       
        particle.velocity = particle.velocity + 0.5 * (a_current + a_next) * delta_T
        # Update stored acceleration to the new value
        particle.acceleration = a_next
        
