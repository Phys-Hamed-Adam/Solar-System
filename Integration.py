from Particle import Particle

class IntegrationMethods:
    """
    Numerical integration methods for updating particle positions and velocities.
    """
    
    @staticmethod
    def euler_cromer(particle, delta_T, particles=None):
        """
        Euler-Cromer method: update velocity first, then use new velocity to update position.
        """
        # Calculate acceleration at current position
        particle.acceleration = particle.calculate_acceleration_from_particles(particle.position, particles)
        # Update velocity
        particle.velocity += particle.acceleration * delta_T
        # Update position using new velocity
        particle.position += particle.velocity * delta_T

    @staticmethod
    def runge_kutta(particle, delta_T, particles):
        """
        Runge-Kutta 4th order method: calculates acceleration at multiple points and takes weighted average.
        """
        # k1: initial slope estimates
        a1 = particle.calculate_acceleration_from_particles(particle.position, particles)
        k1v = delta_T * a1
        k1r = delta_T * particle.velocity

        # k2: midpoint estimates using k1
        p2 = particle.position + 0.5 * k1r
        v2 = particle.velocity + 0.5 * k1v
        a2 = particle.calculate_acceleration_from_particles(p2, particles)
        k2v = delta_T * a2
        k2r = delta_T * v2

        # k3: another midpoint using k2
        p3 = particle.position + 0.5 * k2r
        v3 = particle.velocity + 0.5 * k2v
        a3 = particle.calculate_acceleration_from_particles(p3, particles)
        k3v = delta_T * a3
        k3r = delta_T * v3

        # k4: end point using k3
        p4 = particle.position + k3r
        v4 = particle.velocity + k3v
        a4 = particle.calculate_acceleration_from_particles(p4, particles)
        k4v = delta_T * a4
        k4r = delta_T * v4

        # Weighted average of all four estimates
        dp = (k1r + 2*k2r + 2*k3r + k4r) / 6
        dv = (k1v + 2*k2v + 2*k3v + k4v) / 6
        
        # Update position and velocity
        particle.position += dp
        particle.velocity += dv
        particle.acceleration = a4

    @staticmethod
    def euler_richardson(particle, delta_T, particles):
        """
        Euler-Richardson method: uses midpoint evaluation for second-order accuracy.
        """
        # Calculate acceleration at current position
        a1 = particle.calculate_acceleration_from_particles(particle.position, particles)
        v1 = particle.velocity

        # Estimate position and velocity at midpoint
        mid_pos = particle.position + 0.5 * delta_T * v1
        mid_vel = particle.velocity + 0.5 * delta_T * a1

        # Calculate acceleration at midpoint
        a2 = particle.calculate_acceleration_from_particles(mid_pos, particles)

        # Update position and velocity using midpoint acceleration
        particle.position += delta_T * mid_vel
        particle.velocity += delta_T * a2
        particle.acceleration = a2
        
    @staticmethod
    def verlet(particle, delta_T, particles, a_next=None):
        """
        Verlet method: uses Taylor expansion for position, averages acceleration for velocity.
        """
        # Store current acceleration
        a_current = particle.acceleration.copy()
        
        # Update position using Taylor expansion
        particle.position = particle.position + particle.velocity * delta_T + 0.5 * particle.acceleration * delta_T**2
        # Calculate acceleration at new position
        a_next = particle.calculate_acceleration_from_particles(particle.position, particles)
        
        # Update velocity using average of current and next acceleration
        particle.velocity = particle.velocity + 0.5 * (a_current + a_next) * delta_T
        particle.acceleration = a_next
        
