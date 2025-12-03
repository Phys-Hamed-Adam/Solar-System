import numpy as np
from poliastro import constants
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy.constants import G  # Newton's gravitational constant
from spiceypy import sxform, mxvg
import matplotlib.pyplot as plt
import pygame as pg
# Get positions and velocities for the Sun.# Get the time at 5pm on 27th Nov 2019.
t = Time("2019-11-27 17:00:00.0", scale="tdb")

class Particle:
    """
    A Particle class representing a physical object in a simulation.
    Stores position, velocity, acceleration, mass, and other properties
    needed for numerical integration methods in physics simulations.
    """
    # Gravitational constant (m^3 kg^-1 s^-2)
    G = G.value  # Using value from astropy.constants
    
    def __init__(
        self,
        position=None,
        velocity=None,
        acceleration=None,
        name=None,
        mass=None):
        
        """
        Initialises a Particle object with position, velocity, acceleration,
        name, and mass. Arrays are stored as copies and converted to float.
        """
        
        # Handles default mutable arguments to avoid shared state issues
        # If no position is provided, initialize at origin (0, 0, 0)
        if position is None:
            position = np.array([0, 0, 0], dtype=float)
        # If no velocity is provided, initialize with zero velocity
        if velocity is None:
            velocity = np.array([0, 0, 0], dtype=float)
        # If no acceleration is provided, initialize with default downward acceleration
        # (simulating gravity, approximately -10 m/s^2 in the y-direction)
        if acceleration is None:
            acceleration = np.array([0, -10, 0], dtype=float)  # g

        # Stores the copies of the arrays to prevent shared references
        # Converting to numpy arrays ensures proper vector operations and prevents
        # accidental modification of input arrays from outside the class
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array(acceleration, dtype=float)

        # Store particle name and mass as instance variables
        self.name = name
        self.mass = float(mass)
    
    def calculate_acceleration_from_particles(self, position, particles):
        """
        Calculates the total gravitational acceleration of a particle at a given position
        due to all other particles in the system.
        
        Args:
            position: The position vector (numpy array) where acceleration should be calculated
            particles: List of Particle objects that exert gravitational forces
            
        Returns:
            total_acceleration: 3D acceleration vector (numpy array)
        """
        # Initialize total acceleration as zero vector
        total_acceleration = np.array([0.0, 0.0, 0.0])
        
        # Sum up gravitational acceleration from all other particles
        for particle in particles:
            if particle is not self:  # Don't calculate acceleration from self
                # Calculate displacement vector from this position to the other particle
                r_vector = particle.position - position
                r_magnitude = np.linalg.norm(r_vector)
                
                # Avoid division by zero if particles are at same position
                if r_magnitude > 1e-10:
                    # Calculate unit vector and acceleration magnitude
                    r_hat = r_vector / r_magnitude
                    a_magnitude = self.G * particle.mass / (r_magnitude ** 2)
                    # Add gravitational acceleration (points toward the other particle)
                    total_acceleration += a_magnitude * r_hat
        
        return total_acceleration
        
    def __str__(self):
        """
        Returns a formatted string representation of the particle,
        showing its name, mass, position, velocity, and acceleration.
        """
        return "Particle: {0}, Mass: {1:.3e}, Position: {2}, Velocity: {3}, Acceleration: {4}".format(
        self.name, self.mass,self.position, self.velocity, self.acceleration
    ) 
        
    def Euler_Cromer(self, delta_T, particles=None):
        """
        Updates the particle's position and velocity using the Euler-Cromer method
        (also known as semi-implicit Euler). This is a first-order numerical integration method.
        
        The Euler-Cromer method updates velocity first, then uses the new velocity to update position:
        velocity(t+dt) = velocity(t) + acceleration(t)*dt
        position(t+dt) = position(t) + velocity(t+dt)*dt
        
        This provides better energy conservation than the standard Euler method.
        
        Args:
            delta_T: Time step for the integration
            particles: Optional list of Particle objects. If provided, acceleration will be recalculated
                    from gravitational forces. If None, uses stored acceleration value.
        """
        # If particles are provided, recalculate acceleration from gravitational forces
        if particles is not None:
            self.acceleration = self.calculate_acceleration_from_particles(self.position, particles)
        
        # Update velocity first: current velocity plus acceleration times time step
        self.velocity += self.acceleration * delta_T
        # Update position using the newly updated velocity
        self.position += self.velocity * delta_T

    def Runge_Kutta(self, delta_T, particles):
        """
        Updates the particle's position and velocity using the 4th-order Runge-Kutta method.
        This is a more accurate numerical integration method than Euler, using four
        intermediate calculations (k1, k2, k3, k4) to estimate the change in position and velocity.
        
        The Runge-Kutta method provides better accuracy by evaluating the acceleration
        at multiple points within the time step and taking a weighted average.
        
        Args:
            delta_T: Time step for the integration
            particles: List of all Particle objects in the simulation (for calculating gravitational forces)
        """
        # k1: Calculate initial slope estimates at the current position and velocity
        # These are the first approximations of velocity and position changes
        a1 = self.calculate_acceleration_from_particles(self.position, particles)
        k1v = delta_T * a1
        k1r = delta_T * self.velocity

        # k2: Calculate slope estimates at the midpoint using k1 estimates
        # Estimate position and velocity at midpoint of time step
        p2 = self.position + 0.5 * k1r
        v2 = self.velocity + 0.5 * k1v
        # Calculate acceleration and velocity changes at this midpoint
        a2 = self.calculate_acceleration_from_particles(p2, particles)
        k2v = delta_T * a2
        k2r = delta_T * v2

        # k3: Calculate slope estimates at another midpoint using k2 estimates
        # Refine position and velocity estimates at midpoint using k2
        p3 = self.position + 0.5 * k2r
        v3 = self.velocity + 0.5 * k2v
        # Calculate acceleration and velocity changes at this refined midpoint
        a3 = self.calculate_acceleration_from_particles(p3, particles)
        k3v = delta_T * a3
        k3r = delta_T * v3

        # k4: Calculate slope estimates at the end of the time step using k3 estimates
        # Estimate position and velocity at the end of the time step
        p4 = self.position + k3r
        v4 = self.velocity + k3v
        # Calculate acceleration and velocity changes at the end point
        a4 = self.calculate_acceleration_from_particles(p4, particles)
        k4v = delta_T * a4
        k4r = delta_T * v4

        # Combine: Calculate weighted average of all four estimates
        # The weights (1, 2, 2, 1) give more importance to the midpoint estimates (k2, k3)
        # This provides the final change in position (dp) and velocity (dv) for the time step
        dp = (k1r + 2*k2r + 2*k3r + k4r) / 6
        dv = (k1v + 2*k2v + 2*k3v + k4v) / 6
        
        # Apply the calculated changes to update position and velocity
        self.position += dp
        self.velocity += dv
        # Update stored acceleration to the final calculated acceleration
        self.acceleration = a4

    def Euler_Richardson(self, delta_T, particles):
        """
        Euler-Richardson method (also known as Richardson extrapolation or midpoint method).
        This method uses a midpoint evaluation to achieve second-order accuracy 
        while maintaining the simplicity of Euler.
        
        The method:
        1. Calculates acceleration at current position
        2. Estimates position and velocity at midpoint
        3. Calculates acceleration at midpoint
        4. Uses midpoint acceleration to update position and velocity
        
        Args:
            delta_T: Time step for the integration
            particles: List of all Particle objects in the simulation (for calculating gravitational forces)
        """
        # Calculate acceleration at the current position
        a1 = self.calculate_acceleration_from_particles(self.position, particles)
        v1 = self.velocity

        # Estimate position and velocity at midpoint of the time step
        mid_pos = self.position + 0.5 * delta_T * v1
        mid_vel = self.velocity + 0.5 * delta_T * a1

        # Calculate acceleration at the midpoint position
        a2 = self.calculate_acceleration_from_particles(mid_pos, particles)

        # Update position and velocity using the midpoint acceleration
        self.position += delta_T * mid_vel
        self.velocity += delta_T * a2
        # Update stored acceleration
        self.acceleration = a2
        
    def verlet(self, delta_T, particles, a_next=None):
        """
        Verlet integration method for updating position and velocity.
        This method is commonly used in molecular dynamics simulations because
        it provides good energy conservation and is time-reversible.
        
        Standard Verlet algorithm:
        1. Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. Calculate new acceleration a(t+dt)
        3. Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        
        Args:
            delta_T: Time step for the integration
            particles: List of all Particle objects in the simulation (for calculating gravitational forces)
            a_next: Optional pre-calculated acceleration at next time step. If None, will be calculated.
        """
        # Store current acceleration
        a_current = self.acceleration.copy()
        
        # Update position using Taylor expansion: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        self.position = self.position + self.velocity * delta_T + 0.5 * self.acceleration * delta_T**2
        # Calculate acceleration at the new position (if not provided)
        if a_next is None:
            a_next = self.calculate_acceleration_from_particles(self.position, particles)
        
        # Update velocity using average of current and next acceleration
        # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.velocity = self.velocity + 0.5 * (a_current + a_next) * delta_T
        
        # Update stored acceleration to the new value
        self.acceleration = a_next
        
    def updateGravitationalAcceleration(self,body):
        """
        Calculates the gravitational acceleration vector acting on this particle
        due to another body (e.g., planet, moon, or star).
        
        Uses Newton's law of universal gravitation:
        F = G * (m1 * m2) / r^2
        a = F / m = G * m2 / r^2
        
        Args:
            body: Another Particle object that exerts gravitational force on this particle
            
        Returns:
            a_vector: 3D acceleration vector pointing from this particle toward the body
        """
        
        # Calculate the displacement vector from this particle to the other body
        r_vector= body.position - self.position
        # Calculate the distance (magnitude) between the two particles
        r_magnitude= np.linalg.norm(r_vector)
        # Calculate the unit vector pointing from this particle to the other body
        # This gives the direction of the gravitational force
        r_hat= r_vector/r_magnitude
        # Calculate the magnitude of gravitational acceleration using Newton's law
        # a = G * M / r^2, where M is the mass of the attracting body
        a_magnitude= self.G*body.mass/r_magnitude**2
        # Calculate the acceleration vector (negative sign because acceleration
        # points toward the attracting body, opposite to r_hat direction)
        a_vector=-a_magnitude*r_hat

        return a_vector
    
class Simulation:
    
    def simulate(self, particles, delta_T, method="Euler_Cromer", duration=3600, visualize=False):
        
        """
        Simulates the motion of particles over a given time period using a specified integration method.
        
        Args:
            particles: List of Particle objects to simulate
            delta_T: Time step for the integration
            method: Integration method to use (Euler_Cromer, Runge_Kutta, Euler_Richardson, verlet)
            duration: Duration of simulation in seconds
            visualize: If True, shows pygame visualization in real-time
        """
        
        time = 0
        history = {particle.name: {'x': [], 'y': [], 'z': []} for particle in particles}
        
        # Initialize pygame visualization if requested
        if visualize:
            pg.init()
            screen = pg.display.set_mode((1200, 800))
            pg.display.set_caption("N-Body Solar System Simulation")
            clock = pg.time.Clock()
            running = True
            
            # Color mapping for planets
            colors = {
                'sun': (255, 255, 0),
                'Mercury': (169, 169, 169),
                'Venus': (255, 165, 0),
                'Earth': (0, 100, 200),
                'Mars': (255, 0, 0),
                'Jupiter': (255, 200, 100),
                'Saturn': (255, 220, 150),
                'Uranus': (100, 200, 255),
                'Neptune': (0, 0, 255)
            }
            
            # Calculate scale and offset for visualization
            # Find the range of positions to center the view
            all_positions = np.array([p.position for p in particles])
            center = np.mean(all_positions, axis=0)
            max_dist = np.max(np.linalg.norm(all_positions - center, axis=1))
            scale = min(400, 400) / max_dist if max_dist > 0 else 1
            
        while time < duration:
            # Handle pygame events
            if visualize:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                        break
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            running = False
                            break
                if not running:
                    break
            
            for particle in particles:
                # Store history before updating
                history[particle.name]['x'].append(particle.position[0])
                history[particle.name]['y'].append(particle.position[1])
                history[particle.name]['z'].append(particle.position[2])

                if method == "Euler_Cromer":
                    particle.Euler_Cromer(delta_T, particles)
                elif method == "Runge_Kutta":
                    particle.Runge_Kutta(delta_T, particles)
                elif method == "Euler_Richardson":  
                    particle.Euler_Richardson(delta_T, particles)
                elif method == "verlet":
                    particle.verlet(delta_T, particles)
            
            # Update visualization
            if visualize:
                screen.fill((0, 0, 0))  # Black background
                
                # Draw trajectories
                for particle in particles:
                    if len(history[particle.name]['x']) > 1:
                        color = colors.get(particle.name, (255, 255, 255))
                        points = []
                        for i in range(len(history[particle.name]['x'])):
                            x = history[particle.name]['x'][i] - center[0]
                            y = history[particle.name]['y'][i] - center[1]
                            screen_x = int(600 + x * scale)
                            screen_y = int(400 + y * scale)
                            points.append((screen_x, screen_y))
                        if len(points) > 1:
                            pg.draw.lines(screen, color, False, points, 1)
                
                # Draw particles
                for particle in particles:
                    x = particle.position[0] - center[0]
                    y = particle.position[1] - center[1]
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    color = colors.get(particle.name, (255, 255, 255))
                    
                    # Size based on mass (log scale)
                    size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    pg.draw.circle(screen, color, (screen_x, screen_y), size)
                
                # Display time
                font = pg.font.Font(None, 36)
                time_text = font.render(f"Time: {time/86400:.2f} days", True, (255, 255, 255))
                screen.blit(time_text, (10, 10))
                
                pg.display.flip()
                clock.tick(60)  # Limit to 60 FPS
            
            time += delta_T
        
        # Keep window open after simulation completes
        if visualize:
            # Show final state and wait for user to close window
            screen.fill((0, 0, 0))  # Black background
            
            # Draw all trajectories
            for particle in particles:
                if len(history[particle.name]['x']) > 1:
                    color = colors.get(particle.name, (255, 255, 255))
                    points = []
                    for i in range(len(history[particle.name]['x'])):
                        x = history[particle.name]['x'][i] - center[0]
                        y = history[particle.name]['y'][i] - center[1]
                        screen_x = int(600 + x * scale)
                        screen_y = int(400 + y * scale)
                        points.append((screen_x, screen_y))
                    if len(points) > 1:
                        pg.draw.lines(screen, color, False, points, 1)
            
            # Draw final particle positions
            for particle in particles:
                x = particle.position[0] - center[0]
                y = particle.position[1] - center[1]
                screen_x = int(600 + x * scale)
                screen_y = int(400 + y * scale)
                color = colors.get(particle.name, (255, 255, 255))
                size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                pg.draw.circle(screen, color, (screen_x, screen_y), size)
            
            # Display completion message
            font = pg.font.Font(None, 36)
            time_text = font.render(f"Simulation Complete - {time/86400:.2f} days", True, (255, 255, 255))
            screen.blit(time_text, (10, 10))
            instruction_text = font.render("Press ESC or close window to exit", True, (200, 200, 200))
            screen.blit(instruction_text, (10, 50))
            
            pg.display.flip()
            
            # Wait for user to close window
            waiting = True
            while waiting:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        waiting = False
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            waiting = False
                clock.tick(60)
            
            pg.quit()
            
        return history 
 
def get_data(name, time_obj=None):
    if time_obj is None:
        time_obj = t
    pos, vel = get_body_barycentric_posvel(name, time_obj, ephemeris="jpl")    

    # Make a "state vector" of positions and velocities (in metres and metres/second, respectively).
    statevec = [ 
        pos.xyz[0].to("m").value,
        pos.xyz[1].to("m").value,
        pos.xyz[2].to("m").value,
        vel.xyz[0].to("m/s").value,
        vel.xyz[1].to("m/s").value,
        vel.xyz[2].to("m/s").value,]

    # Get transformation matrix to the ecliptic (use time in Julian Days).
    trans = sxform("J2000", "ECLIPJ2000", t.jd)

    # Transform state vector to ecliptic.
    statevececl = mxvg(trans, statevec)
    # Get positions and velocities.
    position = [statevececl[0], statevececl[1], statevececl[2]]
    velocity = [statevececl[3], statevececl[4], statevececl[5]]
    return position, velocity

msun = (constants.GM_sun / G).value
mmercury = (constants.GM_mercury / G).value
mvenus= (constants.GM_venus / G).value
mearth = (constants.GM_earth / G).value 
mmars= (constants.GM_mars / G).value
mjupiter= (constants.GM_jupiter / G).value
msaturn= (constants.GM_saturn / G).value
muranus= (constants.GM_uranus / G).value
mneptune= (constants.GM_neptune / G).value

position_sun, velocity_sun = get_data("sun", t)
position_mercury, velocity_mercury = get_data("mercury", t)
position_venus, velocity_venus = get_data("venus", t)
position_earth, velocity_earth = get_data("earth", t)
position_mars, velocity_mars = get_data("mars", t)
position_jupiter, velocity_jupiter = get_data("jupiter", t)
position_saturn, velocity_saturn = get_data("saturn", t)
position_uranus, velocity_uranus = get_data("uranus", t)
position_neptune, velocity_neptune= get_data("neptune", t)

Sun =Particle(
        position=position_sun,
        velocity=velocity_sun,
        acceleration=None,
        name="sun",
        mass=msun
)
          
Mercury =Particle(
        position=position_mercury,
        velocity=velocity_mercury,
        acceleration=None,
        name="Mercury",
        mass=mmercury
)

Venus =Particle(
        position=position_venus,
        velocity=velocity_venus,
        acceleration=None,
        name="Venus",
        mass=mvenus
)
Earth =Particle(
        position=position_earth,
        velocity=velocity_earth,
        acceleration=None,
        name="Earth",
        mass=mearth
)
Mars =Particle(
        position=position_mars,
        velocity=velocity_mars,
        acceleration=None,
        name="Mars",
        mass=mmars
)
Jupiter =Particle(
        position=position_jupiter,
        velocity=velocity_jupiter,
        acceleration=None,
        name="Jupiter",
        mass=mjupiter
)
Saturn =Particle(
        position=position_saturn,
        velocity=velocity_saturn,
        acceleration=None,
        name="Saturn",
        mass=msaturn
)
Uranus =Particle(
        position=position_uranus,
        velocity=velocity_uranus,
        acceleration=None,
        name="Uranus",
        mass=muranus
)
Neptune =Particle(
        position=position_neptune,
        velocity=velocity_neptune,
        acceleration=None,
        name="Neptune",
        mass=mneptune)
                        
def visualize_with_pygame(history, particles):
    """
    Visualize simulation results using pygame from stored history.
    
    Args:
        history: Dictionary containing position history for each particle
        particles: List of Particle objects
    """
    pg.init()
    screen = pg.display.set_mode((1200, 800))
    pg.display.set_caption("N-Body Solar System Simulation - History")
    clock = pg.time.Clock()
    
    # Color mapping for planets
    colors = {
        'sun': (255, 255, 0),
        'Mercury': (169, 169, 169),
        'Venus': (255, 165, 0),
        'Earth': (0, 100, 200),
        'Mars': (255, 0, 0),
        'Jupiter': (255, 200, 100),
        'Saturn': (255, 220, 150),
        'Uranus': (100, 200, 255),
        'Neptune': (0, 0, 255)
    }
    
    # Calculate scale and offset
    all_x = []
    all_y = []
    for name, data in history.items():
        all_x.extend(data['x'])
        all_y.extend(data['y'])
    
    if all_x and all_y:
        center_x = (max(all_x) + min(all_x)) / 2
        center_y = (max(all_y) + min(all_y)) / 2
        max_dist = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y)
        ) / 2
        scale = min(400, 400) / max_dist if max_dist > 0 else 1
    else:
        center_x, center_y = 0, 0
        scale = 1
    
    running = True
    frame = 0
    max_frames = max([len(data['x']) for data in history.values()]) if history else 0
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
        
        screen.fill((0, 0, 0))
        
        # Draw trajectories up to current frame
        for particle in particles:
            name = particle.name
            if name in history and len(history[name]['x']) > 0:
                color = colors.get(name, (255, 255, 255))
                points = []
                for i in range(min(frame, len(history[name]['x']))):
                    x = history[name]['x'][i] - center_x
                    y = history[name]['y'][i] - center_y
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    points.append((screen_x, screen_y))
                if len(points) > 1:
                    pg.draw.lines(screen, color, False, points, 1)
                
                # Draw current position
                if frame < len(history[name]['x']):
                    x = history[name]['x'][frame] - center_x
                    y = history[name]['y'][frame] - center_y
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    pg.draw.circle(screen, color, (screen_x, screen_y), size)
        
        # Display frame info
        font = pg.font.Font(None, 36)
        frame_text = font.render(f"Frame: {frame}/{max_frames}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 10))
        
        pg.display.flip()
        clock.tick(30)  # 30 FPS
        
        frame += 1
        if frame >= max_frames:
            frame = 0  # Loop animation
    
    pg.quit()

# Run simulation if this file is executed directly
if __name__ == "__main__":
    # Create list of all particles
    particles = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
    
    # Create simulation instance
    sim = Simulation()
    
    # Run simulation with a small time step (in seconds)
    # Duration: 1 day = 86400 seconds
    # Time step: 1 hour = 3600 seconds
    print("Running simulation...")
    print("Press ESC or close window to exit visualization")
    
    # Run with real-time pygame visualization
    history = sim.simulate(particles, delta_T=3600, method="Runge_Kutta", duration=31536000, visualize=True)
    
