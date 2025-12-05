# Import required libraries 
import numpy as np
from poliastro import constants  # Planetary constants 
from astropy.coordinates import get_body_barycentric_posvel  # Planetary positions/velocities
from astropy.time import Time  # Time handling for astronomical calculations
from astropy.constants import G  # Gravitational constant G
from spiceypy import sxform, mxvg  # Coordinate transformations 
import matplotlib.pyplot as plt  # Plotting 
import pygame as pg  # Visualization
import customtkinter  # Graphical user interface


# Setup the graphical user interface
app = customtkinter.CTk()  
app.title("Gravitational Simulation")

# Time step entry field 
Time_step = customtkinter.CTkEntry(app, placeholder_text="Time Step (seconds)")
Time_step.insert(0, "3600")  # Default: 1 hour (3600 seconds)
Time_step.grid(row=0, column=4, padx=20, pady=20)

# Duration entry field 
Duration = customtkinter.CTkEntry(app, placeholder_text="Duration (seconds)")
Duration.insert(0, "86400")  # Default: 1 day (86400 seconds)
Duration.grid(row=0, column=5, padx=20, pady=20)

# Agrothim selection
optionmenu = customtkinter.CTkOptionMenu(app, values=["RK4", "Verlet", "Euler_Cromer", "Euler_Richardson"])
optionmenu.set("RK4")  # Default to Runge-Kutta 4th order (most accurate)
optionmenu.grid(row=1, column=5, padx=20, pady=20)

# Fetch the data of the planets

t = Time("2019-11-27 17:00:00.0", scale="tdb")

class Particle:
    """
    A Particle class representing a physical object in a simulation.
    Stores position, velocity, acceleration, mass, and the  other required properties
    for numerical integration.
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
            acceleration = np.array([0, 0, 0], dtype=float)  # g

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
        with respect to all the other particles in the system.
        
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
        
        # Initialize simulation time and history tracking
        # History dictionary stores position coordinates for each particle over time
        time = 0
        history = {particle.name: {'x': [], 'y': [], 'z': []} for particle in particles}
        
        # ========================================================================
        # PYGAME VISUALIZATION SETUP
        # ========================================================================
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
                'Neptune': (0, 0, 255),
                'Pluto': (255, 255, 255)
            }
            
            # Calculate scale and offset for visualization
            # Find the range of positions to center the view
            all_positions = np.array([p.position for p in particles])
            center = np.mean(all_positions, axis=0)
            max_dist = np.max(np.linalg.norm(all_positions - center, axis=1))
            # Use a reasonable scale - make sure objects are visible
            scale = min(350, 350) / max_dist if max_dist > 0 else 1e-10
            print(f"Visualization scale: {scale}, max_dist: {max_dist}, center: {center}")
            
        # ========================================================================
        # MAIN SIMULATION LOOP
        # ========================================================================
        while time < duration:
            # Handle pygame events (window close, ESC key, etc.)
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
                    break  # Exit simulation if user closed window
            
            # Update each particle's position and velocity
            for particle in particles:
                # Store current position in history before updating
                # This allows us to visualize trajectories after simulation
                history[particle.name]['x'].append(particle.position[0])
                history[particle.name]['y'].append(particle.position[1])
                history[particle.name]['z'].append(particle.position[2])

                # Apply numerical integration method based on user selection
                # Map GUI method names to actual method names
                if method == "Euler_Cromer":
                    particle.Euler_Cromer(delta_T, particles)
                elif method == "RK4" or method == "Runge_Kutta":
                    # RK4 = Runge-Kutta 4th order (most accurate)
                    particle.Runge_Kutta(delta_T, particles)
                elif method == "Euler_Richardson":  
                    particle.Euler_Richardson(delta_T, particles)
                elif method == "Verlet" or method == "verlet":
                    # Verlet method (good energy conservation)
                    particle.verlet(delta_T, particles)
            
            # ====================================================================
            # UPDATE VISUALIZATION (real-time rendering)
            # ====================================================================
            if visualize:
                screen.fill((0, 0, 0))  # Clear screen with black background
                
                # Draw particle trajectories (paths they've traveled)
                for particle in particles:
                    if len(history[particle.name]['x']) > 1:
                        # Get color for this particle from color mapping
                        color = colors.get(particle.name, (255, 255, 255))
                        points = []
                        # Convert all historical positions to screen coordinates
                        for i in range(len(history[particle.name]['x'])):
                            # Convert from world coordinates to screen coordinates
                            # Center is at (600, 400) - middle of 1200x800 screen
                            x = history[particle.name]['x'][i] - center[0]
                            y = history[particle.name]['y'][i] - center[1]
                            screen_x = int(600 + x * scale)
                            screen_y = int(400 + y * scale)
                            # Only add points that are visible on screen
                            if 0 <= screen_x < 1200 and 0 <= screen_y < 800:
                                points.append((screen_x, screen_y))
                        # Draw trajectory line if we have at least 2 points
                        if len(points) > 1:
                            pg.draw.lines(screen, color, False, points, 1)
                
                # Draw current particle positions as circles
                for particle in particles:
                    # Convert current position to screen coordinates
                    x = particle.position[0] - center[0]
                    y = particle.position[1] - center[1]
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    color = colors.get(particle.name, (255, 255, 255))
                    
                    # Calculate particle size based on mass (logarithmic scale)
                    # Larger masses appear as larger circles
                    if particle.mass > 0:
                        size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    else:
                        size = 5
                    
                    # Only draw if particle is visible on screen
                    if 0 <= screen_x < 1200 and 0 <= screen_y < 800:
                        pg.draw.circle(screen, color, (screen_x, screen_y), size)
                        # Draw particle name label next to the circle
                        font_small = pg.font.Font(None, 20)
                        name_text = font_small.render(particle.name, True, color)
                        screen.blit(name_text, (screen_x + size + 2, screen_y - 10))
                
                # Display current simulation time in days
                font = pg.font.Font(None, 36)
                time_text = font.render(f"Time: {time/86400:.2f} days", True, (255, 255, 255))
                screen.blit(time_text, (10, 10))
                
                # Update display and limit frame rate to 60 FPS
                pg.display.flip()
                clock.tick(60)  
            
            # Advance simulation time by one time step
            time += delta_T
        
        # ========================================================================
        # POST-SIMULATION VISUALIZATION
        # ========================================================================
        # Keep window open after simulation completes to show final state
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
    """
    Retrieves position and velocity data for a celestial body from JPL ephemeris.
    
    Args:
        name: Name of the celestial body (e.g., "sun", "earth", "mars")
        time_obj: Time object for ephemeris lookup. If None, uses default time 't'.
    
    Returns:
        position: 3D position vector in meters [x, y, z] in ecliptic coordinates
        velocity: 3D velocity vector in m/s [vx, vy, vz] in ecliptic coordinates
    """
    # Use default time if none provided
    if time_obj is None:
        time_obj = t
    
    # Get barycentric position and velocity from JPL ephemeris
    # Barycentric = relative to solar system barycenter (center of mass)
    pos, vel = get_body_barycentric_posvel(name, time_obj, ephemeris="jpl")    

    # Convert astropy Quantity objects to meters and m/s, create state vector
    # State vector format: [x, y, z, vx, vy, vz]
    statevec = [ 
        pos.xyz[0].to("m").value,  # X position in meters
        pos.xyz[1].to("m").value,  # Y position in meters
        pos.xyz[2].to("m").value,  # Z position in meters
        vel.xyz[0].to("m/s").value,  # X velocity in m/s
        vel.xyz[1].to("m/s").value,  # Y velocity in m/s
        vel.xyz[2].to("m/s").value,  # Z velocity in m/s
    ]

    # Transform from J2000 equatorial coordinates to Ecliptic coordinates
    # J2000 = equatorial coordinate system (Earth's equator as reference)
    # Ecliptic = coordinate system based on Earth's orbital plane
    # Transformation matrix converts between these coordinate systems
    trans = sxform("J2000", "ECLIPJ2000", t.jd)  # t.jd = Julian Day number

    # Apply transformation matrix to state vector
    statevececl = mxvg(trans, statevec)
    
    # Extract transformed positions and velocities
    position = [statevececl[0], statevececl[1], statevececl[2]]
    velocity = [statevececl[3], statevececl[4], statevececl[5]]
    return position, velocity

# ============================================================================
# PLANETARY MASS CALCULATIONS
# ============================================================================
# Calculate planetary masses from GM (gravitational parameter) values
# GM = G * M, so M = GM / G
# Using poliastro constants which provide GM values for each planet
msun = (constants.GM_sun / G).value
mmercury = (constants.GM_mercury / G).value
mvenus = (constants.GM_venus / G).value
mearth = (constants.GM_earth / G).value 
mmars = (constants.GM_mars / G).value
mjupiter = (constants.GM_jupiter / G).value
msaturn = (constants.GM_saturn / G).value
muranus = (constants.GM_uranus / G).value
mneptune = (constants.GM_neptune / G).value
# Pluto mass calculation (commented out - can be enabled if needed)
#mpluto = (constants.GM_pluto / G).value 
# Alternative: use custom mass (e.g., for black hole simulation)
#mpluto = msun * 3000

# ============================================================================
# PLANETARY POSITION AND VELOCITY DATA RETRIEVAL
# ============================================================================
# Get initial positions and velocities for all planets from JPL ephemeris
# These represent the actual positions of planets at the reference time
position_sun, velocity_sun = get_data("sun", t)
position_mercury, velocity_mercury = get_data("mercury", t)
position_venus, velocity_venus = get_data("venus", t)
position_earth, velocity_earth = get_data("earth", t)
position_mars, velocity_mars = get_data("mars", t)
position_jupiter, velocity_jupiter = get_data("jupiter", t)
position_saturn, velocity_saturn = get_data("saturn", t)
position_uranus, velocity_uranus = get_data("uranus", t)
position_neptune, velocity_neptune = get_data("neptune", t)
# Pluto data (commented out - can be enabled if needed)
#position_pluto, velocity_pluto = get_data("pluto", t)

# ============================================================================
# PARTICLE OBJECT CREATION
# ============================================================================
# Create Particle objects for each celestial body with their initial conditions
# Each particle represents a planet/moon/star in the simulation

Sun = Particle(
        position=position_sun,
        velocity=velocity_sun,
        acceleration=None,  # Will be calculated from gravitational forces
        name="sun",
        mass=msun
)
          
Mercury = Particle(
        position=position_mercury,
        velocity=velocity_mercury,
        acceleration=None,
        name="Mercury",
        mass=mmercury
)

Venus = Particle(
        position=position_venus,
        velocity=velocity_venus,
        acceleration=None,
        name="Venus",
        mass=mvenus
)

Earth = Particle(
        position=position_earth,
        velocity=velocity_earth,
        acceleration=None,
        name="Earth",
        mass=mearth
)

Mars = Particle(
        position=position_mars,
        velocity=velocity_mars,
        acceleration=None,
        name="Mars",
        mass=mmars
)

Jupiter = Particle(
        position=position_jupiter,
        velocity=velocity_jupiter,
        acceleration=None,
        name="Jupiter",
        mass=mjupiter
)

Saturn = Particle(
        position=position_saturn,
        velocity=velocity_saturn,
        acceleration=None,
        name="Saturn",
        mass=msaturn
)

Uranus = Particle(
        position=position_uranus,
        velocity=velocity_uranus,
        acceleration=None,
        name="Uranus",
        mass=muranus
)

Neptune = Particle(
        position=position_neptune,
        velocity=velocity_neptune,
        acceleration=None,
        name="Neptune",
        mass=mneptune
)

# Pluto particle (commented out - can be enabled for special simulations)
# Useful for testing with custom masses (e.g., black hole simulation)
#Pluto_blackHole = Particle(
#        position=position_pluto,
#        velocity=velocity_pluto,
#        acceleration=None,
#        name="Pluto",
#        mass=mpluto
#)
def draw(history, particles):
    """
    Visualize simulation results using pygame from stored history.
    This function replays the simulation by animating through stored position data.
    
    Args:
        history: Dictionary containing position history for each particle
                 Format: {particle_name: {'x': [list], 'y': [list], 'z': [list]}}
        particles: List of Particle objects to visualize
    """
    # Initialize pygame
    pg.init()
    screen = pg.display.set_mode((1200, 800))
    pg.display.set_caption("N-Body Solar System Simulation - History")
    clock = pg.time.Clock()
    
    # Color mapping for planets (RGB tuples)
    # Each planet has a distinct color for easy identification
    colors = {
        'sun': (255, 255, 0),        # Yellow
        'Mercury': (169, 169, 169),   # Gray
        'Venus': (255, 165, 0),       # Orange
        'Earth': (0, 100, 200),       # Blue
        'Mars': (255, 0, 0),          # Red
        'Jupiter': (255, 200, 100),  # Light orange/yellow
        'Saturn': (255, 220, 150),   # Pale yellow
        'Uranus': (100, 200, 255),   # Light blue
        'Neptune': (0, 0, 255),       # Blue
        'pluto-blackHole': (255, 255, 255)  # White
    }
    
    # Animation control variables
    running = True
    frame = 0  # Current frame in animation
    max_frames = max([len(data['x']) for data in history.values()]) if history else 0
    
    # ========================================================================
    # ANIMATION LOOP
    # ========================================================================
    while running:
        # Handle user input (close window, ESC key)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
        
        screen.fill((0, 0, 0))  # Clear screen with black background
        
        # Calculate scale and offset dynamically each frame
        # This ensures the view zooms/centers appropriately as animation progresses
        all_x = []
        all_y = []
        # Collect all positions up to current frame
        for name, data in history.items():
            if len(data['x']) > 0:
                all_x.extend(data['x'][:frame+1])
                all_y.extend(data['y'][:frame+1])
        
        # Calculate center and scale based on visible data
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
        
        # Draw trajectories up to current frame (shows path traveled so far)
        for particle in particles:
            name = particle.name
            if name in history and len(history[name]['x']) > 0:
                color = colors.get(name, (255, 255, 255))
                points = []
                # Convert historical positions to screen coordinates
                for i in range(min(frame, len(history[name]['x']))):
                    x = history[name]['x'][i] - center_x
                    y = history[name]['y'][i] - center_y
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    points.append((screen_x, screen_y))
                # Draw trajectory line
                if len(points) > 1:
                    pg.draw.lines(screen, color, False, points, 1)
                
                # Draw current position as a circle
                if frame < len(history[name]['x']):
                    x = history[name]['x'][frame] - center_x
                    y = history[name]['y'][frame] - center_y
                    screen_x = int(600 + x * scale)
                    screen_y = int(400 + y * scale)
                    # Size based on mass (logarithmic scale)
                    size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    pg.draw.circle(screen, color, (screen_x, screen_y), size)
        
        # Display current frame number
        font = pg.font.Font(None, 36)
        frame_text = font.render(f"Frame: {frame}/{max_frames}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 10))
        
        # Update display
        pg.display.flip()
        clock.tick(2000)  # 2000 Frames per second (very fast animation)
        
        # Advance to next frame
        frame += 1
        if frame >= max_frames:
            frame = 0  # Loop animation when reaching the end
    
    pg.quit()

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# Run simulation if this file is executed directly (not imported as module)
if __name__ == "__main__":
    # Create list of all particles to simulate
    # Add or remove particles from this list to change what's simulated
    particles = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
    
    # Create simulation instance
    sim = Simulation()
    
    def run_simulation():
        """
        Callback function for the "Run Simulation" button.
        Retrieves user input from GUI and runs the simulation.
        """
        try:
            # Get values from GUI entries and convert to proper types
            # These are read when button is clicked, not at module load time
            my_time_step = float(Time_step.get())  # Convert string to float
            my_duration = float(Duration.get())    # Convert string to float
            my_method = optionmenu.get()           # Get selected method name
            
            print(f"Starting simulation: delta_T={my_time_step}, duration={my_duration}, method={my_method}")
            
            # Run simulation with user-specified parameters
            # visualize=True enables real-time pygame visualization
            history = sim.simulate(
                particles, 
                delta_T=my_time_step,      # Time step in seconds
                method=my_method,          # Integration method
                duration=my_duration,      # Total duration in seconds
                visualize=True             # Show pygame window
            )
            print("Simulation completed!")
        except ValueError as e:
            # Handle invalid input (non-numeric values)
            print(f"Error: Invalid input values. Please enter numbers. {e}")
        except Exception as e:
            # Handle any other errors during simulation
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
    
    # Create "Run Simulation" button and connect it to callback function
    Run = customtkinter.CTkButton(app, text="Run Simulation", command=run_simulation)
    Run.grid(row=0, column=6, padx=20, pady=20)
    
    # Start GUI event loop (blocks until window is closed)
    app.mainloop()
    
    #testing the simulation via angular momentum and convesrvation of energy 

