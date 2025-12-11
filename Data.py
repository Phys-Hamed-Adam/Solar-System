from astropy.coordinates import get_body_barycentric_posvel  # Planetary positions and velocities
from astropy.time import Time  
from astropy.constants import G  # Gravitational constant G
from spiceypy import sxform, mxvg  # Coordinate transformations 
from poliastro import constants  # Planetary constants
from Particle import Particle

# Set reference time for planetary ephemeris data
t = Time("2019-11-27 17:00:00.0", scale="tdb")

def get_data(name, time_obj=None):
    """
    Retrieves position and velocity data for a celestial body from JPL ephemeris.
    
    """
    # Use default time if none provided
    if time_obj is None:
        time_obj = t
    
    # Get barycentric position and velocity from JPL ephemeris
    pos, vel = get_body_barycentric_posvel(name, time_obj, ephemeris="jpl")    

    # Convert astropy Quantity objects to meters and m/s, create state vector
    statevec = [ 
        pos.xyz[0].to("m").value, 
        pos.xyz[1].to("m").value, 
        pos.xyz[2].to("m").value,  
        vel.xyz[0].to("m/s").value,  
        vel.xyz[1].to("m/s").value,  
        vel.xyz[2].to("m/s").value,  
    ]

    # Transform from J2000 equatorial coordinates to Ecliptic coordinates
   
    trans = sxform("J2000", "ECLIPJ2000", t.jd)  # t.jd = Julian Day number

    # Apply transformation matrix to state vector
    statevececl = mxvg(trans, statevec)
    
    # Extract transformed positions and velocities
    position = [statevececl[0], statevececl[1], statevececl[2]]
    velocity = [statevececl[3], statevececl[4], statevececl[5]]
    return position, velocity

# Define celestial bodies with their properties
# Format: (ephemeris_name, display_name, GM_constant_name)
celestial_bodies = [
    ("sun", "sun", "GM_sun"),
    ("mercury", "Mercury", "GM_mercury"),
    ("venus", "Venus", "GM_venus"),
    ("earth", "Earth", "GM_earth"),
    ("mars", "Mars", "GM_mars"),
    ("jupiter", "Jupiter", "GM_jupiter"),
    ("saturn", "Saturn", "GM_saturn"),
    ("uranus", "Uranus", "GM_uranus"),
    ("neptune", "Neptune", "GM_neptune"),
]

# Calculate masses and get positions/velocities for all bodies
masses = {}
positions = {}
velocities = {}

for ephemeris_name, display_name, gm_name in celestial_bodies:
    # Calculate mass from GM constant
    gm_constant = getattr(constants, gm_name)
    masses[display_name] = (gm_constant / G).value
    
    # Get position and velocity from ephemeris
    pos, vel = get_data(ephemeris_name, t)
    positions[display_name] = pos
    velocities[display_name] = vel

# Create Particle objects for each celestial body
particles_dict = {}
for ephemeris_name, display_name, gm_name in celestial_bodies:
    particles_dict[display_name] = Particle(
        position=positions[display_name],
        velocity=velocities[display_name],
        acceleration=None,  # Will be calculated from gravitational forces
        name=display_name,
        mass=masses[display_name]
    )

# Create individual variables for backward compatibility
Sun = particles_dict["sun"]
Mercury = particles_dict["Mercury"]
Venus = particles_dict["Venus"]
Earth = particles_dict["Earth"]
Mars = particles_dict["Mars"]
Jupiter = particles_dict["Jupiter"]
Saturn = particles_dict["Saturn"]
Uranus = particles_dict["Uranus"]
Neptune = particles_dict["Neptune"]
