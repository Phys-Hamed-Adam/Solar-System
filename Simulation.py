import customtkinter
from Particle import Particle
from Integration import IntegrationMethods
from tests import Tests
import numpy as np
import pygame as pg
from Data import Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
from plot import plot
# Setup the graphical user interface
app = customtkinter.CTk()  
app.title("Gravitational Simulation")

# Time step entry field 
Time_step = customtkinter.CTkEntry(app, placeholder_text="Time Step (seconds)")
Time_step.insert(0, "86400") # 1 hour
Time_step.grid(row=0, column=4, padx=20, pady=20)

# Duration entry field 
Duration = customtkinter.CTkEntry(app, placeholder_text="Duration (seconds)")
Duration.insert(0, "3066000000000")  #1000000 years
Duration.grid(row=0, column=5, padx=20, pady=20)

# Algorithm selection
optionmenu = customtkinter.CTkOptionMenu(app, values=["RK4", "Verlet", "Euler_Cromer", "Euler_Richardson"])
optionmenu.grid(row=1, column=5, padx=20, pady=20)


class Simulation:
    
    def simulate(self, particles, delta_T, method=None, duration=None, visualize=False, save_params=None):
        """
        Simulates particle motion over a given time period using a specified integration method.
        """
        
        # Initialize simulation time and history tracking
        time = 0
        history = {particle.name: {'x': [], 'y': [], 'z': [], 'vx': [], 'vy': [], 'vz': []} for particle in particles}
        
        # Find Uranus and Sun for orbit tracking
        uranus_particle = None
        sun_particle = None
        for particle in particles:
            if particle.name == "Uranus":
                uranus_particle = particle
            elif particle.name.lower() == "sun":
                sun_particle = particle
        
        # Track Uranus's orbital position for early save detection
        uranus_initial_position = None
        uranus_initial_distance = None
        uranus_has_passed_half_orbit = False
        save_triggered = False
        target_time_days = 120000  # Save at 120000 days
        target_time_seconds = target_time_days * 86400
        # Uranus's expected orbital period: ~30660 days (84 years)
        uranus_expected_period_days = 30660
        uranus_expected_period_seconds = uranus_expected_period_days * 86400
        uranus_min_time_for_orbit = uranus_expected_period_seconds * 0.85  # Require at least 85% of expected period
        
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
            all_positions = np.array([p.position for p in particles])
            center = np.mean(all_positions, axis=0)
            max_dist = np.max(np.linalg.norm(all_positions - center, axis=1))
            scale = 350 / max_dist if max_dist > 0 else 1e-10
            print(f"Visualization scale: {scale}, max_dist: {max_dist}, center: {center}")
            
        # Main simulation loop
        while (duration is None or time < duration) and not save_triggered:
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
                    break  # Exit simulation if user closed window
            
            # Update each particle
            for particle in particles:
                # Store history before updating
                history[particle.name]['x'].append(particle.position[0])
                history[particle.name]['y'].append(particle.position[1])
                history[particle.name]['z'].append(particle.position[2])
                history[particle.name]['vx'].append(particle.velocity[0])
                history[particle.name]['vy'].append(particle.velocity[1])
                history[particle.name]['vz'].append(particle.velocity[2])

                # Apply integration method
                if method == "Euler_Cromer":
                    IntegrationMethods.euler_cromer(particle, delta_T, particles)
                elif method == "RK4" or method == "Runge_Kutta":
                    IntegrationMethods.runge_kutta(particle, delta_T, particles)
                elif method == "Euler_Richardson":  
                    IntegrationMethods.euler_richardson(particle, delta_T, particles)
                elif method == "Verlet" or method == "verlet":
                    IntegrationMethods.verlet(particle, delta_T, particles)
            
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
                            # Only add points that are on screen
                            if 0 <= screen_x < 1200 and 0 <= screen_y < 800:
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
                    if particle.mass > 0:
                        size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    else:
                        size = 5
                    
                    # Only draw if on screen
                    if 0 <= screen_x < 1200 and 0 <= screen_y < 800:
                        pg.draw.circle(screen, color, (screen_x, screen_y), size)
                        # Draw particle name
                        font_small = pg.font.Font(None, 20)
                        name_text = font_small.render(particle.name, True, color)
                        screen.blit(name_text, (screen_x + size + 2, screen_y - 10))
                
                # Display time
                font = pg.font.Font(None, 36)
                time_text = font.render(f"Time: {time/86400:.2f} days", True, (255, 255, 255))
                screen.blit(time_text, (10, 10))
                
                pg.display.flip()
                clock.tick(60)
            
            time += delta_T
            
            # Track Uranus's orbital position to detect full orbit completion
            if uranus_particle and sun_particle and not save_triggered:
                r_vector = uranus_particle.position - sun_particle.position
                current_distance = np.linalg.norm(r_vector)
                
                if uranus_initial_position is None:
                    uranus_initial_position = r_vector.copy()
                    uranus_initial_distance = current_distance
                else:
                    if time >= uranus_expected_period_seconds * 0.5:
                        uranus_has_passed_half_orbit = True
                    
                    position_diff = np.linalg.norm(r_vector - uranus_initial_position)
                    distance_ratio = abs(current_distance - uranus_initial_distance) / uranus_initial_distance
                    
                    if (time >= uranus_min_time_for_orbit and uranus_has_passed_half_orbit and
                        distance_ratio < 0.05 and position_diff < current_distance * 0.1):
                        calculated_period_days = time / 86400
                        print(f"\nUranus completed a full orbit at {calculated_period_days:.2f} days!")
                        print(f"Expected period: ~{uranus_expected_period_days} days")
                        print(f"Period error: {abs(calculated_period_days - uranus_expected_period_days) / uranus_expected_period_days * 100:.2f}%")
                        save_triggered = True
            
            
            
            # Save data if trigger condition is met
            if save_triggered:
                print(f"\nSaving simulation data at {time/86400:.2f} days...")
                try:
                    Tests.save_test_results(particles, history, save_params['delta_T'], 
                                          time, save_params['method'])  # Use actual time instead of duration
                    print("Data saved successfully!")
                except Exception as e:
                    print(f"Error saving results: {e}")
                    import traceback
                    traceback.print_exc()
                # Break out of loop after saving
                break
        
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
                # Size based on mass (log scale) - check for valid mass
                if particle.mass > 0:
                    size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                else:
                    size = 5
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
    
def draw(history, particles):
    """
    Visualize simulation results using pygame from stored history.
    Replays the simulation by animating through stored position data.
    """
    # Initialize pygame
    pg.init()
    screen = pg.display.set_mode((1200, 800))
    pg.display.set_caption("N-Body Solar System Simulation - History")
    clock = pg.time.Clock()
    
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
        'pluto-blackHole': (255, 255, 255)  
    }
    
    running = True
    frame = 0  # Current frame in animation
    max_frames = max([len(data['x']) for data in history.values()]) if history else 0
    
    while running:
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
            scale = 400 / max_dist if max_dist > 0 else 1
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
                    # Size based on mass (logarithmic scale) - check for valid mass
                    if particle.mass > 0:
                        size = max(2, min(20, int(np.log10(particle.mass / 1e20) + 5)))
                    else:
                        size = 5
                    pg.draw.circle(screen, color, (screen_x, screen_y), size)
        
        # Display current frame number
        font = pg.font.Font(None, 36)
        frame_text = font.render(f"Frame: {frame}/{max_frames}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 10))
        
        # Update display
        pg.display.flip()
        clock.tick(2000)  
        
        # Advance to next frame
        frame += 1
        if frame >= max_frames:
            frame = 0  # Loop animation when reaching the end
    pg.quit()

# Run simulation if this file is executed directly
if __name__ == "__main__":
    
    particles = [Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
    for p in particles:
        p.acceleration = p.calculate_acceleration_from_particles(p.position, particles)
    
    # Create simulation instance
    sim = Simulation()
    
    def run_simulation():
        """
        Callback for the "Run Simulation" button.
        Gets user input from GUI and runs the simulation.
        """
        try:
            # Get values from GUI entries and convert to proper types
            my_time_step = float(Time_step.get())  
            my_duration = float(Duration.get())   
            my_method = optionmenu.get()           
            
            print(f"Starting simulation: delta_T={my_time_step}, duration={my_duration}, method={my_method}")
        
            history = sim.simulate(
                particles, 
                delta_T=my_time_step,      
                method=my_method,          # Integration method
                duration=my_duration,      
                visualize=True,            # Show pygame window
                save_params={              # Parameters for saving after pygame exits
                    'delta_T': my_time_step,
                    'duration': my_duration,
                    'method': my_method
                }
            )
            
            print("Simulation completed!")
            
            # Plot Jupiter's energy over time
            if history:
                print("\nPlotting Jupiter's energy over time...")
                try:
                    plot(particles, history, my_time_step)
                except Exception as e:
                    print(f"Error plotting Jupiter energy: {e}")
                    import traceback
                    traceback.print_exc()
        except ValueError as e:
            print(f"Error: Invalid input values. Please enter numbers. {e}")
        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
    
    
    Run = customtkinter.CTkButton(app, text="Run Simulation", command=run_simulation)
    Run.grid(row=0, column=6, padx=20, pady=20)
    
    # Start GUI event loop 
    app.mainloop()
