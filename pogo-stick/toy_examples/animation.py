import matplotlib.animation as animation
import os
import matplotlib.pyplot as plt
import dynamics
import numpy as np

L0 = dynamics.L0

def animate(x_hist,
            save_path="animations/hopper_bounce.gif",
            fps=20):
    
    pos_hist = x_hist[0, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel("Height (m)")
    ax.set_title('Vertical Spring-Mass Hopper Simulation')
    ax.axhline(y=0, color='r', label='Floor')
    # ax.axhline(y=L0, color='grey', linestyle='--', label='Spring Rest Length')
    ax.grid(True,zorder=0)
    ax.legend()
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_xlim(-1,1)
    ax.set_ylim(0,max(pos_hist)+1)
    ax.set_aspect('equal')

    # Initialize the mass circle
    radius = 0.1
    mass_circle = plt.Circle((0, x_hist[0, 0]), radius, color='red')
    ax.add_patch(mass_circle)

    # Initialize the spring
    spring_line = plt.Line2D([0, 0], [0, x_hist[0, 0]], color='black', linewidth=4)
    ax.add_line(spring_line)

    # Add invisible elements for legend
    ax.plot([], [], 'ro', markersize=10, label='Mass')  # Invisible red circle for legend

    def animate(frame):
        # Update mass position
        x_current = x_hist[:, frame]
        mass_circle.center = (0, x_current[0])

        # Update spring
        compression = L0 - x_current[0]
        if compression > 0:
            spring_line.set_data([0, 0], [0, (x_current[0]-(compression/2))])
        else:
            spring_line.set_data([0, 0], [x_current[0]-L0, x_current[0]-radius])
        return [mass_circle, spring_line]


    # Create animation with every 5th frame
    frame_skip = 50
    frames = range(0, len(x_hist[0]), frame_skip)

    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=frames,
        interval=1000/fps,  # 50ms between frames = 20 FPS
        blit=True,
        repeat=True
    )

    # Create animations directory if it doesn't exist
    animation_dir = "animations"
    os.makedirs(animation_dir, exist_ok=True)

    # Save GIF to the animations directory
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    print("Animation saved!")

    plt.show()