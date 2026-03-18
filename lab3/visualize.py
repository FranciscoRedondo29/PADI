import matplotlib.pyplot as plt

class GameVisualizer:
    def __init__(self, agent_name, fish_name):
        self.agent_name = agent_name
        self.fish_name = fish_name
        
        print(f"\n🎮 Visualizing: {self.agent_name}")
        print(f"   Target: {self.fish_name}")
        
        # Enable interactive mode for live updating
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.ax.set_ylim(500, 0)  # Invert Y axis
        self.ax.set_xlim(0, 200)
        self.ax.set_title(f"{self.agent_name} vs {self.fish_name}")
        
        # Create visual elements
        self.bar_rect = plt.Rectangle((70, 0), 60, 80, color='green', alpha=0.7)
        self.fish_circle = plt.Circle((100, 0), 10, color='red')
        
        self.ax.add_patch(self.bar_rect)
        self.ax.add_patch(self.fish_circle)
        
        self.text_status = self.ax.text(10, 20, "", fontsize=12, family='monospace')
        self.frame = 0

    def update(self, state, done):
        """
        Update the visualization with the current state.
        This must be called in a loop from the runner.
        """
        self.frame += 1
        
        self.bar_rect.set_y(state['bar_y'])
        self.fish_circle.center = (100, state['fish_y'])
        
        timer_color = 'green' 
        self.text_status.set_text(
            f"Bar Vel: {state['bar_vel']:+.2f}\n"
            f"Bar Y: {state['bar_y']:+.2f}\n"
            f"Fish Y: {state['fish_y']:+.2f}\n"
            f"Frame: {self.frame}"
        )
        self.text_status.set_color(timer_color)
        
        if done:
            result = "END"
            self.ax.text(100, 250, result, fontsize=36, ha='center', 
                   weight='bold', color=timer_color)
            
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Small pause so the UI updates
        plt.pause(0.001)

    def close(self):
        """
        Close the visualization window.
        """
        plt.ioff()
        plt.close(self.fig)

