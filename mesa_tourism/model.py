import random  
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from agent import TouristGroup, calculate_travel_time

class CityModel(Model):
    """Represents the entire simulation environment"""
    def __init__(self, width, height, num_tourists, guided_ratio, total_time_steps, guided_start_times, self_guided_start_window, self_guided_start_interval, guided_wait_time, self_guided_wait_time, guided_group_size, self_guided_group_size, itinerary):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.num_tourists = num_tourists
        self.guided_ratio = guided_ratio
        self.total_time_steps = total_time_steps
        self.guided_wait_time = guided_wait_time  
        self.self_guided_wait_time = self_guided_wait_time  
        self.schedule_time = 0  # Time counter for simulation

        self.itinerary = itinerary # List of POIs (fixed itinerary)

        self.guided_groups = []
        self.self_guided_groups = []

        # Step 1: Allocate Travelers for Guided Tours
        max_guided_travelers = int(num_tourists * guided_ratio)  
        max_self_guided_travelers = num_tourists - max_guided_travelers  

        num_guided_groups = max_guided_travelers // guided_group_size  
        guided_travelers_assigned = 0  

        guided_start_time_index = 0  # Ensures guided groups start at different times
        for _ in range(num_guided_groups):
            if guided_travelers_assigned + guided_group_size > max_guided_travelers:
                break  # Prevent exceeding total travelers

            start_time = guided_start_times[guided_start_time_index]  
            guided_start_time_index = (guided_start_time_index + 1) % len(guided_start_times)  

            agent = TouristGroup(len(self.guided_groups), self, guided=True, start_time=start_time,
                                 guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time, 
                                 group_size=guided_group_size)
            self.grid.place_agent(agent, self.itinerary[0])
            self.guided_groups.append(agent)
            guided_travelers_assigned += guided_group_size

        # Step 2: Allocate Remaining Travelers for Self-Guided Tours
        num_self_guided_groups = max_self_guided_travelers // self_guided_group_size  
        self_guided_travelers_assigned = 0  

        for _ in range(num_self_guided_groups):
            start_time = np.random.randint(*self_guided_start_window)
            start_time = start_time - (start_time % self_guided_start_interval)  

            group_size = min(np.random.randint(2, self_guided_group_size + 1), max_self_guided_travelers - self_guided_travelers_assigned)
            if group_size <= 0:
                break  

            agent = TouristGroup(len(self.self_guided_groups), self, guided=False, start_time=start_time,
                                 guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time, 
                                 group_size=group_size)
            self.grid.place_agent(agent, random.choice(self.itinerary))
            self.self_guided_groups.append(agent)

            self_guided_travelers_assigned += group_size

        # Ensure that Total Assigned Travelers = num_tourists
        total_assigned = guided_travelers_assigned + self_guided_travelers_assigned
        # assert total_assigned == num_tourists, f"Mismatch in assigned travelers: {total_assigned} != {num_tourists}"
        if total_assigned != num_tourists:
            print(f"Mismatch in assigned travelers: {total_assigned} != {num_tourists}")

        self.datacollector = DataCollector(
            model_reporters={
                "Total Congestion": self.compute_total_congestion,
                "Guided Congestion": self.compute_guided_congestion,
                "Self-Guided Congestion": self.compute_self_guided_congestion,
                "Guided Completion Rate": self.compute_guided_completion_rate,
                "Self-Guided Completion Rate": self.compute_self_guided_completion_rate
            }
        )

    def compute_total_congestion(self):
        """Counts actual number of travelers at POIs (not moving and not completed)."""
        total_travelers_at_pois = sum(
            group.group_size for group in self.guided_groups + self.self_guided_groups 
            if not group.traveling and not group.completed and self.schedule_time >= group.start_time
        )
        return total_travelers_at_pois  # Now correctly tracks only started travelers


    def compute_guided_congestion(self):
        """Counts actual guided travelers at POIs (not moving, not completed, and already started)."""
        total_guided_at_pois = sum(
            group.group_size for group in self.guided_groups 
            if not group.traveling and not group.completed and self.schedule_time >= group.start_time
        )
        return total_guided_at_pois


    def compute_self_guided_congestion(self):
        """Counts actual self-guided travelers at POIs (not moving, not completed, and already started)."""
        total_self_guided_at_pois = sum(
            group.group_size for group in self.self_guided_groups 
            if not group.traveling and not group.completed and self.schedule_time >= group.start_time
        )
        return total_self_guided_at_pois
    
    def compute_guided_completion_rate(self):
        """Tracks the proportion of guided groups that finished."""
        completed_groups = [group for group in self.guided_groups if group.completed]
        return len(completed_groups) / len(self.guided_groups) if len(self.guided_groups) > 0 else 0

    def compute_self_guided_completion_rate(self):
        """Tracks the proportion of self-guided groups that finished."""
        completed_groups = [group for group in self.self_guided_groups if group.completed]
        return len(completed_groups) / len(self.self_guided_groups) if len(self.self_guided_groups) > 0 else 0

    def step(self):
        """Advances the simulation by one time step."""
        self.schedule_time += 1  
        for group in self.guided_groups + self.self_guided_groups:
            group.step()
        self.datacollector.collect(self)
        
        
        
def plot_state_at_time(model, title="Tourist Distribution at Time Step", timestep=None):
    """Visualizes the full simulation state including POIs, labeled paths, and current group positions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot POIs and annotate them
    for idx, (x, y) in enumerate(model.itinerary):
        ax.scatter(x, y, s=120, c='black', marker='X')
        ax.text(x + 1, y + 1, f'POI {idx + 1}', fontsize=9, color='black')

    # Draw arrows between POIs to show direction
    for i in range(len(model.itinerary) - 1):
        x1, y1 = model.itinerary[i]
        x2, y2 = model.itinerary[i + 1]
        ax.annotate("",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

    # Helper function to draw a tourist group
    def draw_group(x, y, group, color, label):
        ax.scatter(x, y, s=group.group_size * 20, c=color, alpha=0.6, label=label)

    # Prevent duplicate legend labels
    labels_drawn = {"Guided": False, "Self-Guided": False}

    # Guided tourists
    for group in model.guided_groups:
        if model.schedule_time >= group.start_time and not group.completed:
            if group.traveling and group.current_step < len(group.path):
                p1 = group.path[group.current_step - 1]
                p2 = group.path[group.current_step]
                ratio = 1 - group.travel_time / calculate_travel_time(p1, p2)
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
            else:
                x, y = group.path[group.current_step]
            draw_group(x, y, group, 'orange', 'Guided Group' if not labels_drawn["Guided"] else "")
            labels_drawn["Guided"] = True

    # Self-guided tourists
    for group in model.self_guided_groups:
        if model.schedule_time >= group.start_time and not group.completed:
            if group.traveling and group.current_step < len(group.path):
                p1 = group.path[group.current_step - 1]
                p2 = group.path[group.current_step]
                ratio = 1 - group.travel_time / calculate_travel_time(p1, p2)
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
            else:
                x, y = group.path[group.current_step]
            draw_group(x, y, group, 'blue', 'Self-Guided Group' if not labels_drawn["Self-Guided"] else "")
            labels_drawn["Self-Guided"] = True

    # Dummy scatter to force POI into legend
    ax.scatter([], [], s=100, c='black', marker='X', label='POI')
    ax.set_title(f"{title} (Minute {model.schedule_time})" if timestep is None else f"{title} (Minute {timestep})")
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_xlabel("Grid X (10m units)")
    ax.set_ylabel("Grid Y (10m units)")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f'output/state_t{model.schedule_time}.png', dpi=fig.dpi)
    plt.show()
