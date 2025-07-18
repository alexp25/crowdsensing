import random  
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np


# Define scale: 1 grid unit = 10 meters
grid_scale = 10  # meters per unit
walking_speed = 83 / grid_scale  # Adjust speed for 10m/unit scale

def calculate_travel_time(poi1, poi2):
    """Calculate travel time between two POIs based on Euclidean distance."""
    distance = np.sqrt((poi1[0] - poi2[0]) ** 2 + (poi1[1] - poi2[1]) ** 2)
    return max(1, int(distance / walking_speed))  # Ensure at least 1-minute travel time

class TouristGroup(Agent):
    """A group of tourists traveling together (instead of individual agents)."""
    def __init__(self, unique_id, model, guided, start_time, guided_wait_time, self_guided_wait_time, group_size):
        super().__init__(model)
        self.unique_id = unique_id
        self.guided = guided  
        self.path = model.itinerary[:]  # Tour itinerary (list of POIs)
        self.current_step = 0
        self.remaining_time = np.random.randint(*guided_wait_time) if guided else np.random.randint(*self_guided_wait_time)
        self.travel_time = 0  
        self.traveling = False  
        self.skip_prob = 0.1 if not guided else 0 # Self-guided tourists have a chance to skip POIs
        self.start_time = start_time  
        self.started = False
        self.completed = False  
        self.group_size = group_size  # Number of people in this group

    def step(self):
        """Handles the movement and waiting of tourist groups between POIs."""
        if self.model.schedule_time < self.start_time:
            return  # Ensure tourists don't start before their scheduled time

        if self.completed:
            return  # Completed travelers should not be counted

        if self.traveling:
            self.travel_time -= 1
            if self.travel_time <= 0:
                self.traveling = False  # Stop traveling once time is up
            return  

        if self.remaining_time > 0:
            self.remaining_time -= 1  # Tourists are waiting at a POI
            return  

        if self.current_step >= len(self.path) - 1:  
            self.completed = True  # Tour is now fully completed
            return  

        # Move to the next POI and set travel time
        next_pos = self.path[self.current_step + 1]
        self.travel_time = calculate_travel_time(self.path[self.current_step], next_pos)  
        self.traveling = True  
        self.current_step += 1

        # Assign waiting time after reaching the POI
        if self.guided:
            self.remaining_time = np.random.randint(self.model.guided_wait_time[0], self.model.guided_wait_time[1])
        else:
            self.remaining_time = np.random.randint(self.model.self_guided_wait_time[0], self.model.self_guided_wait_time[1])
        
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
    fig.savefig(f'state_t{model.schedule_time}.png', dpi=fig.dpi)
    plt.show()


def run_once():
    # Simulation parameters

    # Define scale: 1 grid unit = 10 meters
    width = 100  # x10 meters wide
    height = 100  # x10 meters high

    num_tourists = 1000
    guided_ratio = 0.5  
    time_steps = 840  # 8 AM - 10 PM

    # Fixed Values
    num_pois = 8
    guided_start_times = [120, 360]  # 10 AM, 2 PM 
    self_guided_start_window = (0, 720)  # Self-guided tourists start between 8 AM and 8 PM
    self_guided_start_interval = 10  # New tourists start every 10-30 min within the window
    guided_wait_time = (10, 15)  
    self_guided_wait_time = (5, 10)  

    # Group Sizes (Now Configurable)
    guided_group_size = 20
    self_guided_group_size = 2

    np.random.seed(42)
    itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(num_pois)]
            
    # Run Simulation
    model = CityModel(width, height, num_tourists, guided_ratio, time_steps, guided_start_times, self_guided_start_window, self_guided_start_interval, guided_wait_time, self_guided_wait_time, guided_group_size, self_guided_group_size, itinerary)

    time_steps_2 = int(time_steps/2)
    
    for i in range(time_steps_2):
        model.step()        
        
    plot_state_at_time(model)
    
    for i in range(time_steps_2):
        model.step()

    # Collect Data
    data = model.datacollector.get_model_vars_dataframe()

    # Compute the mean congestion values
    avg_total_congestion = data["Total Congestion"].mean()
    avg_guided_congestion = data["Guided Congestion"].mean()
    avg_self_guided_congestion = data["Self-Guided Congestion"].mean()

    print(f"Average Total Congestion: {avg_total_congestion:.2f}")
    print(f"Average Guided Congestion: {avg_guided_congestion:.2f}")
    print(f"Average Self-Guided Congestion: {avg_self_guided_congestion:.2f}")   
        
    # Plots
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data["Guided Congestion"], c="orange", label="Guided Tourists")
    plt.plot(data["Self-Guided Congestion"], c="blue", label="Self-Guided Tourists")
    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Tourists at POIs")
    plt.title("Guided vs Self-Guided Congestion Over the Day")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(data["Guided Completion Rate"], c="orange", label="Guided Tourists")
    plt.plot(data["Self-Guided Completion Rate"], c="blue", label="Self-Guided Tourists")
    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Completion Rate (%)")
    plt.title("Guided vs Self-Guided Tour Completion Rate")
    plt.legend()

    plt.tight_layout()
    fig.savefig('eval.png', dpi=fig.dpi)
    plt.show()
    
    
def run_simulation(
    num_tourists,
    guided_group_size,
    self_guided_group_size,
    guided_wait_time,
    self_guided_wait_time):
    """Runs the simulation with given parameters and returns average congestion values."""
    
    width = 100  # x10 meters wide
    height = 100  # x10 meters high
    num_pois = 8
    time_steps = 840  # 8 AM - 10 PM
    
    np.random.seed(42)
    itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(num_pois)]
    
    model = CityModel(
        width=width, height=height, num_tourists=num_tourists, guided_ratio=0.5,
        total_time_steps=time_steps, guided_start_times=[120, 360],
        self_guided_start_window=(0, 720), self_guided_start_interval=10,
        guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
        guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
        itinerary=itinerary
    )

    for _ in range(time_steps):  # Simulate a full day
        model.step()

    # Extract congestion data
    data = model.datacollector.get_model_vars_dataframe()
    avg_total_congestion = data["Total Congestion"].mean()
    avg_guided_congestion = data["Guided Congestion"].mean()
    avg_self_guided_congestion = data["Self-Guided Congestion"].mean()

    return avg_total_congestion, avg_guided_congestion, avg_self_guided_congestion
    
    
def run_multi_eval():
    # Parameter ranges for experiments
    group_sizes = list(range(5, 51, 5))  # Test guided group sizes from 5 to 50
    num_tourists_range = list(range(200, 2001, 200))
    num_tourists = 1000  # Fix number of tourists

    # Store results
    total_congestion_results = []
    guided_congestion_results = []
    self_guided_congestion_results = []
    
    for num_tourists in num_tourists_range:
        print("running num tourists: " + str(num_tourists))
        avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, 25, 2, (10, 15), (5, 10))  # Keep self-guided groups small
        total_congestion_results.append(avg_total)
        guided_congestion_results.append(avg_guided)
        self_guided_congestion_results.append(avg_self_guided)
    plot_x = num_tourists_range
    plot_x_label = "Number of Tourists"
    

    # for size in group_sizes:
    #     print("running size: " + str(size))
    #     avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, size, size, (10, 15), (5, 10))  # Keep self-guided groups small
    #     total_congestion_results.append(avg_total)
    #     guided_congestion_results.append(avg_guided)
    #     self_guided_congestion_results.append(avg_self_guided)
    # plot_x = group_sizes
    # plot_x_label = "Group Size"
    
    # waiting_times = list(range(5, 51, 5))
    # for wtime in waiting_times:
    #     print("running wtime: " + str(wtime))
    #     avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, 25, 2, (wtime, wtime+1), (wtime, wtime+1))  # Keep self-guided groups small
    #     total_congestion_results.append(avg_total)
    #     guided_congestion_results.append(avg_guided)
    #     self_guided_congestion_results.append(avg_self_guided)
    # plot_x = waiting_times        
    # plot_x_label = "Waiting time at POIs"

    # Plot results
    fig = plt.figure(figsize=(10, 5))
    plt.plot(plot_x, guided_congestion_results, marker='o', c="orange", label="Guided Tour Congestion")
    plt.plot(plot_x, self_guided_congestion_results, marker='s', c="blue", label="Self-Guided Tour Congestion")
    # plt.plot(group_sizes, total_congestion_results, marker='x', label="Overall Tour Congestion")
    plt.xlabel(plot_x_label)
    plt.ylabel("Average Congestion")
    plt.title("Effect of " + plot_x_label + " on Congestion")
    plt.legend()
    plt.grid(True)
    fig.savefig('multi_eval.png', dpi=fig.dpi)
    plt.show()


run_once()
# run_multi_eval()