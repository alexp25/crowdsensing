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
    """Calculate travel time based on Euclidean distance."""
    distance = np.sqrt((poi1[0] - poi2[0]) ** 2 + (poi1[1] - poi2[1]) ** 2)
    return max(1, int(distance / walking_speed))  # Ensure at least 1-minute travel time

class TouristAgent(Agent):
    def __init__(self, unique_id, model, guided, start_time, wait_time_range):
        super().__init__(model)
        self.unique_id = unique_id
        self.guided = guided  
        self.path = model.itinerary[:]  
        self.current_step = 0
        self.remaining_time = np.random.randint(*wait_time_range) if not guided else wait_time_range[0]
        self.travel_time = 0  # Time remaining while traveling
        self.traveling = False  # Explicit travel state
        self.skip_prob = 0.1 if not guided else 0  
        self.explore_prob = 0.1 if not guided else 0  
        self.start_time = start_time  
        self.started = False  
        self.explored_alternative = False  
        self.completed = False  

    def step(self):
        """Tourists move between POIs considering travel time"""
        if self.model.schedule_time < self.start_time:  
            return  # This tourist hasn't started yet

        if not self.started:
            self.started = True  

        if self.completed:
            return  # Skip completed tourists

        if self.traveling:
            self.travel_time -= 1
            if self.travel_time <= 0:
                self.traveling = False  # Finished traveling, now at POI
            return  # Don't count as congestion until arrived

        if self.current_step < len(self.path):
            if self.remaining_time <= 0:
                if not self.guided and np.random.rand() < self.skip_prob:
                    max_jump = min(len(self.path) - self.current_step - 1, np.random.randint(1, 3))
                    self.current_step += max_jump  

                if self.current_step < len(self.path) - 1:
                    next_pos = self.path[self.current_step + 1]
                    self.travel_time = calculate_travel_time(self.path[self.current_step], next_pos)  # Assign travel time
                    self.traveling = True  # Enter travel state
                    self.current_step += 1
                    self.remaining_time = np.random.randint(*self.model.wait_time_range) if not self.guided else self.model.wait_time_range[0]
                else:
                    self.completed = True  # Mark as completed
            else:
                self.remaining_time -= 1

class CityModel(Model):
    def __init__(self, width, height, num_tourists, guided_ratio, total_time_steps, guided_start_times, self_guided_start_interval, self_guided_start_window, guided_wait_time, self_guided_wait_time):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.num_tourists = num_tourists
        self.guided_ratio = guided_ratio
        self.total_time_steps = total_time_steps
        self.wait_time_range = guided_wait_time  
        self.self_guided_wait_time = self_guided_wait_time
        self.self_guided_start_window = self_guided_start_window
        self.schedule_time = 0  

        # Fixed Number of POIs = 8
        np.random.seed(42)
        self.itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(8)]

        self.guided_tourists = []
        self.self_guided_tourists = []

        # Generate Guided Tour Groups Based on Scheduled Start Times
        for start_time in guided_start_times:
            for i in range(num_tourists // len(guided_start_times)):  
                agent = TouristAgent(len(self.guided_tourists), self, guided=True, start_time=start_time, wait_time_range=guided_wait_time)
                self.grid.place_agent(agent, self.itinerary[0])
                self.guided_tourists.append(agent)

        # Generate Self-Guided Tourists Randomly Throughout the Day
        for i in range(num_tourists // 2):
            # start_time = np.random.randint(0, total_time_steps, size=1)[0]  
            start_time = np.random.randint(*self_guided_start_window)
            # Apply interval effect: round start time to nearest interval
            start_time = start_time - (start_time % self_guided_start_interval)
    
            agent = TouristAgent(len(self.self_guided_tourists), self, guided=False, start_time=start_time, wait_time_range=self_guided_wait_time)
            self.grid.place_agent(agent, random.choice(self.itinerary))
            self.self_guided_tourists.append(agent)

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
        return sum(len([agent for agent in self.guided_tourists + self.self_guided_tourists if not agent.traveling and agent.current_step < len(self.itinerary)])
                   for loc in self.itinerary)

    def compute_guided_congestion(self):
        return sum(len([agent for agent in self.guided_tourists if not agent.traveling and agent.current_step < len(self.itinerary)])
                   for loc in self.itinerary)

    def compute_self_guided_congestion(self):
        return sum(len([agent for agent in self.self_guided_tourists if not agent.traveling and agent.current_step < len(self.itinerary)])
                   for loc in self.itinerary)

    def compute_guided_completion_rate(self):
        completed_agents = [agent for agent in self.guided_tourists if agent.completed]
        return len(completed_agents) / len(self.guided_tourists) if len(self.guided_tourists) > 0 else 0

    def compute_self_guided_completion_rate(self):
        completed_agents = [agent for agent in self.self_guided_tourists if agent.completed]
        return len(completed_agents) / len(self.self_guided_tourists) if len(self.self_guided_tourists) > 0 else 0

    def step(self):
        self.schedule_time += 1  
        for agent in self.guided_tourists + self.self_guided_tourists:
            agent.step()
        self.datacollector.collect(self)


# Simulation parameters

# Define scale: 1 grid unit = 10 meters
width = 200  # 2000 meters wide
height = 200  # 2000 meters high

num_tourists = 20
guided_ratio = 0.5  
time_steps = 840  # 8 AM - 10 PM

# Fixed Values
num_pois = 8
guided_start_times = [120, 360]  # 10 AM, 2 PM 
self_guided_start_window = (0, 720)  # Self-guided tourists start between 8 AM and 8 PM
self_guided_start_interval = 10  # New tourists start every 10-30 min within the window
guided_wait_time = (10, 15)  
self_guided_wait_time = (5, 10)  

# Run Simulation
model = CityModel(width, height, num_tourists, guided_ratio, time_steps, guided_start_times, self_guided_start_interval, self_guided_start_window, guided_wait_time, self_guided_wait_time)

for i in range(time_steps):
    model.step()

# Collect Data
data = model.datacollector.get_model_vars_dataframe()

# Plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
    
plt.plot(data["Guided Congestion"], c="orange", label="Guided Tourists")
plt.plot(data["Self-Guided Congestion"], c="blue", label="Self-Guided Tourists")
plt.xlabel("Time Step (Minutes)")
plt.ylabel("Tourists at POIs")
plt.title("Guided vs Self-Guided Congestion Over the Day")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data["Guided Completion Rate"], label="Guided Tourists")
plt.plot(data["Self-Guided Completion Rate"], label="Self-Guided Tourists")
plt.xlabel("Time Step (Minutes)")
plt.ylabel("Completion Rate (%)")
plt.title("Guided vs Self-Guided Tour Completion Rate")
plt.legend()

plt.tight_layout()
plt.show()
