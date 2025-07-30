
from mesa import Agent
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