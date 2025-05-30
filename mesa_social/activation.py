from mesa import Agent
import numpy as np
import random

# Custom random activation scheduler
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        random.shuffle(self.agents)
        for agent in self.agents:
            agent.step()
