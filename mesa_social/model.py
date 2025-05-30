from mesa import Model
from mesa.space import _Grid
from mesa.datacollection import DataCollector
from agents import BeliefAgent
import numpy as np
import random
from activation import CustomRandomActivation

class BeliefModel:
    def __init__(self, width=10, height=10, dk_ratio=0.5, malicious_ratio=0.2, influencer_ratio=0.1,
                 influence_rate=0.1, learning_rate=0.05):
        self.width = width
        self.height = height
        self.num_agents = width * height
        self.dk_ratio = dk_ratio
        self.malicious_ratio = malicious_ratio
        self.influencer_ratio = influencer_ratio
        self.influence_rate = influence_rate
        self.learning_rate = learning_rate
        self.schedule = CustomRandomActivation(self)
        self.agents = []

        for i in range(self.num_agents):
            r = random.random()
            if r < self.malicious_ratio:
                a_type = "malicious"
            elif r < self.malicious_ratio + self.influencer_ratio:
                a_type = "influencer"
            else:
                a_type = "normal"
            agent = BeliefAgent(i, self, a_type)
            self.agents.append(agent)
            self.schedule.add(agent)

        self.data_belief = []
        self.data_knowledge = []

    def step(self):
        self.schedule.step()
        avg_belief = np.mean([a.belief for a in self.agents])
        avg_knowledge = np.mean([a.knowledge for a in self.agents if a.type == "normal"])
        self.data_belief.append(avg_belief)
        self.data_knowledge.append(avg_knowledge)