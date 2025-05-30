from mesa import Agent
import numpy as np
import random
from activation import CustomRandomActivation

# Agent class
class BeliefAgent:
    def __init__(self, unique_id, model, agent_type):
        self.unique_id = unique_id
        self.model = model
        self.type = agent_type
        self.set_initial_values()

    def set_initial_values(self):
        if self.type == "malicious":
            self.knowledge = 0.0
            self.confidence = 1.0
            self.belief = 1.0
        elif self.type == "influencer":
            self.knowledge = 1.0
            self.confidence = 0.5
            self.belief = 0.5
        else:
            if random.random() < self.model.dk_ratio:
                self.knowledge = random.uniform(0.0, 0.3)
            else:
                self.knowledge = random.uniform(0.3, 1.0)
            self.confidence = -4 * (self.knowledge - 0.3) ** 2 + 1
            self.belief = random.uniform(0.4, 0.6)

    def step(self):
        if self.type in ["malicious", "influencer"]:
            return

        neighbors = random.sample(self.model.agents, k=min(5, len(self.model.agents)))
        avg_belief = np.mean([a.belief for a in neighbors])
        influence = self.model.influence_rate * (avg_belief - self.belief) * self.confidence
        self.belief = np.clip(self.belief + influence, 0, 1)

        humility = 1.0 - self.confidence
        delta_k = self.model.learning_rate * humility * (1.0 - self.knowledge)
        self.knowledge = np.clip(self.knowledge + delta_k, 0, 1)
        self.confidence = -4 * (self.knowledge - 0.3) ** 2 + 1