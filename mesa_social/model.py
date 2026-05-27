from agents import BeliefAgent
import numpy as np
import random


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

        # Numpy arrays for vectorized step computation
        self.beliefs = np.zeros(self.num_agents)
        self.confidences = np.zeros(self.num_agents)
        self.knowledge = np.zeros(self.num_agents)

        self.agents = []
        agent_types = []

        for i in range(self.num_agents):
            r = random.random()
            if r < self.malicious_ratio:
                a_type = "malicious"
            elif r < self.malicious_ratio + self.influencer_ratio:
                a_type = "influencer"
            else:
                a_type = "normal"
            agent_types.append(a_type)
            agent = BeliefAgent(i, self, a_type)
            self.agents.append(agent)

        self.normal_mask = np.array([t == "normal" for t in agent_types])

        self.data_belief = []
        self.data_knowledge = []

    def step(self):
        N = self.num_agents
        K = 5

        # Sample K neighbor indices for every agent at once (with replacement — fast)
        neighbor_idx = np.random.randint(0, N, size=(N, K))

        # Gather neighbor beliefs and confidences via fancy indexing
        neighbor_beliefs = self.beliefs[neighbor_idx]       # (N, K)
        neighbor_weights = self.confidences[neighbor_idx]   # (N, K)

        # Confidence-weighted average belief per agent
        weight_sum = neighbor_weights.sum(axis=1)
        weight_sum = np.where(weight_sum == 0, 1e-9, weight_sum)
        avg_belief = (neighbor_beliefs * neighbor_weights).sum(axis=1) / weight_sum

        # Belief update — normal agents only
        influence = self.influence_rate * (avg_belief - self.beliefs) * self.confidences
        new_beliefs = np.clip(self.beliefs + influence, 0.0, 1.0)
        self.beliefs[self.normal_mask] = new_beliefs[self.normal_mask]

        # Knowledge and confidence update — normal agents only
        humility = 1.0 - self.confidences
        delta_k = self.learning_rate * humility * (1.0 - self.knowledge)
        new_knowledge = np.clip(self.knowledge + delta_k, 0.0, 1.0)
        new_confidence = -4.0 * (new_knowledge - 0.3) ** 2 + 1.0
        self.knowledge[self.normal_mask] = new_knowledge[self.normal_mask]
        self.confidences[self.normal_mask] = new_confidence[self.normal_mask]

        self.data_belief.append(float(np.mean(self.beliefs)))
        if np.any(self.normal_mask):
            self.data_knowledge.append(float(np.mean(self.knowledge[self.normal_mask])))
        else:
            self.data_knowledge.append(0.0)
