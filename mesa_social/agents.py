import random


class BeliefAgent:
    def __init__(self, unique_id, model, agent_type):
        self.unique_id = unique_id
        self.model = model
        self.type = agent_type
        self._init_state()

    def _init_state(self):
        i = self.unique_id
        if self.type == "malicious":
            self.model.knowledge[i] = 0.0
            self.model.confidences[i] = 1.0
            self.model.beliefs[i] = 1.0
        elif self.type == "influencer":
            self.model.knowledge[i] = 1.0
            self.model.confidences[i] = 0.5
            self.model.beliefs[i] = 0.5
        else:
            k = random.uniform(0.0, 0.3) if random.random() < self.model.dk_ratio else random.uniform(0.3, 1.0)
            self.model.knowledge[i] = k
            self.model.confidences[i] = -4 * (k - 0.3) ** 2 + 1
            self.model.beliefs[i] = random.uniform(0.4, 0.6)

    @property
    def belief(self):
        return self.model.beliefs[self.unique_id]

    @property
    def confidence(self):
        return self.model.confidences[self.unique_id]

    @property
    def knowledge(self):
        return self.model.knowledge[self.unique_id]
