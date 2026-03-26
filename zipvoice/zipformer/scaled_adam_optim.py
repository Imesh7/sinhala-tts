from torch.optim import Optimizer

class ScaledAdamOptimizer(Optimizer):
    def __init__(self, learning_rate, scale, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.scale = scale
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def step(self):
        super().step()