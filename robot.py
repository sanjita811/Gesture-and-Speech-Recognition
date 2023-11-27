
import gymnasium as gym

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.env = gym.make("Taxi-v3", render_mode="human")

        self.action_space = [0,1,2,3,4,5]
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
    def render(self):
        return self.env.render()


