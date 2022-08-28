import numpy as np
import scipy
class exp3_bandit():
    def __init__(self, num_arms, args):
        self.action_reward_vector = range(0,num_arms)
        self.num_arms = num_arms

    def update_action_reward_vector(self, reward, task_index):
        self.action_reward_vector[task_index]+= reward
    def act(self):
        action_prob = scipy.special.softmax(self.action_reward_vector)
        task_index = np.random.choice(range(0, self.num_arms), action_prob)
        return task_index