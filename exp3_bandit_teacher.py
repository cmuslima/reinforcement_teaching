import numpy as np
import scipy
import math
class exp3_bandit():
    def __init__(self, num_arms, args):
        
        self.num_arms = num_arms
        self.arms_list = np.arange(0, self.num_arms)
        self.action_reward_vector = [0]*self.num_arms
        self.weight_vector = [1]*self.num_arms
        self.gamma = 0
        self.action_prob = [1]*self.num_arms

   
    def update_action_prob(self, reward, task_index, args):
        if args.student_type == 'DDPG':
            task_index-=2
    

        updated_reward = reward/self.action_prob[task_index]
        print(f'reward = {reward} updated reward = {updated_reward}')
        
        self.weight_vector[task_index] = self.weight_vector[task_index]*math.exp((self.gamma*updated_reward)/self.num_arms)
        print(f'self.weight vector = {self.weight_vector}')
        print('Finished updating the weight vector')
    def act(self, args):
        print(f'self.weight vector in the act function = {self.weight_vector}')
        for i in range(0, self.num_arms):
            weight_index_value = self.weight_vector[i]
            print(f'weight for index {i} = {weight_index_value}')
            print(f'proportion', weight_index_value/np.sum(self.weight_vector))
            self.action_prob[i] = (1-self.gamma)*(weight_index_value/np.sum(self.weight_vector)) + self.gamma/self.num_arms

        print(f'self.action_prob = {self.action_prob}')
        task_index = np.random.choice(self.arms_list, p=self.action_prob)
       
        if args.student_type == 'DDPG':
            task_index+=2
        return task_index