import torch

from rl_starter_files_master.student_utils.format import get_obss_preprocessor
from rl_starter_files_master.student_utils.storage import get_model_state, get_vocab
from .other import device
from rl_starter_files_master.model import ACModel
import numpy as np

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""
    torch.set_num_threads(1)
    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        assert use_memory == False
        #print('in this the problem area top')
        #torch.set_num_threads(1)
        #print('in this the problem area bottom')
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs


        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        
        self.acmodel.load_state_dict(get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

    def get_actions(self, obss):
        #print('len of obss', len(obss))
        #print('type of obss', type(obss))
        #print('obs keys', obss.keys())
        preprocessed_obss = self.preprocess_obss(obss, device=device)
        
        with torch.no_grad():
            
            if self.acmodel.recurrent:
                dist, value, self.memories = self.acmodel(preprocessed_obss, self.memories)
                #print('dist', dist, 'value', value)
            else:
                dist, value, _ = self.acmodel(preprocessed_obss)
                
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
            #print('actions', actions)
        else:
            actions = dist.sample()
            #print('actions', actions)
        #print(f'actions = {actions} values = {value.cpu().numpy()}')
        
        return actions.cpu().numpy(), value.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

    def return_weights(self): #needs more work
        total_weights = 0
        all_weights = []
        for param in self.acmodel.parameters():
            tot = 0
            shape_str = str(torch.Tensor.size(param))
            shape_int = torch.Tensor.size(param)

            try:
                shape0 = shape_int[0]
                tot = 1
            except:
                pass
            try:
                shape1 = shape_int[1]
                tot = 2
            except:
                pass

            try:
                shape2 = shape_int[2]
                tot =3
            except:
                pass

            try:
                shape3 = shape_int[3]
                tot=4
            except:
                pass

            if tot == 4:
                shape = shape0*shape1*shape2*shape3
            elif tot == 3:
                shape = shape0*shape1*shape2
            elif tot == 2:
                shape = shape0*shape1
            elif tot == 1:
                shape = shape0
            
            total_weights+=shape
            new_vector = param.reshape(shape,1)
            #print(f'before = {param} shape = {shape_str}, after = {new_vector}, shape = {np.shape(new_vector)}')
            array_v = new_vector.detach().numpy()
        
            #print('array v', array_v, np.shape(array_v))
            all_weights.append(array_v)
            #all_params.append(np.shape(param[0]))
        
        # print('param list',all_params)
        # #print('total weights', total_weights)
        
        s = np.shape(all_weights[0])
        s = s[0]
        cat_v = np.reshape(all_weights[0], (s,))
        
        for idx, vector in enumerate(all_weights[1:]):
            s = np.shape(vector)
            s = s[0]
            vector = np.reshape(vector, (s,))
         
            cat_v = list(cat_v) + list(vector)
            
    
        
        return cat_v
