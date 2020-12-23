
import random
import os
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal
import torch.optim as optim
from lib.GridData import GridData
from lib.PandapowerTask import PandapowerTask



use_cuda = torch.cuda.is_available()
# print(use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

'''
Replay Buffer
'''
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    

'''
SAC
'''
class ValueNetwork(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class SoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # print(state.shape)
        # print(action.shape)
        x = torch.cat([state, action], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z      = normal.sample()
        action_mt = torch.tanh(z[0,0:6])
        action_sw = torch.sigmoid(z[0,6])
        # print(action)
        action_mt  = action_mt.detach().cpu().numpy()
        # print(action_mt)
        action_sw  = action_sw.detach().cpu().numpy()
        out_action_sw = round(action_sw*1191)
        if out_action_sw>1191:
            out_action_sw=1190
        # print(type(out_action_sw))
        action_out=np.append(action_mt,[out_action_sw],axis=0)
        return action_out
    
'''
Training
'''    
def soft_q_update(batch_size, gamma=0.99,mean_lambda=1e-3,std_lambda=1e-3,z_lambda=0.0,soft_tau=1e-2,):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        pass
    pass

'''
Plot
'''

def plot(time, rewards):
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.title('t= %s' % (time))
    plt.plot(rewards)
    plt.show()
    # plt.ion()
    # plt.plot()
    # plt.pause(10)
    # plt.close()
    pass
'''
Setup
'''
data= GridData()
ppenv = PandapowerTask()
ppenv.init_cache(data,env_mode=True,start=0)
ppenv.init_action_space()

action_dim = 7
state_dim  = 40
hidden_dim = 256

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

# 最大场景数
max_iteration  = 500
# 一个训练集最大步数
time_point   = 50
# 场景编号
i=0
# time 
step  = 0
rewards     = []
batch_size  = 10
#总步数计数器
total_step=0

while i < max_iteration:
    state = ppenv.reset()
    episode_reward = 0
    #实际进行步数的计数器
    j=0
    for time in range(time_point):
        ppenv.init_cache(data,env_mode=True)
        state=ppenv.env_state_cache
        action = policy_net.get_action(state)
        pp_action= ppenv.action_mapping(action)
        _, reward, next_state, done = ppenv.env_step(time,data,pp_action)
        
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            print("Step= %i,updating..."%(step))
            soft_q_update(batch_size)
        
        state = next_state
        episode_reward += reward
        total_step+=1
        j+=1
        if done:
            print("GAME OVER in step",step)
            break
    episode_reward=episode_reward/j    
    rewards.append(episode_reward)
    if i%100==0:
        plot(i, rewards)
    i += 1
    pass