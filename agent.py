import random
from collections import deque
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Actor network responsible for action selection
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Standard three-layer MLP for continuous action mapping
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh activation ensures the output stays within [-1, 1]
        return torch.tanh(self.fc3(x)) 


# Data structure to efficiently store and sample priorities
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # Binary tree represented as an array
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        # Update parent nodes with priority changes
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # Search the tree for the index corresponding to value 's'
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # The root node contains the sum of all priorities
        return self.tree[0]

    def add(self, p, data):
        # Insert new experience with priority 'p'
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        # Update priority and propagate the change up the tree
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        # Retrieve data based on a priority-weighted sample
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


# Experience Replay Buffer with Prioritization and Multi-step support
class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, n_step, gamma, alpha, beta, beta_increment):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

        # Standard single-step experience storage
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
        # N-step experience storage
        self.n_step_state = np.zeros((max_size, state_dim))
        self.n_step_action = np.zeros((max_size, action_dim))
        self.n_step_next_state = np.zeros((max_size, state_dim))
        self.n_step_reward = np.zeros((max_size, 1))
        self.n_step_done = np.zeros((max_size, 1))
        
        # Internal buffer for calculating n-step returns
        self.n_step_buffer = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, next_state, reward, done):
        # Buffer the current step for n-step calculation
        self.n_step_buffer.append((state, action, next_state, reward, done))
        
        # Store basic step data
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        # Process n-step returns if enough steps have passed
        if len(self.n_step_buffer) >= self.n_step:
            n_step_reward = 0
            n_step_done = False
            
            # Discounted sum of rewards over n steps
            for i in range(self.n_step):
                step_reward = self.n_step_buffer[i][3]
                step_done = self.n_step_buffer[i][4]
                n_step_reward += (self.gamma ** i) * step_reward
                if step_done:
                    n_step_done = True
                    break
            
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            if n_step_done:
                _, _, last_next_state, _, _ = self.n_step_buffer[i]
            else:
                _, _, last_next_state, _, _ = self.n_step_buffer[self.n_step-1]
            
            self.n_step_state[self.ptr] = first_state
            self.n_step_action[self.ptr] = first_action
            self.n_step_next_state[self.ptr] = last_next_state
            self.n_step_reward[self.ptr] = n_step_reward
            self.n_step_done[self.ptr] = n_step_done
        else:
            # Placeholder until n steps are reached
            self.n_step_state[self.ptr] = state
            self.n_step_action[self.ptr] = action
            self.n_step_next_state[self.ptr] = next_state
            self.n_step_reward[self.ptr] = reward
            self.n_step_done[self.ptr] = done
        
        # Use maximum priority for new experiences to ensure they are sampled
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        
        experience = (self.ptr, 
                     self.state[self.ptr], self.action[self.ptr], self.next_state[self.ptr], 
                     self.reward[self.ptr], self.done[self.ptr],
                     self.n_step_state[self.ptr], self.n_step_action[self.ptr], 
                     self.n_step_next_state[self.ptr], self.n_step_reward[self.ptr], 
                     self.n_step_done[self.ptr])
        
        self.tree.add(max_priority, experience)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        if len(self.n_step_buffer) > self.n_step:
            self.n_step_buffer.pop(0)
        
        if done:
            self._flush_n_step_buffer()
    
    def _flush_n_step_buffer(self):
        # Empty the n-step buffer at the end of an episode
        while len(self.n_step_buffer) > 1:
            self.n_step_buffer.pop(0)
            if len(self.n_step_buffer) == 0:
                break 

            n_step_reward = 0
            n_step_done = False
            for i, (_, _, _, r, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    n_step_done = True
                    break
            
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            if n_step_done:
                _, _, last_next_state, _, _ = self.n_step_buffer[i]
            else:
                _, _, last_next_state, _, _ = self.n_step_buffer[-1]
            
            buffer_len = len(self.n_step_buffer)
            target_ptr = (self.ptr - buffer_len) % self.max_size
            
            self.n_step_state[target_ptr] = first_state
            self.n_step_action[target_ptr] = first_action
            self.n_step_next_state[target_ptr] = last_next_state
            self.n_step_reward[target_ptr] = n_step_reward
            self.n_step_done[target_ptr] = n_step_done

            experience = (target_ptr,
                         self.state[target_ptr], self.action[target_ptr], self.next_state[target_ptr],
                         self.reward[target_ptr], self.done[target_ptr],
                         self.n_step_state[target_ptr], self.n_step_action[target_ptr],
                         self.n_step_next_state[target_ptr], self.n_step_reward[target_ptr],
                         self.n_step_done[target_ptr])

            tree_idx = target_ptr + self.tree.capacity - 1
            if tree_idx < len(self.tree.data):
                self.tree.data[tree_idx] = experience
        
        self.n_step_buffer.clear()

    def sample(self, batch_size):
        # Sample a batch based on priority segments
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        # Calculate Importance Sampling weights to correct for prioritization bias
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        states, actions, next_states, rewards, dones = [], [], [], [], []
        n_states, n_actions, n_next_states, n_rewards, n_dones = [], [], [], [], []
        
        for experience in batch:
            ptr, state, action, next_state, reward, done, n_state, n_action, n_next_state, n_reward, n_done = experience
            states.append(state); actions.append(action); next_states.append(next_state)
            rewards.append(reward); dones.append(done)
            n_states.append(n_state); n_actions.append(n_action); n_next_states.append(n_next_state)
            n_rewards.append(n_reward); n_dones.append(n_done)
        
        single_step = (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(rewards).reshape(-1, 1).to(self.device),
            torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        )
        
        multi_step = (
            torch.FloatTensor(n_states).to(self.device),
            torch.FloatTensor(n_actions).to(self.device),
            torch.FloatTensor(n_next_states).to(self.device),
            torch.FloatTensor(n_rewards).reshape(-1, 1).to(self.device),
            torch.FloatTensor(n_dones).reshape(-1, 1).to(self.device)
        )
        
        return single_step, multi_step, idxs, torch.FloatTensor(is_weights).to(self.device)

    def update_priorities(self, idxs, errors):
        # Update priorities based on recent TD errors
        for i, idx in enumerate(idxs):
            priority = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)


# Dual-headed Critic network for Clipped Double-Q learning
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Q1 network architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q1 = nn.Linear(256, 256)
        self.fc3_q1 = nn.Linear(256, 1)
        # Q2 network architecture (identical to Q1)
        self.fc1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q2 = nn.Linear(256, 256)
        self.fc3_q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Process Q1
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        # Process Q2
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        return q1, q2

    def Q1(self, state, action):
        # Used for policy optimization
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        return self.fc3_q1(q1)


# Twin Delayed Deep Deterministic Policy Gradient (TD3) Implementation
class TD3:
    def __init__(self, state_dim: int, action_dim: int, td3_params: dict):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.action_dim = action_dim

         # Network and target network initialization
         self.actor = Actor(state_dim, action_dim).to(self.device)
         self.critic1 = Critic(state_dim, action_dim).to(self.device)
         self.critic2 = Critic(state_dim, action_dim).to(self.device)
         
         self.actor_target = Actor(state_dim, action_dim).to(self.device)
         self.critic1_target = Critic(state_dim, action_dim).to(self.device)
         self.critic2_target = Critic(state_dim, action_dim).to(self.device)
         
         # Synchronize initial weights
         self.actor_target.load_state_dict(self.actor.state_dict())
         self.critic1_target.load_state_dict(self.critic1.state_dict())
         self.critic2_target.load_state_dict(self.critic2.state_dict())

         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=td3_params['lr_actor'])
         self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=td3_params['lr_critic'])
         self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=td3_params['lr_critic'])

         self.replay_buffer = PrioritizedReplayBuffer(
             state_dim=state_dim, action_dim=action_dim,
             max_size=td3_params['replay_buffer_size'], n_step=td3_params['n_step'],
             gamma=td3_params['multi_step_gamma'], alpha=td3_params['per_alpha'],
             beta=td3_params['per_beta'], beta_increment=td3_params['per_beta_increment']
         )

         self.batch_size = td3_params['batch_size']
         self.gamma = td3_params['gamma']
         self.tau = td3_params['tau']
         self.policy_noise = td3_params['policy_noise']
         self.noise_clip = td3_params['noise_clip']
         self.policy_freq = td3_params['policy_freq']
         self.max_action = td3_params['max_action']

    def select_action(self, state, noise: float = 0.1):
        # Generate action from actor and add Gaussian exploration noise
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1.0, 1.0)
        return action

    def store_transition(self, s, a, r, s_, d):
        self.replay_buffer.add(s, a, s_, r, d)

    def update(self, total_steps: int):
        if self.replay_buffer.size < self.batch_size:
            return
        
        # Priority sampling for mixed single and multi-step learning
        single_step, multi_step, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = single_step
        n_states, n_actions, n_next_states, n_rewards, n_dones = multi_step

        # Target Policy Smoothing
        noise = torch.clamp(torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip).to(self.device)
        next_actions = torch.clamp(self.actor_target(next_states) + noise, -self.max_action, self.max_action)

        # Single-step TD Target
        target_q1, target_q2 = self.critic1_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        single_step_target = rewards + (1 - dones) * self.gamma * target_q

        # Multi-step TD Target
        n_target_actions = torch.clamp(self.actor_target(n_next_states) + noise, -self.max_action, self.max_action)
        n_target_q1, n_target_q2 = self.critic1_target(n_next_states, n_target_actions)
        n_target_q = torch.min(n_target_q1, n_target_q2)
        multi_step_target = n_rewards + (1 - n_dones) * (self.gamma ** self.replay_buffer.n_step) * n_target_q

        # Q-Network training with mixed loss
        current_q1, current_q2 = self.critic1(states, actions)
        n_current_q1, _ = self.critic1(n_states, n_actions)
        _, n_current_q2 = self.critic2(n_states, n_actions)

        # Calculate hybrid TD error for priority updates
        single_td_err = torch.max(torch.abs(current_q1 - single_step_target.detach()), torch.abs(current_q2 - single_step_target.detach()))
        multi_td_err = torch.max(torch.abs(n_current_q1 - multi_step_target.detach()), torch.abs(n_current_q2 - multi_step_target.detach()))
        td_errors = (0.5 * single_td_err + 0.5 * multi_td_err).cpu().data.numpy().flatten()

        # Critic update based on weighted MSE loss
        c1_loss = (0.5 * F.mse_loss(current_q1, single_step_target.detach(), reduction='none') + 
                   0.5 * F.mse_loss(n_current_q1, multi_step_target.detach(), reduction='none')) * is_weights.unsqueeze(1)
        c2_loss = (0.5 * F.mse_loss(current_q2, single_step_target.detach(), reduction='none') + 
                   0.5 * F.mse_loss(n_current_q2, multi_step_target.detach(), reduction='none')) * is_weights.unsqueeze(1)
        
        self.critic1_optimizer.zero_grad(); c1_loss.mean().backward(); self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(); c2_loss.mean().backward(); self.critic2_optimizer.step()
        
        self.replay_buffer.update_priorities(idxs, td_errors)

        # Delayed Policy Update
        if total_steps % self.policy_freq == 0:
            actor_loss = -self.critic1.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self._soft_update(self.critic2, self.critic2_target), self.critic2_target)

    def _soft_update(self, net, target_net):
        # Target network interpolation
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return net