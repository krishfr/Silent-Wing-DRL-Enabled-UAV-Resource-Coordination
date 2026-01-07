import numpy as np
import time
import random        
from uav import ChannelModel

class UAVEnv:
    def __init__(self, env_params, reward_params, channel_params):
        # Initialize environment, reward, and channel configurations
        self.env_params = env_params
        self.reward_params = reward_params
        self.channel_params = channel_params

        self.map_size = self.env_params['map_size']
        self.max_steps = self.env_params['max_steps']
        
        # Normalization factor used to scale coordinates between -1 and 1
        self.normalization_factor = self.env_params['map_size']  
        self.time_slot = self.env_params['time_slot']

        self.start = np.array(self.env_params['start_pos'])
        self.goal = np.array(self.env_params['goal_pos'])
        
        # Initialize ground communication nodes
        self.legitimate_nodes = [np.array(pos) for pos in self.env_params['legitimate_nodes']]
        
        # Procedurally generate wardens (detectors)
        self._generate_wardens()
        
        # Initial state variables
        self.current_step = 0
        self.state = self.start.copy()
        
        # Constant flight speed magnitude
        self.velocity_magnitude = self.env_params['velocity_magnitude']

        # Setup channel model with time-based seed for stochastic elements
        self.channel_model = ChannelModel(
            channel_params=self.channel_params, 
            env_params=self.env_params, 
            seed=int(time.time())
        )
        
        # Metrics for performance tracking
        self.dep_history = []
        self.transmission_rate_history = []
    
    def _generate_wardens(self):
        # Randomize the number of wardens based on the config range
        min_count, max_count = self.env_params['warden_count_range']
        self.warden_count = random.randint(min_count, max_count)
        
        # Randomize warden placement within the specified range
        min_pos, max_pos = self.env_params['warden_pos_range']
        self.wardens = []
        for _ in range(self.warden_count):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            self.wardens.append(np.array([x, y]))
        
        self.wardens = np.array(self.wardens)
        
    def _calculate_transmission_rates(self):
        # Calculate individual rates for each ground node
        rates_to_nodes = []
        
        for node_pos in self.legitimate_nodes:
            # R_l = log2(1 + SNR) calculation via channel model
            transmission_rate = self.channel_model.calculate_transmission_rate(
                self.state[0], self.state[1], node_pos[0], node_pos[1]
            )
            rates_to_nodes.append(transmission_rate)
        
        total_transmission_rate = sum(rates_to_nodes)
        return total_transmission_rate, rates_to_nodes

    def get_state(self):
        # Normalize coordinates to a range of [-1, 1] relative to the map center
        map_center = self.normalization_factor / 2.0
        pos_norm = (self.state - map_center) / (self.normalization_factor / 2.0)
        pos_norm = np.clip(pos_norm, -1.0, 1.0)
        
        return pos_norm.astype(np.float32)

    def _get_obs(self):
        return self.get_state()

    def reset(self):
        # Re-initialize wardens and reset episode counters/states
        self._generate_wardens()
        self.state = self.start.copy()
        self.current_step = 0
        self.dep_history = []
        self.transmission_rate_history = []
        
        # Compute initial Detection Error Probability (DEP)
        if len(self.wardens) > 0:
            initial_dep = self.channel_model.calculate_dep(
                self.state[0], self.state[1], 
                self.wardens[0][0], self.wardens[0][1]
            )
            self.dep_history.append(initial_dep)
        else:
            self.dep_history.append(0.0)

        self.prev_dist_to_dest = np.linalg.norm(self.state - self.goal)
        self.position_history = [] # For oscillation detection
        
        return self._get_obs()

    def step(self, action):
        reward = 0.0
        
        # Convert action vector to velocity direction; magnitude is constant
        action_norm = np.linalg.norm(action)
        if action_norm > 1e-8:
            velocity_direction = action / action_norm
        else:
            velocity_direction = np.array([1.0, 0.0]) # Default movement if zero action
            
        velocity = velocity_direction * self.velocity_magnitude
        
        # Calculate next position based on dynamics
        next_state = self.state + self.time_slot * velocity

        # Boundary collision check
        boundary_violation = False
        if np.any(next_state < 0.0) or np.any(next_state > self.env_params['map_size']):
            self.state = np.clip(next_state, 0.0, self.env_params['map_size'])
            boundary_violation = True
            reward += self.reward_params['boundary_penalty']
        else:
            self.state = next_state
        
        # Boundary penalty field (potential field)
        # Penalizes the agent as it gets close to the map edges
        boundary_distance = min(
            self.state[0],
            self.state[1],
            self.env_params['map_size'] - self.state[0],
            self.env_params['map_size'] - self.state[1]
        )
        
        boundary_field_penalty = 0.0
        boundary_penalty_threshold = self.reward_params.get('boundary_penalty_threshold', 40.0)
        if boundary_distance < boundary_penalty_threshold:
            # Quadratic penalty function
            boundary_penalty_strength = self.reward_params.get('boundary_penalty_strength', 0.01)
            boundary_field_penalty = -boundary_penalty_strength * (boundary_penalty_threshold - boundary_distance) ** 2
            reward += boundary_field_penalty

        self.current_step += 1

        # Calculate distances for reward components
        dist_to_dest = np.linalg.norm(self.state - self.goal)
        
        # Locate closest warden
        dist_to_warden = float('inf')
        closest_warden_idx = 0
        for i, warden in enumerate(self.wardens):
            dist = np.linalg.norm(self.state - warden)
            if dist < dist_to_warden:
                dist_to_warden = dist
                closest_warden_idx = i

        # Locate closest communication node
        distances_to_legit = [np.linalg.norm(self.state - node) for node in self.legitimate_nodes]
        min_dist_to_legit = min(distances_to_legit)
        closest_legit_idx = distances_to_legit.index(min_dist_to_legit)
        closest_legit_node = self.legitimate_nodes[closest_legit_idx]
        
        # Calculate DEP relative to the primary warden
        DEP = self.channel_model.calculate_dep(
            uav_x=self.state[0],
            uav_y=self.state[1],
            warden_x=self.wardens[0][0],
            warden_y=self.wardens[0][1]
        )
        
        # Calculate throughput
        total_transmission_rate, rates_to_nodes = self._calculate_transmission_rates()
        
        # Legacy tracking for transmission rate to the nearest node
        transmission_rate = self.channel_model.calculate_transmission_rate(
            uav_x=self.state[0],
            uav_y=self.state[1],
            ground_x=closest_legit_node[0],
            ground_y=closest_legit_node[1]
        )
        
        self.dep_history.append(DEP)
        self.transmission_rate_history.append(transmission_rate)
        
        # Main reward components
        transmission_reward = total_transmission_rate * self.reward_params['transmission_weight']
        base_penalty = self.reward_params['base_penalty']

        # Covertness penalty: triggers when DEP drops below the required threshold
        covertness_penalty = 0.0
        if DEP < self.reward_params['covertness_threshold']:
            covertness_penalty = self.reward_params['covert_penalty_weight'] * (self.reward_params['covertness_threshold'] - DEP)
        
        # Goal reaching reward logic
        goal_reward = 0.0
        goal_reached = False
        if dist_to_dest <= self.reward_params['goal_threshold']:
            goal_reward = self.reward_params['goal_reward']
            goal_reached = True
        
        # Distance guidance reward (Potential difference)
        distance_improvement_reward = 0.0
        if hasattr(self, 'prev_dist_to_dest'):
            distance_change = self.prev_dist_to_dest - dist_to_dest
            distance_improvement_reward = distance_change * self.reward_params.get('distance_guidance_weight', 0.1)
        
        self.prev_dist_to_dest = dist_to_dest
        
        # Oscillation penalty logic: penalizes repeating positions within a window
        oscillation_penalty = 0.0
        if not hasattr(self, 'position_history'):
            self.position_history = []
        
        self.position_history.append(self.state.copy())
        history_length = self.reward_params.get('oscillation_history_length', 20)
        if len(self.position_history) > history_length:
            self.position_history.pop(0)
        
        check_steps = self.reward_params.get('oscillation_check_steps', 10)
        distance_threshold = self.reward_params.get('oscillation_distance_threshold', 30.0)
        penalty_value = self.reward_params.get('oscillation_penalty', -50.0)
        
        if len(self.position_history) >= check_steps:
            current_pos = self.state
            recent_positions = self.position_history[-check_steps:]
            for past_pos in recent_positions[:-1]:
                if np.linalg.norm(current_pos - past_pos) < distance_threshold:
                    oscillation_penalty = penalty_value
                    break
        
        # Continuous proximity reward scaled by map size
        max_distance = np.sqrt(self.map_size**2 + self.map_size**2)
        normalized_distance = dist_to_dest / max_distance
        proximity_reward = (1.0 - normalized_distance) * self.reward_params.get('proximity_reward_weight', 0.05)
        
        # Step efficiency penalty to discourage long paths
        step_efficiency_penalty = -self.current_step * self.reward_params.get('step_penalty_weight', 0.01)
        
        # Aggregate guidance rewards
        total_goal_reward = goal_reward + distance_improvement_reward + proximity_reward + step_efficiency_penalty
        
        # Combine all sub-rewards into final scalar
        reward = (transmission_reward + base_penalty + covertness_penalty + 
                  total_goal_reward + boundary_field_penalty + oscillation_penalty)

        # Episode termination checks
        done = False
        termination_reason = "ongoing"
        if goal_reached:
            done = True
            termination_reason = "goal_reached"
        elif self.current_step >= self.max_steps:
            done = True
            termination_reason = "max_steps"

        # Construct comprehensive metadata dictionary
        info = {
            'DEP': DEP, 
            'throughput': total_transmission_rate, 
            'rates_to_nodes': rates_to_nodes,
            'dist_to_dest': dist_to_dest, 
            'dist_to_warden': dist_to_warden, 
            'velocity_magnitude': np.linalg.norm(velocity), 
            'step': self.current_step, 
            'covertness_violation': DEP < self.reward_params['covertness_threshold'],
            'goal_reached': goal_reached,  
            'termination_reason': termination_reason,   
            'boundary_distance': boundary_distance, 
            'boundary_violation': boundary_violation, 
            'transmission_reward': transmission_reward, 
            'base_penalty': base_penalty, 
            'covertness_penalty': covertness_penalty, 
            'goal_reward': goal_reward,  
            'distance_improvement_reward': distance_improvement_reward, 
            'proximity_reward': proximity_reward, 
            'step_efficiency_penalty': step_efficiency_penalty, 
            'total_goal_reward': total_goal_reward,  
            'constraint_penalty': boundary_field_penalty,     
            'oscillation_penalty': oscillation_penalty,   
        }
        
        return self._get_obs(), reward, done, info