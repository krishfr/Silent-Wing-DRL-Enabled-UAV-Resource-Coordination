import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from typing import List, Tuple, Optional, Dict, Any


VISUALIZATION_CONFIG = {
    'building_colors': ['#FFFF99', '#CCFF66', '#99FF33', '#66FF00', '#33CC00', '#009900'],
    'reward_colors': ['#FF0000', '#FF6666', '#FFCCCC', '#FFFFFF', '#CCCCFF', '#6666FF', '#0000FF'],
    'building_edge_color': 'black',
    'building_edge_width': 0.8,
    
    # Transparency settings
    'alpha_transparent': 0.1,
    'alpha_semi': 0.6,
    'alpha_opaque': 0.8,
    
    # Font settings
    'font_small': 7,
    'font_medium': 8,
    'font_normal': 10,
    'font_large': 12,
    'font_title': 14,
    
    # Grid and figure settings
    'grid_size': 100,
    'colormap_bins': 256,
    'grid_alpha': 0.3,
    'figure_size': (15, 10),
    'dpi_high': 300,
    
    # Colorbar settings
    'colorbar_width': 0.02,
    'colorbar_height': 0.45,
    'colorbar_y_start': 0.5,
    
    'building_colorbar_x': 0.815,
    'reward_colorbar_x': 0.90,
    
    # Trajectory settings
    'trajectory_limit': 10,
    'save_frequency': 50,
    'legend_bbox': (1.024, 0.0),
    
    # Performance metric colors
    'dep_color': 'blue',
    'throughput_color': 'green',
    'reward_color': 'purple',
    'moving_avg_color': 'red',
    'moving_avg_window': 20, 
}


class EnvironmentRenderer:
    def __init__(self, config: Dict[str, Any], map_size: float):
        self.config = config
        self.map_size = map_size
        self.reward_params = config.get('REWARD', {})
        
    def create_colorbar(self, fig, ax, cmap, norm, x_position, label, ticks=None):
        # Setup specific axes for the colorbar placement
        cax = fig.add_axes([
            x_position, 
            VISUALIZATION_CONFIG['colorbar_y_start'], 
            VISUALIZATION_CONFIG['colorbar_width'], 
            VISUALIZATION_CONFIG['colorbar_height']
        ])
        
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label(label, fontsize=VISUALIZATION_CONFIG['font_medium'], 
                    labelpad=2, rotation=90, va='top')
        cb.ax.tick_params(labelsize=VISUALIZATION_CONFIG['font_medium'])
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')
        
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels([
                f'{t:.0f}' if t == 0 else
                f'{t:.1f}' if isinstance(t, (int, float)) and t.is_integer() 
                else f'{t:.2f}' for t in ticks
            ])
        
        return cb
    
    def draw_buildings(self, fig, ax, env):
        if not (hasattr(env, 'channel_model') and hasattr(env.channel_model, 'buildings')):
            return
            
        buildings = env.channel_model.buildings
        if not buildings:
            return
            
        # Get building height range for normalization
        heights = [b.building_height for b in buildings]
        min_height, max_height = min(heights), max(heights)
        
        # Create height-based colormap
        cmap = LinearSegmentedColormap.from_list('height_map', 
                                               VISUALIZATION_CONFIG['building_colors'], 
                                               N=VISUALIZATION_CONFIG['colormap_bins'])
        
        # Draw individual building rectangles
        for building in buildings:
            if max_height > min_height:
                norm_height = (building.building_height - min_height) / (max_height - min_height)
            else:
                norm_height = 0.5
            
            color = cmap(norm_height)
            
            rect = patches.Rectangle(
                (building.x_min, building.y_min),
                building.width, building.height,
                linewidth=VISUALIZATION_CONFIG['building_edge_width'], 
                edgecolor=VISUALIZATION_CONFIG['building_edge_color'], 
                facecolor=color, 
                alpha=VISUALIZATION_CONFIG['alpha_opaque'],
                zorder=10
            )
            ax.add_patch(rect)
            
            # Add height labels for large buildings
            if building.width > 40 and building.height > 40:
                ax.text(building.x, building.y, f'{building.building_height:.0f}m',
                       ha='center', va='center', 
                       fontsize=VISUALIZATION_CONFIG['font_small'], 
                       color='black', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                       zorder=11)
        
        # Create colorbar for building heights
        norm = Normalize(vmin=min_height, vmax=max_height)
        ticks = np.linspace(min_height, max_height, 5)
        self.create_colorbar(fig, ax, cmap, norm, VISUALIZATION_CONFIG['building_colorbar_x'], 
                           'Building Height (m)', ticks)
    
    def draw_reward_field(self, fig, ax, env):
        grid_size = VISUALIZATION_CONFIG['grid_size']
        cell_size = self.map_size / grid_size
        
        # Generate grid centers
        x_centers = np.linspace(cell_size/2, self.map_size - cell_size/2, grid_size)
        y_centers = np.linspace(cell_size/2, self.map_size - cell_size/2, grid_size)
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # Compute rewards across the grid
        reward_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                x, y = X[i, j], Y[i, j]
                reward_grid[i, j] = self._calculate_position_reward(x, y, env)
        
        # Define colormap for the reward heat map
        cmap = LinearSegmentedColormap.from_list('reward_field', 
                                               VISUALIZATION_CONFIG['reward_colors'], 
                                               N=VISUALIZATION_CONFIG['colormap_bins'])
        
        # Normalize color range around zero if applicable
        vmin, vmax = np.min(reward_grid), np.max(reward_grid)
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        
        # Display the reward field as an image
        im = ax.imshow(reward_grid, extent=[0, self.map_size, 0, self.map_size], 
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, 
                      alpha=VISUALIZATION_CONFIG['alpha_semi'], zorder=1)
        
        # Create colorbar for reward values
        norm = Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, 7)
        self.create_colorbar(fig, ax, cmap, norm, VISUALIZATION_CONFIG['reward_colorbar_x'], 
                           'Reward Value', ticks)
        
        return im
    
    def _calculate_position_reward(self, x, y, env):
        position = np.array([x, y])
        total_reward = 0.0
        
        # Boundary penalty calculation
        boundary_distance = min(x, y, self.map_size - x, self.map_size - y)
        boundary_penalty_threshold = self.reward_params.get('boundary_penalty_threshold', 40.0)
        if boundary_distance < boundary_penalty_threshold:
            boundary_penalty_strength = self.reward_params.get('boundary_penalty_strength', 0.01)
            boundary_field_penalty = -boundary_penalty_strength * (boundary_penalty_threshold - boundary_distance) ** 2
            total_reward += boundary_field_penalty
        
        # Covertness/DEP penalty calculation
        if hasattr(env, 'wardens') and len(env.wardens) > 0 and hasattr(env, 'channel_model'):
            min_dep = float('inf')
            for warden in env.wardens:
                dep = env.channel_model.calculate_dep(
                    uav_x=x,
                    uav_y=y,
                    warden_x=warden[0],
                    warden_y=warden[1]
                )
                min_dep = min(min_dep, dep)
            
            # Penalize if below covertness threshold
            covertness_threshold = self.reward_params.get('covertness_threshold', 0.5)
            if min_dep < covertness_threshold:
                covert_penalty_weight = self.reward_params.get('covert_penalty_weight', -1000.0)
                covertness_penalty = covert_penalty_weight * (covertness_threshold - min_dep)
                total_reward += covertness_penalty

        # Proximity to goal reward
        if hasattr(env, 'goal'):
            dist_to_goal = np.linalg.norm(position - env.goal)
            max_distance = np.sqrt(self.map_size**2 + self.map_size**2)
            normalized_distance = dist_to_goal / max_distance
            proximity_reward_weight = self.reward_params.get('proximity_reward_weight', 0.05)
            proximity_reward = (1.0 - normalized_distance) * proximity_reward_weight
            total_reward += proximity_reward
        
        # Transmission rate reward
        if hasattr(env, 'legitimate_nodes') and len(env.legitimate_nodes) > 0 and hasattr(env, 'channel_model'):
            total_transmission_rate = 0.0
            for legit_node in env.legitimate_nodes:
                rate = env.channel_model.calculate_transmission_rate(
                    uav_x=x,
                    uav_y=y,
                    ground_x=legit_node[0],
                    ground_y=legit_node[1]
                )
                total_transmission_rate += rate
            
            transmission_weight = self.reward_params.get('transmission_weight', 1.0)
            transmission_reward = total_transmission_rate * transmission_weight
            total_reward += transmission_reward
        
        # Static step penalty
        base_penalty = self.reward_params.get('base_penalty', -1.0)
        total_reward += base_penalty
        
        return total_reward
    
    def draw_covert_holes(self, ax, env):
        if not (hasattr(env, 'wardens') and len(env.wardens) > 0 and 
                hasattr(env, 'channel_model')):
            return
        
        for warden_idx, (warden_x, warden_y) in enumerate(env.wardens):
            # Calculate ray endpoints using method from channel model
            ray_endpoints = env.channel_model.calculate_ray_endpoints(
                warden_x, warden_y, self.map_size)
            
            # Draw individual rays
            for end_x, end_y in ray_endpoints:
                ax.plot([warden_x, end_x], [warden_y, end_y], 
                       color='gray', linewidth=0.8, linestyle='-', alpha=0.05, zorder=15)
            
            # Draw the boundary outline of the covert hole
            if ray_endpoints:
                ray_endpoints.append(ray_endpoints[0])
                xs, ys = zip(*ray_endpoints)
                ax.plot(xs, ys, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, 
                       label='Covert Hole' if warden_idx == 0 else "", zorder=20)
    
    def draw_environment_markers(self, ax, env):
        # Mark start and goal positions
        if hasattr(env, 'start') and hasattr(env, 'goal'):
            ax.scatter(*env.start, c="green", s=100, marker="s", 
                      label="Start", zorder=100)
            ax.scatter(*env.goal, c="red", s=100, marker="s", 
                      label="Goal", zorder=100)
        
        # Mark warden positions
        if hasattr(env, 'wardens') and len(env.wardens) > 0:
            warden_x = [warden[0] for warden in env.wardens]
            warden_y = [warden[1] for warden in env.wardens]
            
            ax.scatter(warden_x, warden_y, c="black", s=100, marker="^", 
                      label=f"Wardens ({len(env.wardens)})", zorder=100)
            
            for i, warden in enumerate(env.wardens):
                ax.annotate(f'W{i+1}', (warden[0], warden[1]), xytext=(5, 5), 
                           textcoords='offset points', 
                           fontsize=VISUALIZATION_CONFIG['font_small'], 
                           color='white', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                   alpha=VISUALIZATION_CONFIG['alpha_semi']), zorder=100)
        
        # Mark legitimate node positions
        if hasattr(env, 'legitimate_nodes') and len(env.legitimate_nodes) > 0:
            legit_x = [node[0] for node in env.legitimate_nodes]
            legit_y = [node[1] for node in env.legitimate_nodes]
            
            ax.scatter(legit_x, legit_y, c="orange", s=80, marker="o", 
                      label=f"Nodes ({len(env.legitimate_nodes)})", zorder=100)


class UnifiedVisualizer:
    
    # Class variable to track if initial environment has been saved
    _initial_env_saved = False
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "figures"):
        self.config = config
        self.output_dir = output_dir
        self.env_params = config.get('ENV', {})
        self.map_size = self.env_params.get('map_size', 1000.0)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.all_trajectories = []
        self.recent_trajectories = []
        self.step_count = 0
        
        self.fig = None
        self.ax = None
        
        self.env_renderer = EnvironmentRenderer(config, self.map_size)
    
    def initialize_plot(self, env):
        self._setup_plot(env)
        
        # Save initial environment map if it doesn't exist
        initial_env_path = os.path.join(self.output_dir, "initial_env.png")
        if not os.path.exists(initial_env_path):
            plt.tight_layout()
            plt.savefig(initial_env_path, dpi=150, bbox_inches='tight')
    
    def _setup_plot(self, env):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Render static environment components
        self.env_renderer.draw_buildings(self.fig, self.ax, env)
        self.env_renderer.draw_reward_field(self.fig, self.ax, env)
        self.env_renderer.draw_covert_holes(self.ax, env)
        self.env_renderer.draw_environment_markers(self.ax, env)
        
        # Plot styling
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_xlabel("X Position (m)", fontsize=VISUALIZATION_CONFIG['font_large'])
        self.ax.set_ylabel("Y Position (m)", fontsize=VISUALIZATION_CONFIG['font_large'])
        self.ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        self.ax.set_title("Initial Environment", fontsize=VISUALIZATION_CONFIG['font_title'])
        
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
    
    def update_trajectory(self, trajectory: np.ndarray, reward: float, episode):
        self.all_trajectories.append((trajectory.copy(), reward))
        self.recent_trajectories.append((trajectory.copy(), reward))
        
        # Maintain a sliding window of recent trajectories
        if len(self.recent_trajectories) > VISUALIZATION_CONFIG['trajectory_limit']:
            self.recent_trajectories.pop(0)
            
        self.step_count += 1
        
        # Trigger periodic saving of trajectory images
        if self.step_count % VISUALIZATION_CONFIG['save_frequency'] == 0:
            self._save_trajectory_image(episode)
    
    def _save_trajectory_image(self, episode):
        if self.fig is None or self.ax is None:
            return
            
        # Clear existing trajectory lines
        for line in self.ax.lines[:]:
            if hasattr(line, '_trajectory_line'):
                line.remove()
                
        # Draw historical trajectories with high transparency
        other_trajs_labeled = False
        for trajectory, _ in self.all_trajectories[:-1]:
            if len(trajectory) > 1:
                label = "Other Trajs" if not other_trajs_labeled else ""
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                   color='black', 
                                   alpha=VISUALIZATION_CONFIG['alpha_transparent'], 
                                   linewidth=1, label=label, zorder=30)
                line._trajectory_line = True
                other_trajs_labeled = True
                    
        # Highlight the best trajectory from the recent window
        if self.recent_trajectories:
            best_trajectory = None
            best_reward = float('-inf')
            for trajectory, reward in self.recent_trajectories:
                if reward > best_reward:
                    best_reward = reward
                    best_trajectory = trajectory
            
            if best_trajectory is not None and len(best_trajectory) > 1:
                line, = self.ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 
                                   color='blue', linewidth=3, 
                                   label='Best Recent Traj', zorder=50)
                line._trajectory_line = True
                
        # Refresh legend and title
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
        
        self.ax.set_title(f"Training Trajectory - Ep {episode}", 
                         fontsize=VISUALIZATION_CONFIG['font_title'])
        filename = f"train_traj_ep_{episode}.png"
        
        plt.savefig(os.path.join(self.output_dir, filename), 
                    dpi=150, bbox_inches='tight')
    
    def _save_test_trajectory_image(self, num_tests):
        if self.fig is None or self.ax is None:
            return
            
        # Clear previous trajectory lines
        for line in self.ax.lines[:]:
            if hasattr(line, '_trajectory_line'):
                line.remove()
                
        if not self.all_trajectories:
            return
            
        # Find best performing test trajectory
        best_trajectory = None
        best_reward = float('-inf')
        for trajectory, reward in self.all_trajectories:
            if reward > best_reward:
                best_reward = reward
                best_trajectory = trajectory
        
        # Plot background test trajectories
        other_trajs_labeled = False
        for trajectory, reward in self.all_trajectories:
            if reward != best_reward and len(trajectory) > 1:
                label = "Other Trajs" if not other_trajs_labeled else ""
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                   color='gray', 
                                   alpha=VISUALIZATION_CONFIG['alpha_transparent'], 
                                   linewidth=1, label=label, zorder=30)
                line._trajectory_line = True
                other_trajs_labeled = True
                    
        # Highlight best test trajectory
        if best_trajectory is not None and len(best_trajectory) > 1:
            line, = self.ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 
                               color='blue', linewidth=3, 
                               label='Best Recent Traj', zorder=50)
            line._trajectory_line = True
                
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
        
        self.ax.set_title(f"Testing Trajectory - Ep {num_tests}", 
                         fontsize=VISUALIZATION_CONFIG['font_title'])
        filename = f"test_traj_ep_{num_tests}.png"
        
        plt.savefig(os.path.join(self.output_dir, filename), 
                    dpi=150, bbox_inches='tight')
    
    def plot_performance_metrics(self, deps: List[float], throughputs: List[float], 
                               episode_rewards: List[float] = None,
                               test_deps: List[float] = None,
                               test_throughputs: List[float] = None,
                               test_rewards: List[float] = None):

        fig, axes = plt.subplots(3, 1, figsize=VISUALIZATION_CONFIG['figure_size'])
        fig.suptitle('Performance Metrics', fontsize=VISUALIZATION_CONFIG['font_title'], y=0.98)
        train_episodes = range(1, len(deps) + 1)
        
        # Detection Error Probability Plot
        axes[0].plot(train_episodes, deps, color=VISUALIZATION_CONFIG['dep_color'], linewidth=1, label='Training')
        
        if test_deps and len(test_deps) > 0:
            test_episodes = range(len(deps) + 1, len(deps) + len(test_deps) + 1)
            axes[0].plot(test_episodes, test_deps, color=VISUALIZATION_CONFIG['dep_color'], linewidth=2, label='Testing')
            # Visual separator between training and testing
            axes[0].axvline(x=len(deps) + 0.5, color='gray', linestyle='--', alpha=0.7)
            axes[0].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        
        axes[0].set_ylabel('DEP', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[0].set_title('Detection Error Probability', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[0].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        # Throughput / Transmission Rate Plot
        axes[1].plot(train_episodes, throughputs, color=VISUALIZATION_CONFIG['throughput_color'], linewidth=1, label='Training')
        
        if test_throughputs and len(test_throughputs) > 0:
            test_episodes = range(len(throughputs) + 1, len(throughputs) + len(test_throughputs) + 1)
            axes[1].plot(test_episodes, test_throughputs, color=VISUALIZATION_CONFIG['throughput_color'], linewidth=2, label='Testing')
            axes[1].axvline(x=len(throughputs) + 0.5, color='gray', linestyle='--', alpha=0.7)
            axes[1].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        
        axes[1].set_ylabel('Throughput', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[1].set_title('Transmission Rate', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[1].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        # Reward / Learning Curve Plot
        if episode_rewards and len(episode_rewards) > 0:
            reward_episodes = range(1, len(episode_rewards) + 1)
            axes[2].plot(reward_episodes, episode_rewards, 
                        color=VISUALIZATION_CONFIG['reward_color'], 
                        linewidth=1, alpha=VISUALIZATION_CONFIG['alpha_semi'], label='Training')
            
            if test_rewards and len(test_rewards) > 0:
                test_reward_episodes = range(len(episode_rewards) + 1, len(episode_rewards) + len(test_rewards) + 1)
                axes[2].plot(test_reward_episodes, test_rewards, 
                            color=VISUALIZATION_CONFIG['reward_color'], linewidth=2, label='Testing')
                axes[2].axvline(x=len(episode_rewards) + 0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Calculate and plot moving average
            if len(episode_rewards) > 10:
                window_size = max(5, min(20, min(VISUALIZATION_CONFIG['moving_avg_window'], len(episode_rewards) // 10)))
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                moving_episodes = range(window_size, len(episode_rewards) + 1)
                axes[2].plot(moving_episodes, moving_avg, 
                            color=VISUALIZATION_CONFIG['moving_avg_color'], 
                            linewidth=1, label=f'Moving Average (window={window_size})')
            
            axes[2].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        else:
            axes[2].plot(train_episodes, [0] * len(train_episodes), 
                        color=VISUALIZATION_CONFIG['reward_color'], linewidth=1)
        
        axes[2].set_ylabel('Total Reward', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].set_xlabel('Episode', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].set_title('Learning Curve', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_metrics.png"), 
                   dpi=VISUALIZATION_CONFIG['dpi_high'], bbox_inches='tight')
        plt.close()
    
    def finalize_visualization(self):
        # Save final state and close plots
        if self.fig is not None:
            plt.ioff()
            plt.savefig(os.path.join(self.output_dir, "final_trajectory_visualization.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close(self.fig)

# Aliases for backwards compatibility
DynamicTrajectoryVisualizer = UnifiedVisualizer
StaticVisualizationGenerator = UnifiedVisualizer





