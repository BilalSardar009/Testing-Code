import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import safety_gymnasium
import gymnasium as gym
from datetime import datetime
import os

# ============= Neural Network Models =============
class ActorNetwork(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = torch.tanh(self.mean(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.value = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)


# ============= Training Logger =============
class TrainingLogger:
    """Real-time training visualization"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_lengths = []
        self.success_rates = []
        self.actor_losses = []
        self.critic_losses = []
        
        # Moving averages
        self.reward_window = deque(maxlen=100)
        self.cost_window = deque(maxlen=100)
        self.success_window = deque(maxlen=100)
        
        # For non-blocking plots
        self.fig = None
        plt.ion()  # Enable interactive mode globally
        
    def log_episode(self, reward, cost, length, success):
        """Log episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_costs.append(cost)
        self.episode_lengths.append(length)
        
        self.reward_window.append(reward)
        self.cost_window.append(cost)
        self.success_window.append(success)
        
        self.success_rates.append(np.mean(self.success_window))
    
    def log_losses(self, actor_loss, critic_loss):
        """Log training losses"""
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
    
    def plot_training_progress(self, episode, save=True, show=True):
        """Plot comprehensive training progress (non-blocking)"""
        
        # Close previous figure if exists
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Episode Rewards
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.reward_window) > 0:
            window_size = min(50, len(self.episode_rewards))
            if len(self.episode_rewards) >= window_size:
                moving_avg = np.convolve(self.episode_rewards, 
                                        np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(self.episode_rewards)), 
                        moving_avg, color='blue', linewidth=2, label=f'MA({window_size})')
        ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Episode Costs (Safety Violations)
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.plot(self.episode_costs, alpha=0.3, color='red', label='Episode Cost')
        if len(self.cost_window) > 0:
            window_size = min(50, len(self.episode_costs))
            if len(self.episode_costs) >= window_size:
                moving_avg = np.convolve(self.episode_costs, 
                                        np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(self.episode_costs)), 
                        moving_avg, color='red', linewidth=2, label=f'MA({window_size})')
        ax2.set_title('Episode Costs (Safety Violations)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Success Rate
        ax3 = self.fig.add_subplot(gs[0, 2])
        ax3.plot(self.success_rates, color='green', linewidth=2)
        ax3.fill_between(range(len(self.success_rates)), self.success_rates, 
                         alpha=0.3, color='green')
        ax3.set_title('Success Rate (100-episode window)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3)
        
        # 4. Episode Lengths
        ax4 = self.fig.add_subplot(gs[1, 0])
        ax4.plot(self.episode_lengths, alpha=0.5, color='purple')
        ax4.set_title('Episode Lengths', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.grid(True, alpha=0.3)
        
        # 5. Reward vs Cost Scatter
        ax5 = self.fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(self.episode_costs, self.episode_rewards, 
                            c=range(len(self.episode_rewards)), 
                            cmap='viridis', alpha=0.6, s=20)
        ax5.set_title('Reward vs Cost Trade-off', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Cost')
        ax5.set_ylabel('Reward')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Episode')
        
        # 6. Training Losses
        ax6 = self.fig.add_subplot(gs[1, 2])
        if len(self.actor_losses) > 0:
            ax6_twin = ax6.twinx()
            ax6.plot(self.actor_losses, color='orange', label='Actor Loss', alpha=0.7)
            ax6_twin.plot(self.critic_losses, color='brown', label='Critic Loss', alpha=0.7)
            ax6.set_title('Training Losses', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Update Step')
            ax6.set_ylabel('Actor Loss', color='orange')
            ax6_twin.set_ylabel('Critic Loss', color='brown')
            ax6.tick_params(axis='y', labelcolor='orange')
            ax6_twin.tick_params(axis='y', labelcolor='brown')
            ax6.grid(True, alpha=0.3)
        
        # 7. Cumulative Metrics
        ax7 = self.fig.add_subplot(gs[2, :])
        episodes = range(len(self.episode_rewards))
        cumulative_rewards = np.cumsum(self.episode_rewards)
        cumulative_costs = np.cumsum(self.episode_costs)
        
        ax7_twin = ax7.twinx()
        ax7.plot(episodes, cumulative_rewards, color='blue', 
                label='Cumulative Reward', linewidth=2)
        ax7_twin.plot(episodes, cumulative_costs, color='red', 
                     label='Cumulative Cost', linewidth=2)
        ax7.set_title('Cumulative Rewards and Costs', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Cumulative Reward', color='blue')
        ax7_twin.set_ylabel('Cumulative Cost', color='red')
        ax7.tick_params(axis='y', labelcolor='blue')
        ax7_twin.tick_params(axis='y', labelcolor='red')
        ax7.grid(True, alpha=0.3)
        ax7.legend(loc='upper left')
        ax7_twin.legend(loc='upper right')
        
        # Add overall statistics
        self.fig.suptitle(f'Training Progress - Episode {episode}\n'
                    f'Avg Reward: {np.mean(self.reward_window):.2f} | '
                    f'Avg Cost: {np.mean(self.cost_window):.2f} | '
                    f'Success Rate: {np.mean(self.success_window)*100:.1f}%',
                    fontsize=14, fontweight='bold')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'{self.log_dir}/training_progress_ep{episode}.png', 
                       dpi=150, bbox_inches='tight')
        
        if show:
            plt.draw()  # Update the figure
            plt.pause(0.001)  # Brief pause to update display (non-blocking)
        else:
            plt.close(self.fig)
    
    def save_metrics(self):
        """Save metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.savez(f'{self.log_dir}/metrics_{timestamp}.npz',
                episode_rewards=self.episode_rewards,
                episode_costs=self.episode_costs,
                episode_lengths=self.episode_lengths,
                success_rates=self.success_rates,
                actor_losses=self.actor_losses,
                critic_losses=self.critic_losses)
        print(f"Metrics saved to {self.log_dir}/metrics_{timestamp}.npz")


# ============= PPO Agent =============
class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 2)
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Memory
        self.memory = []
        
        # Loss tracking
        self.last_actor_loss = 0
        self.last_critic_loss = 0
        
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            
        return action.cpu().numpy().squeeze()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self, batch_size=64, epochs=10):
        """Update policy using PPO"""
        if len(self.memory) < batch_size:
            return
        
        # Convert memory to numpy arrays first, then to tensors (fixes warning)
        states = np.array([t['state'] for t in self.memory], dtype=np.float32)
        actions = np.array([t['action'] for t in self.memory], dtype=np.float32)
        rewards = [t['reward'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        old_log_probs = np.array([t['log_prob'] for t in self.memory], dtype=np.float32)
        old_values = [t['value'] for t in self.memory]
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track losses
        total_actor_loss = 0
        total_critic_loss = 0
        update_count = 0
        
        # PPO update
        for _ in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Actor loss
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -dist.entropy().mean()
                
                total_actor_loss_batch = actor_loss - self.entropy_coef * entropy_loss
                
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # Accumulate losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                update_count += 1
        
        # Store average losses
        self.last_actor_loss = total_actor_loss / update_count if update_count > 0 else 0
        self.last_critic_loss = total_critic_loss / update_count if update_count > 0 else 0
        
        # Clear memory
        self.memory = []
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ============= Training Loop =============
def train_agent(env_id='SafetyPointGoal1-v0', episodes=1000, render_every=50, 
                max_steps=1000, visualize_every=50):
    """Main training loop with Safety-Gymnasium"""
    
    # Create environment with rendering
    env = safety_gymnasium.make(
        env_id,
        render_mode='human',
        width=800,
        height=600
    )
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_id}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action space: {env.action_space}")
    print("=" * 50)
    
    # Create agent and logger
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=3e-4)
    logger = TrainingLogger()
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            # Get action
            action = agent.get_action(state)
            
            # Store old values for PPO (must compute before step)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                mean, log_std = agent.actor(state_tensor)
                std = log_std.exp()
                dist = Normal(mean, std)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                log_prob = dist.log_prob(action_tensor).sum().item()
                value = agent.critic(state_tensor).item()
            
            # Step environment - Safety-Gymnasium returns 6 values
            next_state, reward, cost, terminated, truncated, info = env.step(action)
            
            # Accumulate episode metrics
            episode_cost += cost
            episode_reward += reward
            
            # For PPO, we need to know if episode ended for any reason
            done_flag = terminated or truncated
            
            # Store transition for training
            agent.store_transition(state, action, reward, next_state, done_flag, log_prob, value)
            
            # Update state and counters
            state = next_state
            steps += 1
            
            # Update termination flags for loop control
            done = terminated
            
            # Render periodically
            if episode % render_every == 0:
                env.render()
        
        # Check if goal was reached
        goal_reached = info.get('goal_met', False)
        
        # Log episode
        logger.log_episode(episode_reward, episode_cost, steps, goal_reached)
        
        # Update agent
        if episode > 0 and episode % 10 == 0:
            agent.update(batch_size=64, epochs=10)
            logger.log_losses(agent.last_actor_loss, agent.last_critic_loss)
        
        # Print progress with real-time updates
        if episode % 10 == 0:
            avg_reward = np.mean(logger.reward_window)
            avg_cost = np.mean(logger.cost_window)
            avg_success = np.mean(logger.success_window)
            
            print(f"\rEpisode {episode:4d}/{episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Cost: {avg_cost:6.2f} | "
                  f"Success: {avg_success*100:5.1f}% | "
                  f"Steps: {steps:3d}", end='')
            
            # New line every 50 episodes for cleaner output
            if episode % 50 == 0:
                print()  # New line
        
        # Visualize progress
        if episode > 0 and episode % visualize_every == 0:
            print(f"📊 Generating visualization for episode {episode}...")
            logger.plot_training_progress(episode, save=True, show=True)
        
        # Save checkpoint
        if episode % 100 == 0 and episode > 0:
            print()  # New line before checkpoint message
            agent.save(f"safety_gym_ppo_checkpoint_{episode}.pth")
            logger.save_metrics()
            print(f"💾 Checkpoint saved at episode {episode}")
    
    env.close()
    return agent, logger


# ============= Evaluation =============
def evaluate_agent(agent, env_id='SafetyPointGoal1-v0', episodes=10):
    """Evaluate trained agent"""
    env = safety_gymnasium.make(
        env_id,
        render_mode='human',
        width=800,
        height=600
    )
    
    total_rewards = []
    total_costs = []
    successes = 0
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Use deterministic policy for evaluation
            action = agent.get_action(state, deterministic=True)
            
            # Step environment - Safety-Gymnasium returns 6 values
            state, reward, cost, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += cost
            env.render()
        
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        
        goal_reached = info.get('goal_met', False)
        if goal_reached:
            successes += 1
        
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Cost = {episode_cost:.2f}, Goal Reached = {goal_reached}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Cost: {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")
    print(f"Success Rate: {successes / episodes * 100:.1f}%")
    
    env.close()


# ============= Main Execution =============
if __name__ == "__main__":
    print("Starting Safety-Gymnasium Robot Training")
    print("=" * 50)
    print("Configuration:")
    print("- Environment: Safety-Gymnasium (Official PKU-Alignment)")
    print("- Algorithm: PPO (Proximal Policy Optimization)")
    print("- Safety: Built-in cost functions and constraints")
    print("- Visualization: Real-time training progress plots")
    print("=" * 50)
    
    env_id = 'SafetyPointGoal1-v0'
    
    # Verify Safety-Gymnasium API
    print(f"\n🔍 Verifying Safety-Gymnasium installation...")
    try:
        test_env = safety_gymnasium.make(env_id)
        obs, info = test_env.reset()
        action = test_env.action_space.sample()
        
        # Test step return - should be 6 values
        step_result = test_env.step(action)
        assert len(step_result) == 6, f"Expected 6 return values, got {len(step_result)}"
        
        obs, reward, cost, terminated, truncated, info = step_result
        
        print(f"✅ Safety-Gymnasium API verified!")
        print(f"   - Observation space: {test_env.observation_space.shape}")
        print(f"   - Action space: {test_env.action_space.shape}")
        print(f"   - Step returns 6 values: ✓")
        print(f"   - Cost tracking: ✓")
        test_env.close()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please install: pip install safety-gymnasium gymnasium")
        exit(1)
    
    print(f"\nTraining with environment: {env_id}")
    print("This uses the official Safety-Gymnasium framework")
    print("with built-in hazards, costs, and safety constraints.\n")
    
    # Train agent
    print("Starting training...")
    agent, logger = train_agent(
        env_id=env_id,
        episodes=500,
        render_every=25,
        visualize_every=50
    )
    
    # Final visualization
    print("\n📊 Generating final training visualization...")
    logger.plot_training_progress(len(logger.episode_rewards), save=True, show=True)
    logger.save_metrics()
    
    # Keep final plot open for viewing
    print("\n✅ Training visualization complete! (Plot window will stay open)")
    print("   Close the plot window to continue to evaluation...")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Block here to show final results
    
    # Save final model
    agent.save("safety_gym_ppo_final.pth")
    print("\nFinal model saved as 'safety_gym_ppo_final.pth'")
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    evaluate_agent(agent, env_id=env_id, episodes=5)
    
    print("\nTraining complete!")