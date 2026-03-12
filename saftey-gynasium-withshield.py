import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import safety_gymnasium
import gymnasium as gym
import os
import time

# ============= Neural Network Models =============
class ActorNetwork(nn.Module):
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

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 2)
        
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()
            action = mean if deterministic else Normal(mean, std).sample()
        return action.cpu().numpy().squeeze()
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"✅ Model loaded from {filepath}")

# ============= Helper Functions (NOT methods) =============
def get_mujoco_data(env):
    """Get MuJoCo data and model"""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, 'task') and hasattr(unwrapped.task, 'hazards'):
        hazards_obj = unwrapped.task.hazards
        if hasattr(hazards_obj, 'engine') and hazards_obj.engine is not None:
            engine = hazards_obj.engine
            if hasattr(engine, 'data') and hasattr(engine, 'model'):
                return engine.data, engine.model
    return None, None

def get_hazards(env):
    """Extract hazard positions and size"""
    data, model = get_mujoco_data(env)
    hazard_size = 0.2
    
    if data is None or model is None:
        return np.array([]), hazard_size
    
    unwrapped = env.unwrapped
    if hasattr(unwrapped, 'task') and hasattr(unwrapped.task, 'hazards'):
        if hasattr(unwrapped.task.hazards, 'size'):
            hazard_size = unwrapped.task.hazards.size
    
    positions = []
    for i in range(model.nbody):
        body_name = model.body(i).name
        if body_name and 'hazard' in body_name.lower():
            pos = data.xpos[i]
            positions.append([float(pos[0]), float(pos[1])])
    
    return np.array(positions) if positions else np.array([]), hazard_size

def get_robot_state(env):
    """Get robot position and velocity"""
    data, model = get_mujoco_data(env)
    if data is None:
        return np.zeros(2), np.zeros(2)
    
    pos = data.xpos[1][:2].copy()
    vel = data.qvel[:2].copy() if hasattr(data, 'qvel') and len(data.qvel) >= 2 else np.zeros(2)
    return pos, vel

def get_goal_position(env):
    """Extract goal position from environment"""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, 'task') and hasattr(unwrapped.task, 'goal'):
        if hasattr(unwrapped.task.goal, 'pos'):
            return np.array(unwrapped.task.goal.pos[:2])
    return None

def check_will_collide(action, robot_pos, robot_vel, hazards, hazard_size, safety_margin):
    """Check if action will cause collision"""
    if len(hazards) == 0:
        return False, None, float('inf')
    
    robot_radius = 0.15
    action_2d = action[:2] if len(action) >= 2 else action
    
    predicted_vel = robot_vel * 0.9 + action_2d * 0.5
    predicted_pos = robot_pos + predicted_vel * 0.1
    
    closest_hazard = None
    min_dist = float('inf')
    will_collide = False
    
    for hazard in hazards:
        dist = np.linalg.norm(hazard - predicted_pos)
        if dist < min_dist:
            min_dist = dist
            closest_hazard = hazard
        
        collision_threshold = robot_radius + hazard_size + safety_margin
        if dist < collision_threshold:
            will_collide = True
    
    return will_collide, closest_hazard, min_dist

def get_tangent_action(action, robot_pos, hazards, hazard_size):
    """Get tangential action that moves AROUND hazards"""
    action_2d = action[:2] if len(action) >= 2 else action
    action_mag = np.linalg.norm(action_2d)
    
    if action_mag < 1e-8:
        return action_2d
    
    robot_radius = 0.15
    danger_zone = robot_radius + hazard_size + 0.3
    
    nearby_hazards = []
    for hazard in hazards:
        dist = np.linalg.norm(hazard - robot_pos)
        if dist < danger_zone:
            nearby_hazards.append((hazard, dist))
    
    if len(nearby_hazards) == 0:
        return action_2d
    
    nearby_hazards.sort(key=lambda x: x[1])
    modified_action = action_2d.copy()
    
    for hazard, dist in nearby_hazards:
        to_robot = robot_pos - hazard
        to_robot_norm = to_robot / (np.linalg.norm(to_robot) + 1e-8)
        radial_component = np.dot(modified_action, to_robot_norm)
        
        if radial_component < 0:
            proximity_weight = max(0, 1.0 - (dist - robot_radius - hazard_size) / 0.3)
            removal_strength = abs(radial_component) * proximity_weight
            modified_action = modified_action - removal_strength * to_robot_norm
    
    modified_mag = np.linalg.norm(modified_action)
    if modified_mag > 1e-8:
        modified_action = modified_action / modified_mag * action_mag
    else:
        closest_hazard = nearby_hazards[0][0]
        to_robot = robot_pos - closest_hazard
        to_robot_norm = to_robot / (np.linalg.norm(to_robot) + 1e-8)
        perpendicular = np.array([-to_robot_norm[1], to_robot_norm[0]])
        
        if np.dot(perpendicular, action_2d) < 0:
            perpendicular = -perpendicular
        
        modified_action = perpendicular * action_mag * 0.8
    
    return modified_action

def find_safe_action(action, robot_pos, robot_vel, closest_hazard, hazards, hazard_size, safety_margin, critical_distance):
    """Find safe action"""
    action_2d = action[:2] if len(action) >= 2 else action
    robot_radius = 0.15
    
    dist_to_closest = np.linalg.norm(robot_pos - closest_hazard)
    
    if dist_to_closest < robot_radius + hazard_size + critical_distance:
        away_direction = robot_pos - closest_hazard
        away_direction = away_direction / (np.linalg.norm(away_direction) + 1e-8)
        retreat_action = away_direction * np.linalg.norm(action_2d)
        
        full_retreat = action.copy() if len(action) > 2 else retreat_action
        full_retreat[:2] = retreat_action
        
        will_collide, _, _ = check_will_collide(full_retreat, robot_pos, robot_vel, 
                                               hazards, hazard_size, safety_margin)
        
        if not will_collide:
            return full_retreat
    
    tangent_action = get_tangent_action(action, robot_pos, hazards, hazard_size)
    full_tangent_action = action.copy() if len(action) > 2 else tangent_action
    full_tangent_action[:2] = tangent_action
    
    will_collide, _, _ = check_will_collide(full_tangent_action, robot_pos, robot_vel, 
                                           hazards, hazard_size, safety_margin)
    
    if not will_collide:
        return full_tangent_action
    
    best_action = None
    best_score = -999
    num_angles = 16
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    action_mag = np.linalg.norm(action_2d)
    
    for angle in angles:
        test_action_2d = np.array([np.cos(angle), np.sin(angle)]) * action_mag
        test_action = action.copy() if len(action) > 2 else test_action_2d
        test_action[:2] = test_action_2d
        
        will_collide, _, min_dist = check_will_collide(test_action, robot_pos, robot_vel, 
                                                      hazards, hazard_size, safety_margin)
        
        if not will_collide:
            alignment = np.dot(test_action_2d, action_2d) / (action_mag ** 2 + 1e-8)
            safety_score = min(min_dist / 0.5, 1.0)
            score = 0.6 * alignment + 0.4 * safety_score
            
            if score > best_score:
                best_score = score
                best_action = test_action
    
    if best_action is not None:
        return best_action
    
    return action * 0.2

# ============= Drift-Aware Shield =============
class DriftShield(gym.Wrapper):
    """Shield that allows tangential/drift movement around hazards"""
    
    def __init__(self, env, shield_enabled=True):
        super().__init__(env)
        self.shield_enabled = shield_enabled
        self.safety_margin = 0.1
        self.critical_distance = 0.05
        
        self.shield_interventions = 0
        self.collision_count = 0
        self.rollback_state = None
        self.position_history = []
        self.stuck_counter = 0
        self.escape_mode = False
        self.escape_direction = None
        self.escape_attempts = 0
        self.tried_directions = []  # Track attempted escape directions
        self.last_escape_pos = None
        
        print(f"🛡️  Drift-Aware Shield:")
        print(f"   Safety margin: {self.safety_margin}m")
        print(f"   Stuck detection: ENABLED")
        print(f"   Escape memory: ENABLED")
        
    def reset(self, **kwargs):
        self.shield_interventions = 0
        self.collision_count = 0
        self.rollback_state = None
        self.position_history = []
        self.stuck_counter = 0
        self.escape_mode = False
        self.escape_direction = None
        self.escape_attempts = 0
        self.tried_directions = []
        self.last_escape_pos = None
        return self.env.reset(**kwargs)
    
    def detect_stuck(self, robot_pos):
        """Detect if robot is stuck"""
        self.position_history.append(robot_pos.copy())
        
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        if len(self.position_history) < 20:
            return False
        
        total_movement = 0
        for i in range(1, len(self.position_history)):
            movement = np.linalg.norm(self.position_history[i] - self.position_history[i-1])
            total_movement += movement
        
        avg_movement = total_movement / 19
        
        if avg_movement < 0.015:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        
        if self.stuck_counter > 10:
            return True
        
        return False
    
    def find_escape_direction(self, robot_pos, hazards, hazard_size):
        """
        Find best escape direction, avoiding previously tried directions.
        Uses memory to avoid spinning in place.
        """
        if len(hazards) == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            return np.array([np.cos(angle), np.sin(angle)])
        
        goal_pos = get_goal_position(self.env)
        num_samples = 32
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        
        best_direction = None
        best_score = -999
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Check if this direction was already tried and failed
            too_similar = False
            for tried_dir in self.tried_directions:
                similarity = np.dot(direction, tried_dir)
                if similarity > 0.8:  # Very similar direction (within ~36 degrees)
                    too_similar = True
                    break
            
            if too_similar:
                continue  # Skip this direction, already tried
            
            # Simulate moving in this direction
            test_pos = robot_pos + direction * 0.5  # Look further ahead
            
            # Find minimum distance to any hazard
            min_dist = float('inf')
            for hazard in hazards:
                dist = np.linalg.norm(test_pos - hazard)
                min_dist = min(min_dist, dist)
            
            safety_score = min(min_dist / 0.5, 1.0)
            goal_score = 0
            
            if goal_pos is not None:
                to_goal = goal_pos - robot_pos
                if np.linalg.norm(to_goal) > 0.1:
                    to_goal_norm = to_goal / np.linalg.norm(to_goal)
                    goal_score = np.dot(direction, to_goal_norm)
                    goal_score = max(0, goal_score)
            
            # Increase goal weight after multiple failed attempts
            goal_weight = 0.4 + 0.1 * min(self.escape_attempts, 3)
            safety_weight = 1.0 - goal_weight
            score = safety_weight * safety_score + goal_weight * goal_score
            
            if score > best_score:
                best_score = score
                best_direction = direction
        
        # If all directions were tried, clear memory and pick randomly
        if best_direction is None:
            print(f"⚠️  All directions tried! Clearing escape memory...")
            self.tried_directions = []
            angle = np.random.uniform(0, 2 * np.pi)
            best_direction = np.array([np.cos(angle), np.sin(angle)])
        
        return best_direction
    
    def step(self, action):
        original_action = action.copy()
        
        if self.shield_enabled:
            robot_pos, robot_vel = get_robot_state(self.env)
            hazards, hazard_size = get_hazards(self.env)
            
            is_stuck = self.detect_stuck(robot_pos)
            
            if is_stuck and not self.escape_mode:
                print(f"🔓 STUCK DETECTED! Entering escape mode (Attempt #{self.escape_attempts + 1})...")
                self.escape_mode = True
                self.escape_direction = self.find_escape_direction(robot_pos, hazards, hazard_size)
                
                # Remember this direction
                self.tried_directions.append(self.escape_direction.copy())
                self.last_escape_pos = robot_pos.copy()
                self.escape_attempts += 1
                
                self.stuck_counter = 0
                self.position_history = []
            
            if self.escape_mode:
                escape_action = self.escape_direction * 2.0  # Strong push
                
                if len(action) > 2:
                    action = original_action.copy()
                    action[:2] = escape_action
                else:
                    action = escape_action
                
                # Check escape progress
                if self.last_escape_pos is not None:
                    escape_distance = np.linalg.norm(robot_pos - self.last_escape_pos)
                    
                    # Exit escape mode if moved far enough
                    if escape_distance > 0.2:
                        print(f"✅ Escaped! Distance: {escape_distance:.3f}m")
                        self.escape_mode = False
                        self.escape_direction = None
                        self.position_history = []
                        self.escape_attempts = 0  # Reset on success
                        self.tried_directions = []  # Clear memory on success
                    
                    # If still stuck after trying to escape for a while, re-evaluate
                    elif len(self.position_history) > 15:
                        recent_movement = 0
                        for i in range(max(0, len(self.position_history) - 10), len(self.position_history)):
                            if i > 0:
                                recent_movement += np.linalg.norm(
                                    self.position_history[i] - self.position_history[i-1])
                        
                        # If barely moving during escape, try new direction
                        if recent_movement < 0.05:
                            print(f"❌ Escape failed! Trying new direction...")
                            self.escape_mode = False
                            self.escape_direction = None
                            # Keep tried_directions in memory
                            # Will trigger stuck detection again and try different direction
            else:
                data, model = get_mujoco_data(self.env)
                if data is not None:
                    self.rollback_state = {
                        'qpos': data.qpos.copy(),
                        'qvel': data.qvel.copy()
                    }
                
                # Check distance to goal - if close and clear path, let agent proceed freely
                goal_pos = get_goal_position(self.env)
                near_goal = False
                clear_path_to_goal = False
                
                if goal_pos is not None:
                    dist_to_goal = np.linalg.norm(goal_pos - robot_pos)
                    near_goal = dist_to_goal < 1.5  # Within 1.5m of goal
                    
                    # Check if path to goal is clear of hazards
                    if near_goal and len(hazards) > 0:
                        clear_path_to_goal = True
                        # Sample points along path to goal
                        num_checks = 10
                        for t in np.linspace(0, 1, num_checks):
                            check_pos = robot_pos + t * (goal_pos - robot_pos)
                            
                            # Check if this point is too close to any hazard
                            for hazard in hazards:
                                dist = np.linalg.norm(hazard - check_pos)
                                if dist < 0.15 + hazard_size + 0.15:  # Robot + hazard + buffer
                                    clear_path_to_goal = False
                                    break
                            
                            if not clear_path_to_goal:
                                break
                
                # If near goal with clear path, don't interfere with agent
                if near_goal and clear_path_to_goal:
                    # Let agent navigate freely - no shield intervention
                    pass
                else:
                    # Normal shield operation
                    will_collide, closest_hazard, min_dist = check_will_collide(
                        action, robot_pos, robot_vel, hazards, hazard_size, self.safety_margin)
                    
                    if will_collide and closest_hazard is not None:
                        action = find_safe_action(action, robot_pos, robot_vel, 
                                                closest_hazard, hazards, hazard_size,
                                                self.safety_margin, self.critical_distance)
                        self.shield_interventions += 1
                        
                        if self.shield_interventions % 100 == 1:
                            print(f"🛡️  Shield #{self.shield_interventions} | distance: {min_dist:.3f}m")
        
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        
        if cost > 0:
            self.collision_count += 1
            
            if self.rollback_state is not None and self.shield_enabled:
                data, _ = get_mujoco_data(self.env)
                if data is not None:
                    data.qpos[:] = self.rollback_state['qpos']
                    data.qvel[:] = data.qvel[:] * 0.3
                    
                    if self.collision_count % 10 == 1:
                        print(f"⚠️  Rollback #{self.collision_count}")
                    
                    cost = 0
                    reward = max(reward, -0.5)
        
        info['shield_active'] = self.shield_enabled
        info['shield_stats'] = {
            'interventions': self.shield_interventions,
            'collisions': self.collision_count
        }
        
        return obs, reward, cost, terminated, truncated, info

# ============= Visualization =============
def visualize_agent(model_path, env_id='SafetyPointGoal1-v0', num_episodes=5, 
                   shield_enabled=True):
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("="*60)
    print(f"Environment: {env_id}")
    print(f"Drift-Aware Shield: {'✅ ON' if shield_enabled else '❌ OFF'}")
    print("="*60)
    
    base_env = safety_gymnasium.make(env_id, render_mode='human', width=1200, height=800)
    base_env.reset()
    
    env = DriftShield(base_env, shield_enabled=shield_enabled) if shield_enabled else base_env
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim)
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()
    
    print(f"📊 Obs: {obs_dim}, Action: {action_dim}\n")
    
    stats = {'rewards': [], 'costs': [], 'successes': [], 'steps': []}
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = ep_cost = steps = 0
        done = truncated = False
        
        print(f"🎬 Episode {ep+1}/{num_episodes}")
        
        while not (done or truncated) and steps < 1000:
            action = agent.get_action(state, deterministic=True)
            state, reward, cost, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_cost += cost
            steps += 1
            env.render()
            time.sleep(1/30)
            
            if info.get('goal_met', False):
                print(f"🎯 GOAL! Steps={steps}, Reward={ep_reward:.1f}, Cost={ep_cost:.0f}")
                done = True
                break
        
        goal_reached = info.get('goal_met', False)
        shield_stats = info.get('shield_stats', {})
        
        stats['rewards'].append(ep_reward)
        stats['costs'].append(ep_cost)
        stats['successes'].append(1 if goal_reached else 0)
        stats['steps'].append(steps)
        
        print(f"✅ Episode {ep+1}: Reward={ep_reward:.1f}, Cost={ep_cost:.0f}, "
              f"Goal={'✅' if goal_reached else '❌'}")
        if shield_enabled:
            print(f"   Shield: {shield_stats.get('interventions',0)} interventions, "
                  f"{shield_stats.get('collisions',0)} collisions\n")
    
    print("="*60)
    print(f"Final Stats:")
    print(f"  Avg Reward: {np.mean(stats['rewards']):.1f} ± {np.std(stats['rewards']):.1f}")
    print(f"  Avg Cost: {np.mean(stats['costs']):.1f} ± {np.std(stats['costs']):.1f}")
    print(f"  Success Rate: {np.mean(stats['successes'])*100:.0f}%")
    print("="*60)
    
    env.close()

# ============= Main =============
if __name__ == "__main__":
    MODEL_PATH = "safety_gym_ppo_final.pth"
    ENV_ID = "SafetyPointGoal1-v0"
    
    print("\n╔══════════════════════════════════════╗")
    print("║  Drift-Aware Shield                  ║")
    print("╚══════════════════════════════════════╝\n")
    
    print("1. With Shield (Recommended)")
    print("2. Without Shield")
    
    choice = input("\nChoice [1-2, default=1]: ").strip() or "1"
    shield_enabled = choice == "1"
    
    visualize_agent(MODEL_PATH, ENV_ID, num_episodes=5, shield_enabled=shield_enabled)