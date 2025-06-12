# adapt.py
# pylint: disable=line-too-long, too-many-locals, too-many-statements, redefined-outer-name, wrong-import-position

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import imageio # For video
import csv

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# IMPORTANT: Set MODEL_SAVE_DIR to a NEW directory for this specific run
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models_universal_custom_rewards_v3_short") 
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- Environment and Wrappers ---
from til_environment.gridworld import raw_env as TILRawEnv
from til_environment.gridworld import DEFAULT_REWARDS_DICT as BASE_ENV_SPARSE_DEFAULTS
from til_environment.gridworld import RewardNames, NUM_ITERS as ENV_NUM_ITERS_FROM_GRIDWORLD
from til_environment.types import Action as EnvAction
from custom_rewards import CustomRewardWrapper # Ensure this is your LATEST (V3.1) custom_rewards.py
from pettingzoo.utils import wrappers as pz_wrappers
from pettingzoo.utils.conversions import aec_to_parallel_wrapper


# --- Constants ---
GRID_SIZE = 16 
VIEWCONE_HEIGHT = 7 
VIEWCONE_WIDTH = 5  
N_ACTIONS = len(EnvAction) 

# --- CONFIGURABLE PARAMETERS ---
LOAD_PREVIOUS_MODEL = True 
# Point these to your V1 model checkpoint (e.g., the one from ep 20,000)
MODEL_TO_LOAD_FILENAME = "universal_agent_best.pth" # This is used if OLD_MODEL_ABSOLUTE_PATH is None
OLD_MODEL_ABSOLUTE_PATH = "/home/jupyter/til-25/rl/rlagent_spencer/models_universal_custom_rewards_v3_short/universal_agent_best.pth" 

FORCE_EPSILON_START_ON_LOAD = False 
EPSILON_START_CONTINUE = 0.25 # Start with re-exploration for V3 rewards

# If True, attempts to load optimizer state. If False, optimizer starts fresh.
# Recommended: False when changing LR or adapting to significantly new rewards.
LOAD_OPTIMIZER_STATE_ON_CONTINUE = False 

LEARNING_RATE = 1e-5
GAMMA = 0.99
REPLAY_BUFFER_CAPACITY = 500000 
BATCH_SIZE = 256 
TARGET_UPDATE_FREQ = 1000 

EPSILON_START_SCRATCH = 1.0 # Not used as LOAD_PREVIOUS_MODEL is True
EPSILON_END = 0.01 
# Total episodes for this run (V1 model was at 20k, so this means 40k new episodes)
NUM_EPISODES = 40000 # New total target (e.g., if V1 was 20k, this means 40k new eps)


LEARNING_STARTS_AFTER_STEPS = 2000 
LEARN_EVERY_N_STEPS = 4
SAVE_MODEL_EVERY_N_EPISODES = 1000 
LOG_EVERY_N_EPISODES = 100 

ENV_RENDER_MODE = None 
NOVICE_MAP_TRAINING = True 
ENABLE_VIDEO_RECORDING = False 
VIDEO_SAVE_DIR = os.path.join(SCRIPT_DIR, "videos_v1_adapted_to_v3") 
RECORD_VIDEO_TRIGGER_INTERACTION = int(0.8 * (NUM_EPISODES * 100 * 4)) # 80% of total interactions in this run
# --- End Configurable Parameters ---


# --- Helper: Preprocess Observation ---
def preprocess_observation(obs_dict: dict, max_steps_for_norm: int) -> tuple[np.ndarray, np.ndarray]:
    raw_viewcone_list = obs_dict.get('viewcone', [[0] * VIEWCONE_WIDTH for _ in range(VIEWCONE_HEIGHT)])
    raw_viewcone = np.array(raw_viewcone_list, dtype=np.uint8)
    vc_channels = 7
    cnn_input_raw = np.zeros((vc_channels, VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.float32)
    if raw_viewcone.shape == (VIEWCONE_HEIGHT, VIEWCONE_WIDTH):
        tile_type = raw_viewcone & 0b11
        cnn_input_raw[0] = tile_type / 3.0
        cnn_input_raw[1] = (raw_viewcone >> 2) & 1; cnn_input_raw[2] = (raw_viewcone >> 3) & 1
        cnn_input_raw[3] = (raw_viewcone >> 4) & 1; cnn_input_raw[4] = (raw_viewcone >> 5) & 1
        cnn_input_raw[5] = (raw_viewcone >> 6) & 1; cnn_input_raw[6] = (raw_viewcone >> 7) & 1
    direction_val = obs_dict.get('direction', 0); location_val = obs_dict.get('location', [0, 0])
    step_val = obs_dict.get('step', 0); agent_is_scout_raw = obs_dict.get('scout', 0)
    direction = int(direction_val.item()) if isinstance(direction_val, np.ndarray) and direction_val.size ==1 else int(direction_val)
    if isinstance(location_val, np.ndarray) and location_val.shape==(2,): loc_x, loc_y = int(location_val[0]), int(location_val[1])
    elif isinstance(location_val, (list, tuple)) and len(location_val)==2: loc_x, loc_y = int(location_val[0]), int(location_val[1])
    else: loc_x, loc_y = 0,0
    step = int(step_val.item()) if isinstance(step_val, np.ndarray) and step_val.size == 1 else int(step_val)
    scout_flag = float(agent_is_scout_raw.item()) if isinstance(agent_is_scout_raw, np.ndarray) and agent_is_scout_raw.size == 1 else float(agent_is_scout_raw)
    dir_one_hot = np.zeros(4, dtype=np.float32); dir_one_hot[min(3,max(0,direction))] = 1.0
    norm_loc_x = loc_x / max(1,(GRID_SIZE-1)); norm_loc_y = loc_y / max(1,(GRID_SIZE-1))
    norm_step = step / max(1,(max_steps_for_norm-1.0))
    role_feature = np.array([scout_flag], dtype=np.float32)
    flat_features_list = [dir_one_hot, np.array([norm_loc_x], dtype=np.float32), np.array([norm_loc_y], dtype=np.float32), np.array([norm_step], dtype=np.float32), role_feature]
    return cnn_input_raw, np.concatenate(flat_features_list).astype(np.float32)

# --- Helper: QNetwork ---
class QNetwork(nn.Module):
    def __init__(self, cnn_input_channels: int, flat_input_size: int, n_actions: int):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(cnn_input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        with torch.no_grad():
            dummy_cnn_input = torch.randn(1, cnn_input_channels, VIEWCONE_HEIGHT, VIEWCONE_WIDTH)
            x_cnn_calc = F.relu(self.bn1(self.conv1(dummy_cnn_input)))
            x_cnn_calc = F.relu(self.bn2(self.conv2(x_cnn_calc)))
        self.cnn_flat_size = int(np.prod(x_cnn_calc.shape[1:]))
        self.fc_flat1 = nn.Linear(flat_input_size, 64) 
        self.bn_flat1 = nn.BatchNorm1d(64)
        self.fc_combined1 = nn.Linear(self.cnn_flat_size + 64, 256)
        self.bn_combined1 = nn.BatchNorm1d(256)
        self.fc_output = nn.Linear(256, n_actions)

    def forward(self, cnn_x: torch.Tensor, flat_x: torch.Tensor) -> torch.Tensor:
        x_cnn = F.relu(self.bn1(self.conv1(cnn_x)))
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_flat = F.relu(self.bn_flat1(self.fc_flat1(flat_x)))
        x_combined = torch.cat((x_cnn, x_flat), dim=1)
        x = F.relu(self.bn_combined1(self.fc_combined1(x_combined)))
        q_values = self.fc_output(x)
        return q_values

Experience = namedtuple("Experience", field_names=["cnn_state", "flat_state", "action", "reward", "next_cnn_state", "next_flat_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int): self.buffer = deque(maxlen=capacity)
    def add(self, cnn_s, flat_s, a, r, cnn_s_p, flat_s_p, d): self.buffer.append(Experience(cnn_s, flat_s, a, r, cnn_s_p, flat_s_p, d))
    def sample(self, batch_size: int): return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []
    def __len__(self) -> int: return len(self.buffer)

class UniversalDQNAgent:
    def __init__(self, cnn_input_channels: int, flat_input_size: int, n_actions: int,
                 replay_buffer_capacity: int, batch_size: int, gamma: float,
                 lr: float, target_update_freq: int,
                 epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int,
                 device: torch.device, max_steps_per_episode_for_preprocessing: int):
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr # Store lr for re-initializing optimizer if needed
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0 
        self.epsilon_start = epsilon_start 
        self.epsilon = epsilon_start       
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps) 
        self.max_steps_const_for_preprocess = max_steps_per_episode_for_preprocessing
        self.policy_net = QNetwork(cnn_input_channels, flat_input_size, n_actions).to(self.device)
        self.target_net = QNetwork(cnn_input_channels, flat_input_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True) # Use self.lr
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def _preprocess_and_to_tensor(self, obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_raw, flat_raw = preprocess_observation(obs_dict, self.max_steps_const_for_preprocess)
        cnn_tensor = torch.from_numpy(cnn_raw).unsqueeze(0).float().to(self.device)
        flat_tensor = torch.from_numpy(flat_raw).unsqueeze(0).float().to(self.device)
        return cnn_tensor, flat_tensor

    def select_action(self, obs_dict: dict, is_training: bool = True) -> int:
        sample = random.random()
        if is_training and self.epsilon_decay_steps > 0: 
            decay_amount_per_call = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount_per_call)
        if is_training and sample < self.epsilon:
            return random.randrange(self.n_actions) 
        else: 
            original_mode = self.policy_net.training 
            self.policy_net.eval() 
            with torch.no_grad():
                cnn_state_tensor, flat_state_tensor = self._preprocess_and_to_tensor(obs_dict)
                q_values = self.policy_net(cnn_state_tensor, flat_state_tensor)
            if original_mode: self.policy_net.train() 
            return q_values.max(1)[1].item() 

    def store_experience(self, current_obs_dict: dict, action: int, reward: float, 
                         next_obs_dict: dict | None, done: bool):
        if current_obs_dict is None: return 
        cnn_s, flat_s = preprocess_observation(current_obs_dict, self.max_steps_const_for_preprocess)
        if next_obs_dict is None or done: 
            next_cnn_s = np.zeros_like(cnn_s)
            next_flat_s = np.zeros_like(flat_s)
        else:
            next_cnn_s, next_flat_s = preprocess_observation(next_obs_dict, self.max_steps_const_for_preprocess)
        self.replay_buffer.add(cnn_s, flat_s, action, reward, next_cnn_s, next_flat_s, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None 
        experiences = self.replay_buffer.sample(self.batch_size)
        if not experiences: return None 
        batch = Experience(*zip(*experiences))
        cnn_states = torch.from_numpy(np.array(batch.cnn_state)).float().to(self.device)
        flat_states = torch.from_numpy(np.array(batch.flat_state)).float().to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device) 
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_cnn_states = torch.from_numpy(np.array(batch.next_cnn_state)).float().to(self.device)
        next_flat_states = torch.from_numpy(np.array(batch.next_flat_state)).float().to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.bool).unsqueeze(1).to(self.device) 
        self.policy_net.train() 
        current_q_values = self.policy_net(cnn_states, flat_states).gather(1, actions)
        with torch.no_grad(): 
            next_q_values_target_net = self.target_net(next_cnn_states, next_flat_states).max(1)[0].unsqueeze(1)
            next_q_values_target_net[dones] = 0.0
        expected_q_values = rewards + (self.gamma * next_q_values_target_net)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) 
        self.optimizer.step()
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def save_checkpoint(self, path: str, episode: int | None = None, total_interactions: int | None = None):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.lr, # <<<< SAVE CURRENT LEARNING RATE
            'epsilon': self.epsilon, 
            'train_step_counter': self.train_step_counter,
            'epsilon_start_val_if_continued': self.epsilon_start, 
            'episode': episode,
            'total_interactions': total_interactions
        }
        torch.save(checkpoint, path)
        print(f"UniversalDQNAgent Checkpoint saved: {path} (Ep: {episode}, EnvInteractions: {total_interactions})")

    def load_checkpoint(self, path: str, 
                        new_epsilon_start_for_continue: float, 
                        new_epsilon_end: float, 
                        new_total_decay_steps: int,
                        load_optimizer_state_flag: bool): # <<<< RENAMED PARAMETER
        loaded_episode, loaded_interactions = 0, 0
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            
            # Optimizer loading logic based on the flag
            if load_optimizer_state_flag:
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        # Optional: Check if loaded LR matches current agent's LR
                        # saved_optim_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                        # if abs(saved_optim_lr - self.lr) > 1e-7: # If LRs are different
                        #     print(f"  WARNING: Optimizer LR in checkpoint ({saved_optim_lr}) differs from current agent LR ({self.lr}). Re-initializing optimizer with current LR.")
                        #     self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
                        # else:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"  Optimizer state loaded from checkpoint.")
                    except Exception as e_optim_load:
                        print(f"  WARNING: Could not load optimizer state from checkpoint: {e_optim_load}. Optimizer re-initialized with current LR: {self.lr}.")
                        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
                else:
                    print(f"  INFO: Optimizer state not found in checkpoint. Optimizer uses fresh state with current LR: {self.lr}.")
                    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True) # Ensure fresh if not found
            else:
                print(f"  INFO: Optimizer state NOT loaded as per configuration. Optimizer re-initialized with current LR: {self.lr}.")
                self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True) # Re-initialize

            self.epsilon_start = new_epsilon_start_for_continue 
            self.epsilon = self.epsilon_start 
            self.epsilon_end = new_epsilon_end 
            self.epsilon_decay_steps = new_total_decay_steps 
            
            self.train_step_counter = checkpoint.get('train_step_counter', 0) 
            loaded_episode = checkpoint.get('episode', 0)
            loaded_interactions = checkpoint.get('total_interactions', 0)
            
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
            self.policy_net.train() 
            self.target_net.eval()  
            print(f"UniversalDQNAgent Checkpoint loaded: {path}. Last Ep: {loaded_episode}, EnvInteractions: {loaded_interactions}")
            print(f"  Epsilon reconfigured: Effective Start for decay = {self.epsilon_start:.4f} (current: {self.epsilon:.4f}), End = {self.epsilon_end:.4f}, Decay Steps = {self.epsilon_decay_steps}")
        except Exception as e: 
            print(f"ERROR loading checkpoint {path}: {e}")
            raise
        return loaded_episode, loaded_interactions

def append_to_csv(filename: str, data_row: list, header: list | None = None):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists and header: 
                writer.writerow(header)
            writer.writerow(data_row)
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")

def main():
    run_start_time = datetime.now()
    print(f"Starting training run at {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure MODEL_SAVE_DIR is set correctly for this run
    # (Global MODEL_SAVE_DIR is used directly now)
    if not os.path.exists(MODEL_SAVE_DIR): 
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created model save directory: {MODEL_SAVE_DIR}")
    if ENABLE_VIDEO_RECORDING and not os.path.exists(VIDEO_SAVE_DIR): 
        os.makedirs(VIDEO_SAVE_DIR)
        print(f"Created video save directory: {VIDEO_SAVE_DIR}")

    video_has_been_recorded_this_run = False 

    csv_log_filename = os.path.join(MODEL_SAVE_DIR, "rewards_log.csv") # Simplified log name
    csv_header = ["Episode", "Timestamp", "TotalEnvInteractions", "AvgTotalReward_Window", 
                  "AvgLoss_Window", "Epsilon", "CaptureRate_Window"]
    if not os.path.exists(csv_log_filename): 
        append_to_csv(csv_log_filename, [], header=csv_header)
    print(f"Logging training data to: {csv_log_filename}")

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    base_rewards_config = BASE_ENV_SPARSE_DEFAULTS.copy() 
    print(f"Base rewards passed to TILRawEnv (sparse from gridworld.py): {base_rewards_config}")
    
    aec_env_raw = TILRawEnv(render_mode=ENV_RENDER_MODE if ENABLE_VIDEO_RECORDING else None,
                           novice=NOVICE_MAP_TRAINING, debug=False, rewards_dict=base_rewards_config)
    aec_env_custom_rewards = CustomRewardWrapper(aec_env_raw) 
    aec_env_ordered = pz_wrappers.OrderEnforcingWrapper(aec_env_custom_rewards)
    p_env = aec_to_parallel_wrapper(aec_env_ordered) 
    print("Environment Initialized: TILRawEnv -> CustomRewardWrapper -> OrderEnforcing -> Parallel. Using Custom Rewards V3.1.")
    
    max_steps_this_run = ENV_NUM_ITERS_FROM_GRIDWORLD 
    try: 
        true_raw_env = p_env.unwrapped
        while hasattr(true_raw_env, 'unwrapped') and not isinstance(true_raw_env, TILRawEnv):
            true_raw_env = true_raw_env.unwrapped
        if hasattr(true_raw_env, 'NUM_ITERS'):
            max_steps_this_run = true_raw_env.NUM_ITERS
    except Exception:
        print(f"Warning: Could not get NUM_ITERS from env, using imported {ENV_NUM_ITERS_FROM_GRIDWORLD}")
    print(f"Using max_steps_per_episode: {max_steps_this_run}")

    _ = p_env.reset(seed=42) 
    first_agent_id = p_env.possible_agents[0]
    temp_observations_dict, _ = p_env.reset(seed=42) 
    sample_obs_for_shape = temp_observations_dict[first_agent_id]
    for k_shape, v_def_shape in [('viewcone',[[0]*VIEWCONE_WIDTH for _ in range(VIEWCONE_HEIGHT)]), 
                                 ('direction',0), ('location',[0,0]), ('step',0), ('scout',0)]:
        if k_shape not in sample_obs_for_shape: 
            sample_obs_for_shape[k_shape] = v_def_shape
    cnn_raw_example, flat_raw_example = preprocess_observation(sample_obs_for_shape, max_steps_this_run)
    CNN_CHANNELS_DIM = cnn_raw_example.shape[0]
    FLAT_FEATURES_DIM = flat_raw_example.shape[0]
    print(f"Determined CNN Channels: {CNN_CHANNELS_DIM}, Flat Features: {FLAT_FEATURES_DIM}, N_Actions: {N_ACTIONS}")

    start_episode = 1 
    current_run_env_interactions = 0
    num_players_for_decay = len(p_env.possible_agents) if p_env.possible_agents else 4

    # --- Logic for Model Loading and Epsilon Setup ---
    _model_path_to_actually_load = None 
    
    if LOAD_PREVIOUS_MODEL: 
        if OLD_MODEL_ABSOLUTE_PATH and os.path.exists(OLD_MODEL_ABSOLUTE_PATH):
            _model_path_to_actually_load = OLD_MODEL_ABSOLUTE_PATH
            print(f"Using explicit OLD_MODEL_ABSOLUTE_PATH: {_model_path_to_actually_load}")
        else:
            default_load_path = os.path.join(MODEL_SAVE_DIR, MODEL_TO_LOAD_FILENAME) # Check current save dir
            if os.path.exists(default_load_path):
                _model_path_to_actually_load = default_load_path
                print(f"Found model to load in current MODEL_SAVE_DIR: {_model_path_to_actually_load}")
            else:
                # If not in current, try constructing path relative to OLD_MODEL_ABSOLUTE_PATH's directory if it's a dir
                if OLD_MODEL_ABSOLUTE_PATH and os.path.isdir(os.path.dirname(OLD_MODEL_ABSOLUTE_PATH)):
                    fallback_load_path = os.path.join(os.path.dirname(OLD_MODEL_ABSOLUTE_PATH), MODEL_TO_LOAD_FILENAME)
                    if os.path.exists(fallback_load_path):
                        _model_path_to_actually_load = fallback_load_path
                        print(f"Found model to load in directory of OLD_MODEL_ABSOLUTE_PATH: {_model_path_to_actually_load}")
                if not _model_path_to_actually_load: # If still not found
                    print(f"WARNING: LOAD_PREVIOUS_MODEL is True, but model not found. Searched OLD_MODEL_ABSOLUTE_PATH ('{OLD_MODEL_ABSOLUTE_PATH}'), and in current MODEL_SAVE_DIR ('{default_load_path}'). Will start scratch.")
    else:
        print("LOAD_PREVIOUS_MODEL is False. Starting training from scratch.")

    # --- Determine Epsilon Parameters for Agent Initialization ---
    agent_epsilon_start_for_init = EPSILON_START_SCRATCH 
    # Default decay: 65% of total episodes for scratch run
    default_scratch_decay_episodes = int(NUM_EPISODES * 0.65) 
    agent_epsilon_decay_steps_for_init = default_scratch_decay_episodes * max_steps_this_run * num_players_for_decay
    if agent_epsilon_decay_steps_for_init <= 0: agent_epsilon_decay_steps_for_init = 1

    if _model_path_to_actually_load: # If a valid model path was determined for loading
        try:
            temp_checkpoint = torch.load(_model_path_to_actually_load, map_location=current_device)
            loaded_ep_for_decay_calc = temp_checkpoint.get('episode', 0)
            
            # Total episodes for this entire training campaign (including loaded ones) is NUM_EPISODES
            # Episodes remaining for this specific script execution:
            remaining_episodes_this_execution = NUM_EPISODES - loaded_ep_for_decay_calc
            if remaining_episodes_this_execution < 0: remaining_episodes_this_execution = 0 
            
            # Decay over 50% of the *remaining episodes in this execution*
            episodes_for_this_decay_phase = int(remaining_episodes_this_execution * 0.65)
            if episodes_for_this_decay_phase <= 0: episodes_for_this_decay_phase = 1 
                
            agent_epsilon_decay_steps_for_init = episodes_for_this_decay_phase * max_steps_this_run * num_players_for_decay
            
            saved_epsilon_in_checkpoint = temp_checkpoint.get('epsilon')
            if FORCE_EPSILON_START_ON_LOAD:
                agent_epsilon_start_for_init = EPSILON_START_CONTINUE
                print(f"  Loading model. FORCE_EPSILON_START_ON_LOAD is True. New decay starts from configured EPSILON_START_CONTINUE: ~{agent_epsilon_start_for_init:.4f}.")
            elif saved_epsilon_in_checkpoint is not None:
                agent_epsilon_start_for_init = saved_epsilon_in_checkpoint
                print(f"  Loading model. Using saved epsilon from checkpoint as new decay start: ~{agent_epsilon_start_for_init:.4f}.")
            else: 
                agent_epsilon_start_for_init = EPSILON_START_CONTINUE
                print(f"  Loading model. No epsilon in checkpoint and FORCE_EPSILON_START_ON_LOAD is False. Using EPSILON_START_CONTINUE: ~{agent_epsilon_start_for_init:.4f}.")
                
        except Exception as e_peek:
            print(f"ERROR peeking into checkpoint {_model_path_to_actually_load}: {e_peek}. Switching to scratch parameters for agent.")
            _model_path_to_actually_load = None 
            agent_epsilon_start_for_init = EPSILON_START_SCRATCH
            agent_epsilon_decay_steps_for_init = default_scratch_decay_episodes * max_steps_this_run * num_players_for_decay # Recalculate for scratch
            if agent_epsilon_decay_steps_for_init <= 0: agent_epsilon_decay_steps_for_init = 1
    
    # Initialize the agent
    shared_agent = UniversalDQNAgent(
        cnn_input_channels=CNN_CHANNELS_DIM, flat_input_size=FLAT_FEATURES_DIM, n_actions=N_ACTIONS,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY, batch_size=BATCH_SIZE, gamma=GAMMA, 
        lr=LEARNING_RATE, # Use LEARNING_RATE from global config
        target_update_freq=TARGET_UPDATE_FREQ, 
        epsilon_start=agent_epsilon_start_for_init, 
        epsilon_end=EPSILON_END, 
        epsilon_decay_steps=agent_epsilon_decay_steps_for_init, 
        device=current_device,
        max_steps_per_episode_for_preprocessing=max_steps_this_run
    )

    if _model_path_to_actually_load: 
        try:
            loaded_ep, loaded_interactions = shared_agent.load_checkpoint(
                _model_path_to_actually_load, 
                new_epsilon_start_for_continue=agent_epsilon_start_for_init, 
                new_epsilon_end=EPSILON_END,
                new_total_decay_steps=agent_epsilon_decay_steps_for_init,
                load_optimizer_state_flag=LOAD_OPTIMIZER_STATE_ON_CONTINUE # <<<< Pass global flag
            )
            start_episode = loaded_ep + 1
            current_run_env_interactions = loaded_interactions
        except Exception as e_load_final:
            print(f"WARNING: Final load attempt for {_model_path_to_actually_load} failed: {e_load_final}. Agent will use scratch parameters.")
            start_episode = 1
            current_run_env_interactions = 0
            # Ensure agent is fully reset to scratch if loading fails here
            shared_agent.optimizer = optim.AdamW(shared_agent.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
            shared_agent.epsilon_start = EPSILON_START_SCRATCH
            shared_agent.epsilon = EPSILON_START_SCRATCH
            shared_agent.epsilon_decay_steps = default_scratch_decay_episodes * max_steps_this_run * num_players_for_decay
            if shared_agent.epsilon_decay_steps <= 0: shared_agent.epsilon_decay_steps = 1
    
    print(f"Universal DQN Agent Initialized on {shared_agent.device}.")
    print(f"  Learning Rate: {shared_agent.optimizer.param_groups[0]['lr']:.1e}") # Print actual LR used by optimizer
    print(f"  Epsilon config: Effective Start for this run's decay = {shared_agent.epsilon_start:.4f} (current actual epsilon = {shared_agent.epsilon:.4f}), End={shared_agent.epsilon_end:.4f}, DecaySteps={shared_agent.epsilon_decay_steps}")

    LOGGING_WINDOW_SIZE = 100 
    PLOT_SMOOTHING_WINDOW_SIZE = max(LOGGING_WINDOW_SIZE, NUM_EPISODES // 100 if NUM_EPISODES > 0 else LOGGING_WINDOW_SIZE)

    best_avg_reward_overall = -float('inf')
    all_episode_total_rewards_agg = [] 
    all_episode_avg_losses = []        
    scout_capture_history_window = deque(maxlen=LOGGING_WINDOW_SIZE)

    print(f"\nStarting training from episode {start_episode} up to {NUM_EPISODES} total episodes...")
    if start_episode > NUM_EPISODES:
        print("Start episode is beyond total episodes or no new episodes to run. Exiting.")
        if hasattr(p_env, 'close'): p_env.close()
        return

    # ... (The rest of your training loop: for episode_idx in range(start_episode, NUM_EPISODES + 1): ...)
    # ... (No changes needed in the loop itself for this request) ...
    for episode_idx in range(start_episode, NUM_EPISODES + 1):
        frames_for_this_episode_video = []
        is_recording_this_episode = ENABLE_VIDEO_RECORDING and \
                                    current_run_env_interactions >= RECORD_VIDEO_TRIGGER_INTERACTION and \
                                    not video_has_been_recorded_this_run
        
        if is_recording_this_episode and (p_env.render_mode != "rgb_array" if hasattr(p_env,'render_mode') else True) :
            print(f"Info Ep {episode_idx}: Switching ENV_RENDER_MODE to 'rgb_array' for video capture.")
            if hasattr(p_env, 'close'): p_env.close() 
            aec_env_raw_video = TILRawEnv(render_mode="rgb_array", novice=NOVICE_MAP_TRAINING, debug=False, rewards_dict=base_rewards_config)
            aec_env_custom_rewards_video = CustomRewardWrapper(aec_env_raw_video)
            aec_env_ordered_video = pz_wrappers.OrderEnforcingWrapper(aec_env_custom_rewards_video)
            p_env = aec_to_parallel_wrapper(aec_env_ordered_video)
            if hasattr(p_env.unwrapped, 'populate_state_after_reset'): 
                 p_env.unwrapped.populate_state_after_reset()

        try:
            current_seed = episode_idx if NOVICE_MAP_TRAINING else random.randint(0, 2**32 - 1)
            observations, infos = p_env.reset(seed=current_seed) 
        except Exception as e_reset_loop:
            print(f"Error p_env.reset() ep {episode_idx}: {e_reset_loop}. Skipping episode."); 
            continue

        current_episode_agent_rewards = {agent_id: 0.0 for agent_id in p_env.possible_agents}
        episode_step_losses = [] 
        episode_scout_captured_flag_this_ep = False

        for step_num in range(max_steps_this_run): 
            if not p_env.agents: break 
            actions_to_submit = {}
            current_observations_for_step = observations.copy() 
            for agent_id in p_env.agents: 
                if agent_id not in current_observations_for_step or current_observations_for_step[agent_id] is None:
                    continue          
                obs_for_agent = current_observations_for_step[agent_id]
                if 'scout' not in obs_for_agent: obs_for_agent['scout'] = 0.0 
                action = shared_agent.select_action(obs_for_agent, is_training=True)
                actions_to_submit[agent_id] = action
            
            next_observations, rewards_this_step, terminations, truncations, infos_this_step = {}, {}, {}, {}, {}
            if not actions_to_submit and p_env.agents : 
                print(f"Warning Ep {episode_idx} Step {step_num}: Active agents {p_env.agents} but no actions generated. Breaking step loop.")
                break 
            elif not actions_to_submit and not p_env.agents: 
                break 
            if actions_to_submit: 
                try:
                    next_observations, rewards_this_step, terminations, truncations, infos_this_step = p_env.step(actions_to_submit)
                except Exception as e_step:
                    print(f"Error p_env.step() ep {episode_idx}, step {step_num}: {e_step}. Ending episode."); break
            else: 
                break
            current_run_env_interactions += len(actions_to_submit) 
            if is_recording_this_episode:
                try:
                    frame = p_env.render() 
                    if frame is not None: frames_for_this_episode_video.append(frame)
                except Exception as e_render: 
                    print(f"Warning: Render failed during video recording ep {episode_idx}: {e_render}")
                    pass 
            for agent_id_acted in actions_to_submit.keys():
                prev_obs = current_observations_for_step.get(agent_id_acted)
                if prev_obs is None : continue 
                action_taken_by_agent = actions_to_submit[agent_id_acted]
                reward_received = rewards_this_step.get(agent_id_acted, 0.0) 
                agent_terminated = terminations.get(agent_id_acted, False)
                agent_truncated = truncations.get(agent_id_acted, False)
                agent_done_this_step = agent_terminated or agent_truncated
                obs_prime_for_agent = next_observations.get(agent_id_acted) 
                shared_agent.store_experience(
                    prev_obs, action_taken_by_agent, reward_received, 
                    obs_prime_for_agent if not agent_done_this_step else None, 
                    agent_done_this_step
                )
                current_episode_agent_rewards[agent_id_acted] += reward_received
                true_raw_env_for_scout_check = p_env.unwrapped
                while hasattr(true_raw_env_for_scout_check, 'unwrapped') and not isinstance(true_raw_env_for_scout_check, TILRawEnv):
                    true_raw_env_for_scout_check = true_raw_env_for_scout_check.unwrapped
                if hasattr(true_raw_env_for_scout_check, 'scout') and \
                   agent_id_acted == true_raw_env_for_scout_check.scout and \
                   agent_terminated and step_num < max_steps_this_run -1 : 
                    episode_scout_captured_flag_this_ep = True
            observations = next_observations 
            if current_run_env_interactions >= LEARNING_STARTS_AFTER_STEPS and \
               current_run_env_interactions % LEARN_EVERY_N_STEPS == 0:
                if len(shared_agent.replay_buffer) >= BATCH_SIZE:
                    loss = shared_agent.learn()
                    if loss is not None: episode_step_losses.append(loss)
            if not p_env.agents: break 
        if is_recording_this_episode and frames_for_this_episode_video:
            timestamp_str = run_start_time.strftime("%Y%m%d_%H%M%S") 
            video_filename = os.path.join(VIDEO_SAVE_DIR,f"universal_ep_{episode_idx}_{timestamp_str}.mp4")
            try:
                imageio.mimsave(video_filename, frames_for_this_episode_video, fps=10)
                print(f"SUCCESS: Saved video {video_filename}")
                video_has_been_recorded_this_run = True 
            except Exception as e_video_save: 
                print(f"ERROR saving video {video_filename}: {e_video_save}")
        frames_for_this_episode_video.clear()
        avg_total_reward_this_episode = np.mean(list(current_episode_agent_rewards.values())) if current_episode_agent_rewards else 0.0
        all_episode_total_rewards_agg.append(avg_total_reward_this_episode)
        avg_loss_this_episode = np.mean(episode_step_losses) if episode_step_losses else None 
        all_episode_avg_losses.append(avg_loss_this_episode) 
        scout_capture_history_window.append(1 if episode_scout_captured_flag_this_ep else 0)

        if episode_idx % LOG_EVERY_N_EPISODES == 0:
            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nEpisode {episode_idx}/{NUM_EPISODES} ({current_time_str})")
            print(f"  Env Interactions: {current_run_env_interactions}, Epsilon: {shared_agent.epsilon:.4f}")
            reward_log_window = all_episode_total_rewards_agg[-LOGGING_WINDOW_SIZE:]
            avg_reward_log_period = np.mean(reward_log_window) if reward_log_window else 0.0
            loss_log_window = [l for l in all_episode_avg_losses[-LOGGING_WINDOW_SIZE:] if l is not None] 
            avg_loss_log_period = np.mean(loss_log_window) if loss_log_window else None 
            avg_capture_rate_log_window = np.mean(list(scout_capture_history_window)) if scout_capture_history_window else 0.0
            print(f"  Avg Total Reward (last {len(reward_log_window)} eps): {avg_reward_log_period:.2f}")
            if avg_loss_log_period is not None: 
                print(f"  Avg Loss (last {len(loss_log_window)} learning eps): {avg_loss_log_period:.4f}")
            print(f"  Capture Rate (last {len(scout_capture_history_window)} eps): {avg_capture_rate_log_window:.2%}")
            csv_data_row = [ episode_idx, current_time_str, current_run_env_interactions,
                f"{avg_reward_log_period:.4f}", 
                f"{avg_loss_log_period:.4f}" if avg_loss_log_period is not None else "N/A", 
                f"{shared_agent.epsilon:.4f}", 
                f"{avg_capture_rate_log_window:.4f}" ]
            append_to_csv(csv_log_filename, csv_data_row) 
            if avg_reward_log_period > best_avg_reward_overall and episode_idx > LOGGING_WINDOW_SIZE * 2: 
                best_avg_reward_overall = avg_reward_log_period
                print(f"  NEW BEST Avg Reward: {best_avg_reward_overall:.2f}! Saving best_universal_agent.pth")
                shared_agent.save_checkpoint(os.path.join(MODEL_SAVE_DIR, "universal_agent_best.pth"), episode_idx, current_run_env_interactions)

        if episode_idx > 0 and episode_idx % SAVE_MODEL_EVERY_N_EPISODES == 0:
            chk_fname = f"universal_agent_ep_{episode_idx}.pth"
            shared_agent.save_checkpoint(os.path.join(MODEL_SAVE_DIR, chk_fname), episode_idx, current_run_env_interactions)

    if hasattr(p_env, 'close'): p_env.close()
    final_chk_fname = f"universal_agent_final_ep{NUM_EPISODES}.pth"
    shared_agent.save_checkpoint(os.path.join(MODEL_SAVE_DIR, final_chk_fname), NUM_EPISODES, current_run_env_interactions)
    print("\nTraining complete. Final model checkpoint saved.")
    
    plt.figure(figsize=(12, 6)) 
    plt.subplot(1, 2, 1)
    if len(all_episode_total_rewards_agg) >= PLOT_SMOOTHING_WINDOW_SIZE and PLOT_SMOOTHING_WINDOW_SIZE > 0 : 
        valid_indices_rewards = np.arange(PLOT_SMOOTHING_WINDOW_SIZE - 1, len(all_episode_total_rewards_agg))
        smoothed_rewards = np.convolve(all_episode_total_rewards_agg, 
                                       np.ones(PLOT_SMOOTHING_WINDOW_SIZE)/PLOT_SMOOTHING_WINDOW_SIZE, 
                                       mode='valid')
        if len(smoothed_rewards) == len(valid_indices_rewards): 
             plt.plot(valid_indices_rewards, smoothed_rewards, label=f'Smoothed Avg Reward (window {PLOT_SMOOTHING_WINDOW_SIZE})')
    plt.plot(all_episode_total_rewards_agg, alpha=0.3, label='Avg Reward per Episode')
    plt.xlabel("Episode"); plt.ylabel("Avg Total Reward (Custom)"); plt.title("Episode Rewards"); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    valid_losses_indices = [i for i, l_val in enumerate(all_episode_avg_losses) if l_val is not None]
    valid_losses_values = [all_episode_avg_losses[i] for i in valid_losses_indices] 
    if valid_losses_values: 
        if len(valid_losses_values) >= PLOT_SMOOTHING_WINDOW_SIZE and PLOT_SMOOTHING_WINDOW_SIZE > 0:
            smoothed_losses = np.convolve(valid_losses_values, 
                                          np.ones(PLOT_SMOOTHING_WINDOW_SIZE)/PLOT_SMOOTHING_WINDOW_SIZE, 
                                          mode='valid')
            if smoothed_losses.size > 0: 
                start_idx_for_smoothed_loss_plot = PLOT_SMOOTHING_WINDOW_SIZE -1 
                if start_idx_for_smoothed_loss_plot < len(valid_losses_indices):
                    x_axis_smoothed_losses = valid_losses_indices[start_idx_for_smoothed_loss_plot : start_idx_for_smoothed_loss_plot + len(smoothed_losses)]
                    if len(x_axis_smoothed_losses) == len(smoothed_losses): 
                        plt.plot(x_axis_smoothed_losses, smoothed_losses, label=f'Smoothed Avg Loss (window {PLOT_SMOOTHING_WINDOW_SIZE})')
        plt.plot(valid_losses_indices, valid_losses_values, alpha=0.3, label='Avg Loss per Learning Episode')
    plt.xlabel("Episode Number (where learning occurred)"); plt.ylabel("Average Loss"); plt.title("Training Loss"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plot_fn = os.path.join(MODEL_SAVE_DIR, f"training_plots_{run_start_time.strftime('%Y%m%d_%H%M%S')}.png")
    try: 
        plt.savefig(plot_fn)
        print(f"Plots saved: {plot_fn}")
    except Exception as e_plot: 
        print(f"Could not save plots: {e_plot}")

if __name__ == '__main__':
    main()