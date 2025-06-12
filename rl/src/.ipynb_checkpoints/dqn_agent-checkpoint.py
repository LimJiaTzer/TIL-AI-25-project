# dqn_agent.py (for inference, Option 1 fix)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# --- Constants ---
VIEWCONE_HEIGHT = 7
VIEWCONE_WIDTH = 5
GRID_SIZE = 16
N_ACTIONS = 5
MAX_STEPS_PER_EPISODE_DEFAULT = 100 # A default if not passed, or used by agent for its own storage

# --- Helper: Preprocess Observation ---
# Now takes max_steps_for_norm as an argument
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
    
    direction_val = obs_dict.get('direction', 0)
    location_val = obs_dict.get('location', [0, 0])
    step_val = obs_dict.get('step', 0)
    agent_is_scout_raw = obs_dict.get('scout', 0)

    direction = int(direction_val.item()) if isinstance(direction_val, np.ndarray) and direction_val.size ==1 else int(direction_val)
    if isinstance(location_val, np.ndarray) and location_val.shape==(2,): loc_x, loc_y = int(location_val[0]), int(location_val[1])
    elif isinstance(location_val, list) and len(location_val)==2: loc_x, loc_y = int(location_val[0]), int(location_val[1])
    else: loc_x, loc_y = 0,0
    step = int(step_val.item()) if isinstance(step_val, np.ndarray) and step_val.size == 1 else int(step_val)
    scout_flag = float(agent_is_scout_raw.item()) if isinstance(agent_is_scout_raw, np.ndarray) and agent_is_scout_raw.size == 1 else float(agent_is_scout_raw)

    dir_one_hot = np.zeros(4, dtype=np.float32); dir_one_hot[min(3,max(0,direction))] = 1.0
    norm_loc_x = loc_x / max(1,(GRID_SIZE-1)); norm_loc_y = loc_y / max(1,(GRID_SIZE-1))
    norm_step = step / max(1,(max_steps_for_norm-1.0)) # Use passed argument
    role_feature = np.array([scout_flag], dtype=np.float32)
    
    flat_features = [dir_one_hot, np.array([norm_loc_x]), np.array([norm_loc_y]), np.array([norm_step]), role_feature]
    return cnn_input_raw, np.concatenate(flat_features).astype(np.float32)

# --- Helper: QNetwork ---
class QNetwork(nn.Module):
    # ... (definition as you provided, it's fine)
    def __init__(self, cnn_input_channels: int, flat_input_size: int, n_actions: int):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(cnn_input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        with torch.no_grad():
            dummy_cnn_input = torch.randn(1, cnn_input_channels, VIEWCONE_HEIGHT, VIEWCONE_WIDTH)
            x = F.relu(self.bn1(self.conv1(dummy_cnn_input)))
            x = F.relu(self.bn2(self.conv2(x)))
        self.cnn_flat_size = int(np.prod(x.shape[1:]))
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


class DQNAgent:
    def __init__(self, cnn_input_channels: int, flat_input_size: int, n_actions: int,
                 device: torch.device | None = None,
                 # Added this to store for internal preprocess calls
                 max_steps_per_episode_for_preprocessing: int = MAX_STEPS_PER_EPISODE_DEFAULT
                 ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.max_steps_const_for_preprocess = max_steps_per_episode_for_preprocessing

        self.policy_net = QNetwork(cnn_input_channels, flat_input_size, n_actions).to(self.device)
        self.policy_net.eval()
        print(f"Inference DQNAgent: Initialized policy_net on {self.device}.")

    def _preprocess_and_to_tensor(self, obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_raw, flat_raw = preprocess_observation(obs_dict, self.max_steps_const_for_preprocess)
        cnn_tensor = torch.from_numpy(cnn_raw).unsqueeze(0).float().to(self.device)
        flat_tensor = torch.from_numpy(flat_raw).unsqueeze(0).float().to(self.device)
        return cnn_tensor, flat_tensor

    def select_action(self, obs_dict: dict) -> int:
        self.policy_net.eval()
        with torch.no_grad():
            cnn_state_tensor, flat_state_tensor = self._preprocess_and_to_tensor(obs_dict)
            q_values = self.policy_net(cnn_state_tensor, flat_state_tensor)
            return q_values.max(1)[1].item()

    def load_model_weights(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                print(f"Loaded policy_net_state_dict from checkpoint: {path}")
            else:
                self.policy_net.load_state_dict(checkpoint)
                print(f"Loaded bare state_dict into policy_net from: {path}")
            self.policy_net.eval()
            print(f"Inference DQNAgent: Model weights loaded from {path} and set to eval mode.")
        except FileNotFoundError: print(f"ERROR: Model file not found at {path}."); raise
        except Exception as e: print(f"ERROR: Failed to load model weights from {path}. Error: {e}"); raise