from __future__ import annotations

import asyncio
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Action Space
# ═══════════════════════════════════════════════════════════════════════════════
class DroneAction(str, Enum):
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    HOVER = "hover"


ACTION_LIST: List[str] = [a.value for a in DroneAction]
ACTION_TO_IDX: Dict[str, int] = {a.value: i for i, a in enumerate(DroneAction)}
IDX_TO_ACTION: Dict[int, str] = {i: a.value for i, a in enumerate(DroneAction)}


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network
# ═══════════════════════════════════════════════════════════════════════════════
class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete drone navigation."""

    def __init__(self, state_dim: int, action_dim: int = 4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling DQN: Q = V + (A - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ═══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 100_000):
        self.memory: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.memory)


# ═══════════════════════════════════════════════════════════════════════════════
# State Encoder (Drone Observations → Vector)
# ═══════════════════════════════════════════════════════════════════════════════
class StateEncoder:
    """Converts raw drone observations into a normalized state vector."""

    def __init__(self, max_distance: float = 100.0, image_width: float = 640.0, image_height: float = 480.0):
        self.max_distance = max_distance
        self.image_width = image_width
        self.image_height = image_height
        self.state_dim = 10

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """Returns a (10,) numpy array."""
        # --- Position & Target -------------------------------------------------
        pose = state.get("pose") or state.get("drone_pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        target = state.get("target") or pose

        dx = (target.get("x", 0.0) - pose.get("x", 0.0)) / self.max_distance
        dy = (target.get("y", 0.0) - pose.get("y", 0.0)) / self.max_distance
        dz = (target.get("z", 0.0) - pose.get("z", 0.0)) / self.max_distance

        # --- Velocity (if available, else zero) --------------------------------
        vel = state.get("velocity") or {"x": 0.0, "y": 0.0, "z": 0.0}
        vx = vel.get("x", 0.0) / 10.0
        vy = vel.get("y", 0.0) / 10.0
        vz = vel.get("z", 0.0) / 10.0

        # --- Obstacle Features from YOLO Detections ----------------------------
        detections: List[Dict[str, Any]] = state.get("detections", [])
        if detections:
            # Pick the most confident detection
            best = max(detections, key=lambda d: d.get("confidence", 0.0))
            bbox = best.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = bbox
            center_x = ((float(x1) + float(x2)) / 2.0) / self.image_width
            center_y = ((float(y1) + float(y2)) / 2.0) / self.image_height
            size = ((float(x2) - float(x1)) * (float(y2) - float(y1))) / (self.image_width * self.image_height)
            # Confidence as a proxy for closeness (higher conf = closer in many setups)
            obstacle_proximity = min(best.get("confidence", 0.0), 1.0)
        else:
            center_x = 0.5
            center_y = 0.5
            size = 0.0
            obstacle_proximity = 0.0

        # --- Time --------------------------------------------------------------
        step_norm = min(state.get("step", 0), 1000) / 1000.0

        vector = np.array([
            dx, dy, dz,           # 0-2 : relative target position
            vx, vy, vz,           # 3-5 : normalized velocity
            obstacle_proximity,   # 6   : 0 (none) -> 1 (very close)
            center_x,             # 7   : obstacle horizontal position in frame
            size,                 # 8   : obstacle relative size
            step_norm,            # 9   : normalized mission time
        ], dtype=np.float32)

        return np.clip(vector, -5.0, 5.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RL Agent
# ═══════════════════════════════════════════════════════════════════════════════
class RLAgent:
    """
    Production DQN agent for drone navigation.
    Supports both inference (async main loop) and training.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_path: Optional[Path] = None,
        seed: Optional[int] = None,
        state_dim: int = 10,
        action_dim: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: Optional[str] = None,
    ):
        self.logger = logger
        self.model_path = model_path
        self.state_encoder = StateEncoder()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info("RLAgent using device: %s", self.device)

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Load existing model if available
        if self.model_path and self.model_path.exists():
            self.load(self.model_path)
            self.is_dummy = False
            self.logger.info("Loaded trained model from %s", self.model_path)
        else:
            self.logger.warning("No model found at %s. Starting fresh.", self.model_path)
            self.is_dummy = True  # Will use epsilon-greedy random until trained

    # -------------------------------------------------------------------------
    # Public Async API (used by main.py)
    # -------------------------------------------------------------------------
    async def select_action(self, state: Dict[str, Any]) -> str:
        """Async action selection. Thread-safe for inference."""
        if self.is_dummy or random.random() < self.epsilon:
            action = self._get_random_action()
        else:
            # Offload inference to thread so PyTorch doesn't block the event loop
            action = await asyncio.to_thread(self._get_inference_action, state)

        self.logger.debug("RL agent selected action: %s (ε=%.3f)", action, self.epsilon)
        return action

    def _get_random_action(self) -> str:
        """Weighted random for baseline exploration."""
        weighted = [
            DroneAction.FORWARD, DroneAction.FORWARD, DroneAction.FORWARD, DroneAction.FORWARD,
            DroneAction.LEFT, DroneAction.RIGHT, DroneAction.HOVER,
        ]
        return random.choice(weighted).value

    def _get_inference_action(self, state: Dict[str, Any]) -> str:
        """Greedy action from Q-network."""
        state_vec = self.state_encoder.encode(state)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action_idx = int(q_values.argmax(dim=1).item())

        return IDX_TO_ACTION[action_idx]

    # -------------------------------------------------------------------------
    # Training API (used by train_rl.py)
    # -------------------------------------------------------------------------
    def remember(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        s = self.state_encoder.encode(state)
        a = ACTION_TO_IDX[action]
        s_next = self.state_encoder.encode(next_state)
        self.memory.push(s, a, reward, s_next, done)

    def train_step(self) -> Optional[float]:
        """Perform one gradient update. Returns loss or None if buffer too small."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # Huber loss (smooth L1)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info("Target network updated at step %s", self.train_step_count)

        return loss.item()

    def update_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        """Call before deployment to disable dropout/batch-norm updates and set epsilon low."""
        self.policy_net.eval()
        self.epsilon = 0.0  # Pure exploitation during flight
        self.is_dummy = False

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        """Save policy network and training state."""
        save_path = path or self.model_path or Path("models/dqn_drone_model.pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
        }
        torch.save(checkpoint, save_path)
        self.logger.info("Model checkpoint saved to %s", save_path)

    def load(self, path: Path) -> None:
        """Load policy network and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", 1.0)
        self.train_step_count = checkpoint.get("train_step_count", 0)
        self.policy_net.train()  # Keep in train mode until set_eval_mode() is called