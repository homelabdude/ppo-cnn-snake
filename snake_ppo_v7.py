"""
snake_ppo_v7.py  —  PPO + Torus CNN + Strided Attention + LSTM (truncated BPTT)
=====================================================================
  - Strided self-attention after CNN trunk
    CNN output [B,64,40,40] -> stride-4 conv -> [B,64,10,10]
    Multi-head self-attention (4 heads) over 100 spatial positions
    Lets the network dynamically attend to distant grid regions
  - LSTM(512,512) with proper truncated BPTT (T_bptt=16)
    Fixes the broken independent treatment from v5 that caused
    exploding clip fractions
  - Rollout buffer extended to store LSTM hidden states
  - Minibatches are sequences of T_bptt steps, not random transitions

Architecture
  Input  : [B, 4, 40, 40]
  CNN    : TorusConv(4->32->64->64)  ->  [B, 64, 40, 40]
  Stride : Conv2d(64->64, stride=4) ->  [B, 64, 10, 10]  (100 positions)
  Attn   : MultiheadAttention(64, 4 heads) over 100 positions
  Pool   : mean pool -> Linear(64->512) -> ReLU
  LSTM   : LSTMCell(512, 512)
  Heads  : actor Linear(512,3)   critic Linear(512,1)

Reward  : +1.0 eat  −1.0 die  flood_fill Δreach  shaping·dist  −revisit
Body obs: gradient encoding (tail~0, near-head~1)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -- Display constants ---------------------------------------------------------
WIDTH      = 600
HEIGHT     = 600
GRID_SIZE  = 15
FULL_CELLS = WIDTH // GRID_SIZE   # 40
PANEL_W    = 260
LSTM_SIZE  = 512
ATTN_DIM   = 64     # CNN output channels = attention dim
ATTN_HEADS = 4
STRIDE     = 4      # spatial downsampling factor
T_BPTT     = 16     # truncated BPTT sequence length

# -- Colours -------------------------------------------------------------------
GREY        = (30,  30,  30)
DARK_GREY   = (20,  20,  20)
RED         = (255, 60,  60)
GREEN       = (0,   220, 80)
BLUE        = (10,  127, 255)
BLACK       = (0,   0,   0)
OFF_WHITE   = (200, 200, 200)
YELLOW      = (255, 215, 0)
PANEL_COLOR = (15,  15,  25)
ACCENT      = (0,   180, 255)
ORANGE      = (255, 140,  0)

# -- Directions ----------------------------------------------------------------
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, RIGHT, DOWN, LEFT]

# -- PPO hyperparameters -------------------------------------------------------
NUM_ENVS        = 128
ROLLOUT_STEPS   = 256
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.1
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.08
LR              = 1e-5
MAX_GRAD_NORM   = 0.5
PPO_EPOCHS      = 2
MINIBATCH_SEQS  = 32    # number of sequences per minibatch (each T_BPTT long)
TOTAL_STEPS     = 200_000_000
EVAL_INTERVAL   = 5
SAVE_INTERVAL   = 100
CHECKPOINT_PATH = "snake_ppo_v7_ckpt.pt"

# -- Reward coefficients -------------------------------------------------------
REVISIT_COEF = 0.002
FLOOD_COEF   = 0.02

# -- Curriculum ----------------------------------------------------------------
CURRICULUM_STAGES = [
    {"grid":  8, "graduate_score":  5.0, "shaping": 0.000, "label": "Stage 0 —  8×8"},
    {"grid": 10, "graduate_score":  5.0, "shaping": 0.000, "label": "Stage 1 — 10×10"},
    {"grid": 12, "graduate_score":  5.0, "shaping": 0.005, "label": "Stage 2 — 12×12"},
    {"grid": 15, "graduate_score":  6.0, "shaping": 0.005, "label": "Stage 3 — 15×15"},
    {"grid": 18, "graduate_score":  6.0, "shaping": 0.004, "label": "Stage 4 — 18×18"},
    {"grid": 20, "graduate_score":  7.0, "shaping": 0.004, "label": "Stage 5 — 20×20"},
    {"grid": 22, "graduate_score":  7.0, "shaping": 0.004, "label": "Stage 6 — 22×22"},
    {"grid": 25, "graduate_score":  8.0, "shaping": 0.004, "label": "Stage 7 — 25×25"},
    {"grid": 28, "graduate_score":  8.0, "shaping": 0.003, "label": "Stage 8 — 28×28"},
    {"grid": 32, "graduate_score":  9.0, "shaping": 0.003, "label": "Stage 9 — 32×32"},
    {"grid": 36, "graduate_score": 10.0, "shaping": 0.002, "label": "Stage 10 — 36×36"},
    {"grid": 40, "graduate_score": None, "shaping": 0.000, "label": "Stage 11 — 40×40"},
]
GRAD_WINDOW = 100

# LSTM activates from this stage onwards — before this, CNN+attention only
# Stage 5 = 20×20, where episodes get long enough for memory to help
USE_LSTM_FROM_STAGE = 5


# ===============================================================================
#  GEOMETRY HELPER
# ===============================================================================

def torus_dist(ax, ay, bx, by, cells):
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return min(dx, cells - dx) + min(dy, cells - dy)


# ===============================================================================
#  1.  ENVIRONMENT  (identical to v7)
# ===============================================================================

class SnakeEnv:
    def __init__(self, grid_size: int = 40, shaping_weight: float = 0.0):
        self._prev_reach = None
        self.cell_visits = None
        self.done = None
        self.score = None
        self.max_steps_no_eat = None
        self.steps_no_eat = None
        self.fruit = None
        self.length = None
        self.dir_idx = None
        self.body = None
        self.grid_size      = grid_size
        self.shaping_weight = shaping_weight
        self.reset()

    def _rand_cell(self):
        return (random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1))

    def _new_fruit(self):
        body = set(self.body)
        while True:
            c = self._rand_cell()
            if c not in body:
                return c

    def _flood_fill(self, max_depth: int = 30) -> float:
        g        = self.grid_size
        head     = self.body[0]
        body_set = set(list(self.body)[1:-1]) if len(self.body) > 1 else set()
        visited  = {head}
        queue    = deque([(head, 0)])
        while queue:
            (cx, cy), depth = queue.popleft()
            if depth >= max_depth:
                continue
            for ddx, ddy in DIRS:
                nx   = (cx + ddx) % g
                ny   = (cy + ddy) % g
                npos = (nx, ny)
                if npos not in visited and npos not in body_set:
                    visited.add(npos)
                    queue.append((npos, depth + 1))
        return len(visited) / (g * g)

    def reset(self):
        g  = self.grid_size
        cx = random.randint(2, g - 3)
        cy = random.randint(2, g - 3)
        self.body             = deque([(cx, cy)])
        self.length           = 2
        self.dir_idx          = random.randint(0, 3)
        self.fruit            = self._new_fruit()
        self.steps_no_eat     = 0
        self.max_steps_no_eat = max(200, g * g // 4)
        self.score            = 0
        self.done             = False
        self.cell_visits      = {(cx, cy): 1}
        self._prev_reach      = self._flood_fill()
        return self._obs()

    def step(self, action: int):
        assert not self.done
        self.dir_idx = (self.dir_idx + (action - 1)) % 4
        dx, dy       = DIRS[self.dir_idx]
        hx, hy       = self.body[0]
        g            = self.grid_size
        new_head     = ((hx + dx) % g, (hy + dy) % g)

        body_set = set(self.body)
        if len(self.body) >= self.length:
            body_set.discard(self.body[-1])
        if new_head in body_set:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        self.body.appendleft(new_head)
        self.steps_no_eat += 1

        visits = self.cell_visits.get(new_head, 0)
        if visits > 0:
            penalty = REVISIT_COEF * (visits ** 1.2) * max(1.0, self.length / 5.0)
            reward  = -penalty
        else:
            reward = 0.0
        self.cell_visits[new_head] = visits + 1

        if self.shaping_weight > 0.0:
            old_d   = torus_dist(hx, hy, self.fruit[0], self.fruit[1], g)
            new_d   = torus_dist(new_head[0], new_head[1],
                                 self.fruit[0], self.fruit[1], g)
            reward += self.shaping_weight * (old_d - new_d)

        if new_head == self.fruit:
            self.length       += 1
            self.score        += 1
            self.steps_no_eat  = 0
            self.cell_visits   = {k: max(1, int(v * 0.5))
                                  for k, v in self.cell_visits.items()}
            self.fruit         = self._new_fruit()
            self._prev_reach   = self._flood_fill()
            return self._obs(), 1.0, False, {"score": self.score}
        else:
            if len(self.body) > self.length:
                self.body.pop()

        new_reach        = self._flood_fill()
        reward          += FLOOD_COEF * (new_reach - self._prev_reach)
        self._prev_reach = new_reach

        if self.steps_no_eat > self.max_steps_no_eat:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        return self._obs(), reward, False, {"score": self.score}

    def _obs(self) -> np.ndarray:
        g   = self.grid_size
        obs = np.zeros((4, FULL_CELLS, FULL_CELLS), dtype=np.float32)
        hx, hy    = self.body[0]
        obs[0, hy, hx] = 1.0
        # Gradient body encoding — tail~0, near-head~1
        body_list = list(self.body)
        for i, (bx, by) in enumerate(body_list[1:], 1):
            obs[1, by, bx] = 1.0 - (i / max(len(body_list), 1))
        fx, fy = self.fruit
        obs[2, fy, fx] = 1.0
        obs[3, :g, :g] = self.dir_idx / 3.0
        return obs


class VecSnakeEnv:
    def __init__(self, n: int, grid_size: int, shaping_weight: float):
        self.n              = n
        self.grid_size      = grid_size
        self.shaping_weight = shaping_weight
        self.envs           = [SnakeEnv(grid_size, shaping_weight) for _ in range(n)]

    def set_stage(self, grid_size: int, shaping_weight: float):
        self.grid_size      = grid_size
        self.shaping_weight = shaping_weight
        self.envs           = [SnakeEnv(grid_size, shaping_weight) for _ in range(self.n)]

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        results = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        obs_list, rew_list, done_list, info_list = zip(*results)
        obs_array = np.stack(obs_list)
        for i, done in enumerate(done_list):
            if done:
                obs_array[i] = self.envs[i].reset()
        return (obs_array,
                np.array(rew_list,  dtype=np.float32),
                np.array(done_list, dtype=bool),
                list(info_list))


# ===============================================================================
#  2.  CURRICULUM MANAGER  (identical to v7)
# ===============================================================================

class CurriculumManager:
    def __init__(self):
        self.stage        = 0
        self.score_window = deque(maxlen=GRAD_WINDOW)

    @property
    def current_stage(self):  return CURRICULUM_STAGES[self.stage]
    @property
    def grid_size(self):      return self.current_stage["grid"]
    @property
    def shaping_weight(self): return self.current_stage["shaping"]
    @property
    def label(self):          return self.current_stage["label"]
    @property
    def avg_score(self):
        if not self.score_window: return 0.0
        return sum(self.score_window) / len(self.score_window)

    @property
    def use_lstm(self):
        return self.stage >= USE_LSTM_FROM_STAGE

    def record_episode(self, score: int) -> bool:
        self.score_window.append(score)
        threshold = self.current_stage["graduate_score"]
        if (threshold is not None
                and len(self.score_window) == GRAD_WINDOW
                and self.avg_score >= threshold):
            self.stage += 1
            self.score_window.clear()
            return True
        return False


# ===============================================================================
#  3.  MODEL  — Torus CNN + Strided Attention + LSTM
# ===============================================================================

class TorusConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pad  = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        p = self.pad
        x = torch.cat([x[:, :, :, -p:], x, x[:, :, :, :p]], dim=3)
        x = torch.cat([x[:, :, -p:, :], x, x[:, :, :p, :]], dim=2)
        return self.conv(x)


class ActorCriticCNN(nn.Module):
    """
    Torus CNN -> Strided Self-Attention -> LSTM -> Actor/Critic

    CNN extracts local spatial features.
    Attention allows any spatial position to attend to any other —
    the snake near a corner can attend to the open space far away.
    LSTM with proper BPTT provides temporal memory for planning.
    """

    def __init__(self):
        super().__init__()

        # Torus CNN trunk
        self.conv1 = TorusConv2d(4,  32)
        self.conv2 = TorusConv2d(32, 64)
        self.conv3 = TorusConv2d(64, 64)
        self.relu  = nn.ReLU()

        # Strided conv for spatial downsampling 40×40 -> 10×10
        self.stride_conv = nn.Conv2d(64, ATTN_DIM, kernel_size=4, stride=STRIDE)

        # Self-attention over 100 spatial positions
        self.attn     = nn.MultiheadAttention(ATTN_DIM, ATTN_HEADS, batch_first=True)
        self.attn_norm = nn.LayerNorm(ATTN_DIM)

        # Project to LSTM input size
        self.proj = nn.Linear(ATTN_DIM, LSTM_SIZE)

        # LSTM for temporal memory
        self.lstm = nn.LSTMCell(LSTM_SIZE, LSTM_SIZE)

        # Heads
        self.actor  = nn.Linear(LSTM_SIZE, 3)
        self.critic = nn.Linear(LSTM_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1.conv, self.conv2.conv, self.conv3.conv]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.stride_conv.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.stride_conv.bias)
        nn.init.orthogonal_(self.proj.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.proj.bias)
        for name, p in self.lstm.named_parameters():
            if "weight" in name: nn.init.orthogonal_(p)
            elif "bias" in name: nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def cnn_attn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN + attention trunk. Returns [B, LSTM_SIZE]."""
        # CNN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Stride -> [B, ATTN_DIM, 10, 10]
        x = self.relu(self.stride_conv(x))

        # Reshape to sequence -> [B, 100, ATTN_DIM]
        x = x.flatten(2).transpose(1, 2)

        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)

        # Mean pool -> [B, ATTN_DIM]
        x = x.mean(dim=1)

        # Project to LSTM size
        return self.relu(self.proj(x))

    def forward(self, obs: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor,
                use_lstm: bool = True):
        """
        obs      : [B, 4, 40, 40]
        h,c      : [B, LSTM_SIZE]
        use_lstm : False on early curriculum stages — CNN+attn only, stable on short episodes
        Returns logits [B,3], value [B], h' [B,L], c' [B,L]
        """
        feat = self.cnn_attn(obs)
        if use_lstm:
            h, c = self.lstm(feat, (h, c))
            out  = h
        else:
            out  = feat   # skip LSTM — h,c pass through unchanged
        return self.actor(out), self.critic(out).squeeze(-1), h, c

    def zero_state(self, n: int, device: torch.device):
        return (torch.zeros(n, LSTM_SIZE, device=device),
                torch.zeros(n, LSTM_SIZE, device=device))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
            use_lstm: bool = True):
        logits, value, h2, c2 = self(obs, h, c, use_lstm)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return (action.cpu().numpy(),
                log_prob.cpu().numpy(),
                value.cpu().numpy(),
                h2, c2)


# ===============================================================================
#  4.  ROLLOUT BUFFER  — stores LSTM hidden states for BPTT
# ===============================================================================

class RolloutBuffer:
    """
    Stores T×N transitions including LSTM hidden states.
    During update, processes sequences of T_BPTT steps using stored
    initial hidden states — proper truncated BPTT, not independent approx.
    """

    def __init__(self, T: int, N: int, obs_shape: Tuple):
        self.T, self.N  = T, N
        self.obs        = np.zeros((T, N, *obs_shape), dtype=np.float32)
        self.actions    = np.zeros((T, N), dtype=np.int64)
        self.log_probs  = np.zeros((T, N), dtype=np.float32)
        self.rewards    = np.zeros((T, N), dtype=np.float32)
        self.dones      = np.zeros((T, N), dtype=np.float32)
        self.values     = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns    = np.zeros((T, N), dtype=np.float32)
        # LSTM hidden states at each timestep
        self.h          = np.zeros((T, N, LSTM_SIZE), dtype=np.float32)
        self.c          = np.zeros((T, N, LSTM_SIZE), dtype=np.float32)
        self.ptr        = 0

    def add(self, obs, actions, log_probs, rewards, dones, values, h, c):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones.astype(np.float32)
        self.values[self.ptr]    = values
        self.h[self.ptr]         = h
        self.c[self.ptr]         = c
        self.ptr                += 1

    def compute_gae(self, last_values: np.ndarray):
        gae    = np.zeros(self.N, dtype=np.float32)
        next_v = last_values
        for t in reversed(range(self.T)):
            nnt   = 1.0 - self.dones[t]
            delta = self.rewards[t] + GAMMA * next_v * nnt - self.values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * nnt * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]
            next_v             = self.values[t]
        adv = self.advantages.ravel()
        self.advantages = (self.advantages - adv.mean()) / (adv.std() + 1e-8)

    def get_sequence_minibatches(self, device):
        """
        Yield minibatches of sequences for truncated BPTT.
        Each minibatch contains MINIBATCH_SEQS sequences of T_BPTT steps.
        Sequences are sampled from (env, time_offset) pairs.
        """
        # Valid sequence start positions
        # Sequences use non-overlapping T_BPTT-stride windows; done boundaries
        # are handled inside the update loop via the done mask.
        seq_starts = []
        for n in range(self.N):
            for t in range(0, self.T - T_BPTT + 1, T_BPTT):
                seq_starts.append((t, n))

        random.shuffle(seq_starts)

        for i in range(0, len(seq_starts), MINIBATCH_SEQS):
            batch = seq_starts[i:i + MINIBATCH_SEQS]
            if len(batch) == 0:
                continue

            # Stack sequences
            obs_seq   = []
            act_seq   = []
            lp_seq    = []
            ret_seq   = []
            adv_seq   = []
            done_seq  = []
            init_h    = []
            init_c    = []

            for t0, n in batch:
                obs_seq.append(self.obs[t0:t0+T_BPTT, n])
                act_seq.append(self.actions[t0:t0+T_BPTT, n])
                lp_seq.append(self.log_probs[t0:t0+T_BPTT, n])
                ret_seq.append(self.returns[t0:t0+T_BPTT, n])
                adv_seq.append(self.advantages[t0:t0+T_BPTT, n])
                done_seq.append(self.dones[t0:t0+T_BPTT, n])
                init_h.append(self.h[t0, n])
                init_c.append(self.c[t0, n])

            # [S, T_BPTT, ...] -> tensors
            yield (
                torch.from_numpy(np.stack(obs_seq)).to(device),    # [S,T,4,40,40]
                torch.from_numpy(np.stack(act_seq)).to(device),    # [S,T]
                torch.from_numpy(np.stack(lp_seq)).to(device),     # [S,T]
                torch.from_numpy(np.stack(ret_seq)).to(device),    # [S,T]
                torch.from_numpy(np.stack(adv_seq)).to(device),    # [S,T]
                torch.from_numpy(np.stack(done_seq)).to(device),   # [S,T]
                torch.from_numpy(np.stack(init_h)).to(device),     # [S,L]
                torch.from_numpy(np.stack(init_c)).to(device),     # [S,L]
            )

    def reset(self):
        self.ptr = 0


# ===============================================================================
#  5.  PPO TRAINER
# ===============================================================================

class PPOTrainer:
    def __init__(self, device: torch.device):
        self.device     = device
        self.model      = ActorCriticCNN().to(device)
        self.optimizer  = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.curriculum = CurriculumManager()
        self.envs       = VecSnakeEnv(NUM_ENVS,
                                      self.curriculum.grid_size,
                                      self.curriculum.shaping_weight)
        self.buffer     = RolloutBuffer(ROLLOUT_STEPS, NUM_ENVS,
                                        (4, FULL_CELLS, FULL_CELLS))

        self.total_steps_done = 0
        self.update_count     = 0

        self.ep_score_buf  = deque(maxlen=200)
        self.ep_reward_buf = deque(maxlen=200)
        self._ep_rewards   = np.zeros(NUM_ENVS, dtype=np.float32)

        # LSTM hidden state per env
        self.lstm_h, self.lstm_c = self.model.zero_state(NUM_ENVS, device)

        self.obs = self.envs.reset()

    def _update_lr(self):
        frac = max(0.0, 1.0 - self.total_steps_done / TOTAL_STEPS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = LR * frac

    def collect_rollout(self):
        self.model.eval()
        self.buffer.reset()
        graduated = False
        use_lstm  = self.curriculum.use_lstm

        for _ in range(ROLLOUT_STEPS):
            obs_t = torch.from_numpy(self.obs).to(self.device)

            actions, log_probs, values, h2, c2 = \
                self.model.act(obs_t, self.lstm_h, self.lstm_c, use_lstm)

            # Store hidden state BEFORE update (initial state for BPTT)
            self.buffer.add(
                self.obs, actions, log_probs,
                np.zeros(NUM_ENVS, dtype=np.float32),  # rewards filled below
                np.zeros(NUM_ENVS, dtype=bool),
                values,
                self.lstm_h.cpu().numpy(),
                self.lstm_c.cpu().numpy(),
            )
            # Actually step envs
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # Patch rewards and dones into buffer
            t = self.buffer.ptr - 1
            self.buffer.rewards[t] = rewards
            self.buffer.dones[t]   = dones.astype(np.float32)

            # Zero LSTM state for done envs
            done_t = torch.from_numpy(dones).float().to(self.device)
            mask   = (1.0 - done_t).unsqueeze(1)
            self.lstm_h = h2 * mask
            self.lstm_c = c2 * mask

            self._ep_rewards += rewards

            for i, done in enumerate(dones):
                if done:
                    ep_score = infos[i]["score"]
                    self.ep_score_buf.append(ep_score)
                    self.ep_reward_buf.append(float(self._ep_rewards[i]))
                    self._ep_rewards[i] = 0
                    if self.curriculum.record_episode(ep_score) and not graduated:
                        graduated = True

            self.obs = next_obs
            self.total_steps_done += NUM_ENVS

        if graduated:
            self._on_graduate()

        with torch.no_grad():
            _, last_values, _, _ = self.model(
                torch.from_numpy(self.obs).to(self.device),
                self.lstm_h, self.lstm_c, use_lstm)
        self.buffer.compute_gae(last_values.cpu().numpy())

    def _on_graduate(self):
        prev = self.curriculum.stage - 1
        ckpt = f"snake_ppo_v7_ckpt_stage{prev}.pt"
        self.save(ckpt)
        print(f"\n{'='*60}")
        print(f"  GRADUATION -> {self.curriculum.label}")
        print(f"  Shaping weight : {self.curriculum.shaping_weight}")
        print(f"  Rollback ckpt  : {ckpt}")
        print(f"{'='*60}\n")
        self.envs.set_stage(self.curriculum.grid_size,
                            self.curriculum.shaping_weight)
        # Reset LSTM state on graduation
        self.lstm_h, self.lstm_c = self.model.zero_state(NUM_ENVS, self.device)
        self.obs = self.envs.reset()

    def update(self):
        self.model.train()
        self._update_lr()
        tl, tpg, tent, tcf, nb = 0.0, 0.0, 0.0, 0.0, 0
        use_lstm = self.curriculum.use_lstm

        for _ in range(PPO_EPOCHS):
            for (obs_b, act_b, old_lp_b, ret_b, adv_b,
                 done_b, h_b, c_b) in \
                    self.buffer.get_sequence_minibatches(self.device):

                S, T = obs_b.shape[:2]

                # Process sequence with BPTT
                h, c = h_b, c_b   # initial hidden states from rollout

                all_logits = []
                all_values = []

                for t in range(T):
                    logits, value, h, c = self.model(obs_b[:, t], h, c, use_lstm)
                    all_logits.append(logits)
                    all_values.append(value)

                    # Zero hidden state at episode boundaries
                    mask = (1.0 - done_b[:, t]).unsqueeze(1)
                    h = h * mask
                    c = c * mask

                logits_seq = torch.stack(all_logits, dim=1)   # [S,T,3]
                values_seq = torch.stack(all_values, dim=1)   # [S,T]

                # Flatten S*T for loss computation
                logits_flat = logits_seq.reshape(S*T, -1)
                values_flat = values_seq.reshape(S*T)
                act_flat    = act_b.reshape(S*T)
                old_lp_flat = old_lp_b.reshape(S*T)
                ret_flat    = ret_b.reshape(S*T)
                adv_flat    = adv_b.reshape(S*T)

                dist     = Categorical(logits=logits_flat)
                new_lp   = dist.log_prob(act_flat)
                entropy  = dist.entropy().mean()
                ratio    = torch.exp(new_lp - old_lp_flat)
                surr1    = ratio * adv_flat
                surr2    = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_flat
                pg_loss  = -torch.min(surr1, surr2).mean()
                v_loss   = 0.5 * (values_flat - ret_flat).pow(2).mean()
                loss     = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                tl  += loss.item();    tpg += pg_loss.item()
                tent+= entropy.item(); tcf += ((ratio-1).abs() > CLIP_EPS).float().mean().item()
                nb  += 1

        self.update_count += 1
        return {"loss": tl/nb, "pg_loss": tpg/nb,
                "entropy": tent/nb, "clip_frac": tcf/nb}

    def save(self, path: str = CHECKPOINT_PATH):
        torch.save({
            "model":            self.model.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
            "total_steps":      self.total_steps_done,
            "updates":          self.update_count,
            "curriculum_stage": self.curriculum.stage,
            "lstm_h":           self.lstm_h.cpu(),
            "lstm_c":           self.lstm_c.cpu(),
        }, path)
        print(f"  [ckpt -> {path}]")

    def load(self, path: str = CHECKPOINT_PATH):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps_done    = ckpt.get("total_steps", 0)
        self.update_count        = ckpt.get("updates", 0)
        self.curriculum.stage    = ckpt.get("curriculum_stage", 0)
        self.lstm_h              = ckpt.get("lstm_h",
            torch.zeros(NUM_ENVS, LSTM_SIZE)).to(self.device)
        self.lstm_c              = ckpt.get("lstm_c",
            torch.zeros(NUM_ENVS, LSTM_SIZE)).to(self.device)
        self.envs.set_stage(self.curriculum.grid_size,
                            self.curriculum.shaping_weight)
        self.obs = self.envs.reset()
        print(f"  [ckpt ← {path}  steps={self.total_steps_done:,}  "
              f"stage={self.curriculum.stage}]")


# ===============================================================================
#  6.  PYGAME EVAL
# ===============================================================================

class LiveSnake:
    def __init__(self, env: SnakeEnv, model: ActorCriticCNN, device: torch.device):
        self.env    = env
        self.model  = model
        self.device = device
        self.obs    = env.reset()
        self.done   = False
        self.score  = 0
        self.h, self.c = model.zero_state(1, device)

    def step(self, use_lstm: bool = True):
        obs_t = torch.from_numpy(self.obs[None]).to(self.device)
        with torch.no_grad():
            logits, _, self.h, self.c = self.model(obs_t, self.h, self.c, use_lstm)
        action = int(logits.argmax(dim=-1).item())
        self.obs, reward, self.done, info = self.env.step(action)
        if self.done:
            self.h, self.c = self.model.zero_state(1, self.device)
        if reward == 1.0:
            self.score = info["score"]

    def render(self, surface, cell_px: int):
        env = self.env
        pygame.draw.rect(surface, (50, 50, 70),
                         (0, 0, env.grid_size * cell_px,
                          env.grid_size * cell_px), 1)
        fx, fy = env.fruit
        pygame.draw.circle(surface, RED,
                           (fx*cell_px + cell_px//2, fy*cell_px + cell_px//2),
                           max(3, cell_px//2))
        for idx, (bx, by) in enumerate(env.body):
            px = bx * cell_px + cell_px // 2
            py = by * cell_px + cell_px // 2
            if idx == 0:
                pygame.draw.circle(surface, GREEN, (px, py),
                                   max(4, int(cell_px/1.6)))
                ddx, _ = DIRS[env.dir_idx]
                eo = max(2, cell_px//4)
                er = max(1, cell_px//4)
                if ddx == 0:
                    pygame.draw.circle(surface, BLACK, (px+eo, py), er)
                    pygame.draw.circle(surface, BLACK, (px-eo, py), er)
                else:
                    pygame.draw.circle(surface, BLACK, (px, py+eo), er)
                    pygame.draw.circle(surface, BLACK, (px, py-eo), er)
            else:
                t   = idx / max(len(env.body), 1)
                col = (int(10  + (1-t)*(BLUE[0]-10)),
                       int(100 + (1-t)*(BLUE[1]-100)),
                       int(200 + (1-t)*(BLUE[2]-200)))
                pygame.draw.circle(surface, col, (px, py), max(2, cell_px//2))


def draw_panel(screen, trainer: PPOTrainer, metrics: dict, snake: LiveSnake):
    px = WIDTH + 8
    pygame.draw.rect(screen, PANEL_COLOR, (px, 0, PANEL_W, HEIGHT + 8))
    pygame.draw.line(screen, ACCENT, (px, 0), (px, HEIGHT + 8), 1)

    f_title = pygame.font.SysFont("consolas", 14, bold=True)
    f_body  = pygame.font.SysFont("consolas", 13)
    f_small = pygame.font.SysFont("consolas", 11)

    y = 16
    screen.blit(f_title.render("SNAKE  PPO  v7", True, ACCENT), (px+12, y))
    y += 24
    pygame.draw.line(screen, (40,40,60), (px+8,y), (px+PANEL_W-8,y), 1)
    y += 10

    def row(label, val, color=OFF_WHITE):
        nonlocal y
        lbl = f_body.render(label, True, (120,120,140))
        v   = f_body.render(str(val), True, color)
        screen.blit(lbl, (px+12, y))
        screen.blit(v,   (px+PANEL_W-v.get_width()-12, y))
        y += 20

    cur    = trainer.curriculum
    ep_sc  = list(trainer.ep_score_buf)
    avg_sc = f"{sum(ep_sc)/len(ep_sc):.2f}" if ep_sc else "—"
    max_sc = f"{max(ep_sc)}"                if ep_sc else "—"
    thresh = cur.current_stage["graduate_score"]
    thr_s  = f"{thresh:.1f}" if thresh else "—"

    row("Stage",       cur.label,                              ORANGE)
    row("Grid",        f"{cur.grid_size}×{cur.grid_size}",     ORANGE)
    row("Shaping",     f"{cur.shaping_weight:.3f}",            YELLOW)
    row("Grad target", thr_s,                                  YELLOW)
    row("Window avg",  f"{cur.avg_score:.2f}",                 YELLOW)
    y += 4
    row("Updates",     trainer.update_count,                   YELLOW)
    row("Steps",       f"{trainer.total_steps_done/1e6:.2f}M", YELLOW)
    row("Eval score",  snake.score,                            GREEN)
    row("Avg score",   avg_sc)
    row("Max score",   max_sc,                                 GREEN)
    row("Loss",        f"{metrics.get('loss',0):.4f}")
    row("Entropy",     f"{metrics.get('entropy',0):.3f}",      ACCENT)
    row("Clip frac",   f"{metrics.get('clip_frac',0):.3f}")
    row("LR",          f"{trainer.optimizer.param_groups[0]['lr']:.2e}")

    y += 8
    pygame.draw.line(screen, (40,40,60), (px+8,y), (px+PANEL_W-8,y), 1)
    y += 10

    if thresh:
        screen.blit(f_small.render("GRADUATION PROGRESS", True, (100,100,120)),
                    (px+12, y))
        y += 14
        bw  = PANEL_W - 24
        bh  = 10
        pct = min(1.0, cur.avg_score / thresh)
        pygame.draw.rect(screen, (30,30,50),  (px+12, y, bw, bh))
        pygame.draw.rect(screen, ORANGE,       (px+12, y, int(bw*pct), bh))
        pygame.draw.rect(screen, (60,60,80),  (px+12, y, bw, bh), 1)
        y += bh + 8

    screen.blit(f_small.render("SCORE HISTORY", True, (100,100,120)), (px+12, y))
    y += 14
    history = list(trainer.ep_score_buf)[-40:]
    if len(history) > 1:
        gw, gh = PANEL_W - 24, 50
        gx, gy = px + 12, y
        pygame.draw.rect(screen, (25,25,40), (gx, gy, gw, gh))
        max_v = max(history) if max(history) > 0 else 1
        pts   = [(gx + int(i/(len(history)-1)*gw),
                  gy + gh - int((v/max_v)*gh))
                 for i, v in enumerate(history)]
        pygame.draw.lines(screen, ACCENT, False, pts, 1)
        if thresh and thresh <= max_v:
            ty = gy + gh - int((thresh/max_v)*gh)
            pygame.draw.line(screen, ORANGE, (gx, ty), (gx+gw, ty), 1)
        y += gh + 8

    status_color = GREEN if not snake.done else RED
    screen.blit(f_body.render("ALIVE" if not snake.done else "DIED",
                               True, status_color), (px+12, y))
    screen.blit(f_small.render("SPACE=skip  F=speed  Q=quit", True, (70,70,90)),
                (px+12, HEIGHT-20))


# ===============================================================================
#  7.  MAIN
# ===============================================================================

def run_eval_episode(trainer: PPOTrainer, screen, surface, clock,
                     metrics: dict, speed_idx_ref: List[int]) -> bool:
    SPEEDS  = [5, 10, 20, 40, 80]
    g       = trainer.curriculum.grid_size
    cell_px = WIDTH // g

    env   = SnakeEnv(g, trainer.curriculum.shaping_weight)
    snake = LiveSnake(env, trainer.model, trainer.device)
    trainer.model.eval()

    while not snake.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key   == pygame.K_q:     return True
                elif event.key == pygame.K_SPACE: return False
                elif event.key == pygame.K_f:
                    speed_idx_ref[0] = (speed_idx_ref[0]+1) % len(SPEEDS)

        snake.step(trainer.curriculum.use_lstm)

        surface.fill(GREY)
        for gx in range(0, g * cell_px, cell_px):
            for gy in range(0, g * cell_px, cell_px):
                pygame.draw.circle(surface, (40,40,40),
                                   (gx+cell_px//2, gy+cell_px//2), 1)
        snake.render(surface, cell_px)

        screen.fill(DARK_GREY)
        screen.blit(surface, (0, 0))
        draw_panel(screen, trainer, metrics, snake)
        pygame.display.flip()
        clock.tick(SPEEDS[speed_idx_ref[0]] + math.ceil(snake.score/5)*2)

    pygame.time.wait(600)
    return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    trainer = PPOTrainer(device)

    if os.path.exists(CHECKPOINT_PATH):
        ans = input(f"Found {CHECKPOINT_PATH} — load? [y/N] ").strip().lower()
        if ans == "y":
            trainer.load()

    pygame.init()
    total_w = WIDTH + 8 + PANEL_W
    screen  = pygame.display.set_mode((total_w, HEIGHT + 8))
    pygame.display.set_caption("Snake — PPO v7 (CNN + Attention + LSTM)")
    surface = pygame.Surface((WIDTH + 8, HEIGHT + 8)).convert()
    clock   = pygame.time.Clock()

    speed_idx    = [1]
    t0           = time.time()

    stages_str = " -> ".join(str(s["grid"]) for s in CURRICULUM_STAGES)
    print(f"\nSnake PPO v7  |  envs={NUM_ENVS}  T={ROLLOUT_STEPS}  "
          f"batch={NUM_ENVS*ROLLOUT_STEPS:,}  device={device}")
    print(f"Curriculum : {stages_str}")
    print(f"Model      : TorusCNN -> StridedAttn(100pos,4heads) -> LSTM(512) -> Actor/Critic")
    print(f"BPTT       : T_bptt={T_BPTT}  sequences, not independent transitions")
    print(f"Reward     : sparse +1/-1  +  flood fill  +  shaping  +  revisit")
    print("-" * 60)

    while trainer.total_steps_done < TOTAL_STEPS:
        trainer.collect_rollout()
        last_metrics = trainer.update()

        u      = trainer.update_count
        fps    = trainer.total_steps_done / (time.time() - t0 + 1e-9)
        ep_sc  = list(trainer.ep_score_buf)
        avg_sc = sum(ep_sc) / len(ep_sc) if ep_sc else 0.0
        max_sc = max(ep_sc)              if ep_sc else 0

        print(f"Update {u:5d} | "
              f"steps {trainer.total_steps_done/1e6:5.2f}M | "
              f"fps {fps:6.0f} | "
              f"{trainer.curriculum.label} | "
              f"lstm {'ON ' if trainer.curriculum.use_lstm else 'off'} | "
              f"shaping {trainer.curriculum.shaping_weight:.3f} | "
              f"score avg {avg_sc:5.2f} max {max_sc:3d} | "
              f"loss {last_metrics['loss']:.4f} | "
              f"ent {last_metrics['entropy']:.3f} | "
              f"clip {last_metrics['clip_frac']:.3f}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                trainer.save(); pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                trainer.save(); pygame.quit(); sys.exit()

        if u % EVAL_INTERVAL == 0:
            if run_eval_episode(trainer, screen, surface, clock,
                                last_metrics, speed_idx):
                trainer.save(); pygame.quit(); sys.exit()

        if u % SAVE_INTERVAL == 0:
            trainer.save()

    print("\nTraining complete.")
    trainer.save()
    pygame.quit()


if __name__ == "__main__":
    main()