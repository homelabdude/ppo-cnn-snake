"""
snake_ppo.py  -  PPO + CNN for Snake
=====================================================================
Architecture
  Input  : [B, 4, 40, 40]  float32 on GPU
             ch0  head      (single 1)
             ch1  body      (1s on body cells)
             ch2  fruit     (single 1)
             ch3  direction (whole channel = normalised dir index 0-1)
  Trunk  : 3× Conv2d(->32->64->64, k=3, pad=1) + ReLU
           Flatten -> Linear(64·40·40, 512) -> ReLU
  Heads  : actor  Linear(512, 3)  -> Relative
           critic Linear(512, 1)  -> V(s)

PPO details
  128 parallel envs  (Vectorised, pure-Python, CPU)
  Rollout  : T=256 steps  ->  batch 128×256 = 32 768 transitions
  GAE      : γ=0.99  λ=0.95
  PPO clip : ε=0.2
  Loss     : L_clip + 0.5·L_value − 0.01·L_entropy
  Optimiser: Adam lr=2.5e-4, anneal to 0 over training
  Epochs   : 4 per rollout   minibatch 2048
  Gradient clip : 0.5

Reward    : +1.0 eat fruit   −1.0 die   0 otherwise   (sparse)

Training  : headless; pygame eval episode every EVAL_INTERVAL updates
Controls  : Q quit   F cycle speed   SPACE skip current eval
"""

from __future__ import annotations

import math
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

# -- Grid / display constants --------------------------------------------------
WIDTH      = 600
HEIGHT     = 600
GRID_SIZE  = 15
CELL_COUNT = WIDTH // GRID_SIZE   # 40

PANEL_W    = 240

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

# -- Directions ----------------------------------------------------------------
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, RIGHT, DOWN, LEFT]   # index used as "current heading"

# -- PPO / training hyperparameters --------------------------------------------
NUM_ENVS        = 128
ROLLOUT_STEPS   = 256          # steps per env per rollout
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.1
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.05
LR              = 1e-4
MAX_GRAD_NORM   = 0.5
PPO_EPOCHS      = 4
MINIBATCH_SIZE  = 2048
TOTAL_STEPS     = 50_000_000   # anneal LR over this
EVAL_INTERVAL   = 20           # eval every N PPO updates
SAVE_INTERVAL   = 100          # checkpoint every N updates
CHECKPOINT_PATH = "snake_ppo_ckpt.pt"

# -- Anti-looping --------------------------------------------------------------
MAX_STEPS_NO_EAT = 200         # env resets if snake hasn't eaten in this many steps


# ===============================================================================
#  1.  ENVIRONMENT
# ===============================================================================

# Manhattan distance with torus wrapping - shortest path on a wrapping grid
def torus_dist(ax, ay, bx, by):
    dx = abs(ax - bx)
    dy = abs(ay - by)
    dx = min(dx, CELL_COUNT - dx)
    dy = min(dy, CELL_COUNT - dy)
    return dx + dy


def _rand_cell():
    return (random.randint(0, CELL_COUNT - 1),
            random.randint(0, CELL_COUNT - 1))


class SnakeEnv:
    """
    Single Snake environment.
    State returned: numpy float32 array [4, CELL_COUNT, CELL_COUNT]
    Actions       : 0=straight  1=turn-right  2=turn-left  3=turn-around (masked out)
                    (same relative-action scheme as the GA)
    Reward        : +1 eat  −1 die  0 otherwise
    """

    def __init__(self):
        self.done = None
        self.score = None
        self.steps_no_eat = None
        self.fruit = None
        self.dir_idx = None
        self.length = None
        self.body = None
        self.reset()

    # -- helpers ---------------------------------------------------------------

    def _new_fruit(self):
        body = set(self.body)
        while True:
            c = _rand_cell()
            if c not in body:
                return c

    # -- public API ------------------------------------------------------------
    def reset(self):
        cx = random.randint(2, CELL_COUNT - 3)
        cy = random.randint(2, CELL_COUNT - 3)
        self.body          = deque([(cx, cy)])
        self.length        = 2
        self.dir_idx       = random.randint(0, 3)
        self.fruit         = self._new_fruit()
        self.steps_no_eat  = 0
        self.score         = 0
        self.done          = False
        return self._obs()

    def step(self, action: int):
        """action ∈ {0,1,2} - straight/right/left"""
        assert not self.done, "call reset() after done"

        # Relative -> absolute direction
        self.dir_idx = (self.dir_idx + (action - 1)) % 4
        dx, dy       = DIRS[self.dir_idx]
        hx, hy       = self.body[0]
        new_head     = ((hx + dx) % CELL_COUNT,
                        (hy + dy) % CELL_COUNT)

        # Collision with body (allow tail tip - it will be removed)
        body_set = set(self.body)
        if len(self.body) >= self.length:
            body_set.discard(self.body[-1])   # tail is about to move

        if new_head in body_set:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        self.body.appendleft(new_head)
        self.steps_no_eat += 1

        old_dist = torus_dist(hx, hy, self.fruit[0], self.fruit[1])
        new_dist = torus_dist(new_head[0], new_head[1], self.fruit[0], self.fruit[1])
        reward = 0.01 * (old_dist - new_dist)

        if new_head == self.fruit:
            self.length       += 1
            self.score        += 1
            self.steps_no_eat  = 0
            self.fruit         = self._new_fruit()
            reward             = 1.0
        else:
            if len(self.body) > self.length:
                self.body.pop()

        if self.steps_no_eat > MAX_STEPS_NO_EAT:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        return self._obs(), reward, False, {"score": self.score}

    def _obs(self) -> np.ndarray:
        """Return [4, CELL_COUNT, CELL_COUNT] float32."""
        obs = np.zeros((4, CELL_COUNT, CELL_COUNT), dtype=np.float32)

        # ch0 head
        hx, hy = self.body[0]
        obs[0, hy, hx] = 1.0

        # ch1 body
        for bx, by in list(self.body)[1:]:
            obs[1, by, bx] = 1.0

        # ch2 fruit
        fx, fy = self.fruit
        obs[2, fy, fx] = 1.0

        # ch3 direction (uniform channel)
        obs[3, :, :] = self.dir_idx / 3.0

        return obs


class VecSnakeEnv:
    """
    Synchronous vectorised wrapper - NUM_ENVS independent SnakeEnv instances.
    All step / reset calls stay on CPU (Python); only observations are
    batched into a single numpy array for GPU transfer.
    """

    def __init__(self, n: int = NUM_ENVS):
        self.n    = n
        self.envs = [SnakeEnv() for _ in range(n)]

    def reset(self) -> np.ndarray:
        obs = [e.reset() for e in self.envs]
        return np.stack(obs)   # [N, 4, H, W]

    def step(self, actions: np.ndarray):
        """
        actions : int array [N]
        returns : obs [N,4,H,W], rewards [N], dones [N], infos list[N]
        """
        results = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        obs_list, rew_list, done_list, info_list = zip(*results)

        # Auto-reset done envs
        obs_array = np.stack(obs_list)
        for i, done in enumerate(done_list):
            if done:
                obs_array[i] = self.envs[i].reset()

        return (obs_array,
                np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=bool),
                list(info_list))


# ===============================================================================
#  2.  MODEL
# ===============================================================================

class ActorCriticCNN(nn.Module):
    """
    Shared CNN trunk -> actor head (logits) + critic head (scalar V).
    Input : [B, 4, 40, 40]
    """

    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * CELL_COUNT * CELL_COUNT, 512), nn.ReLU(),
        )

        self.actor  = nn.Linear(512, 3)   # 3 relative actions
        self.critic = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """
        obs : [N, 4, H, W]  (already on device)
        Returns action [N], log_prob [N], value [N] - all on CPU numpy.
        """
        logits, value = self(obs)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


# ===============================================================================
#  3.  ROLLOUT BUFFER
# ===============================================================================

class RolloutBuffer:
    """
    Stores T×N transitions, computes GAE advantages, and yields minibatches.
    Everything stays as numpy until get_minibatches() converts to tensors.
    """

    def __init__(self, T: int, N: int, obs_shape: Tuple):
        self.T, self.N   = T, N
        self.obs         = np.zeros((T, N, *obs_shape), dtype=np.float32)
        self.actions     = np.zeros((T, N),              dtype=np.int64)
        self.log_probs   = np.zeros((T, N),              dtype=np.float32)
        self.rewards     = np.zeros((T, N),              dtype=np.float32)
        self.dones       = np.zeros((T, N),              dtype=np.float32)
        self.values      = np.zeros((T, N),              dtype=np.float32)
        self.advantages  = np.zeros((T, N),              dtype=np.float32)
        self.returns     = np.zeros((T, N),              dtype=np.float32)
        self.ptr         = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones.astype(np.float32)
        self.values[self.ptr]    = values
        self.ptr                += 1

    def compute_gae(self, last_values: np.ndarray):
        """GAE-λ.  last_values: [N] V(s_{T+1})."""
        gae    = np.zeros(self.N, dtype=np.float32)
        next_v = last_values
        for t in reversed(range(self.T)):
            next_non_term = 1.0 - self.dones[t]
            delta         = (self.rewards[t]
                             + GAMMA * next_v * next_non_term
                             - self.values[t])
            gae           = delta + GAMMA * GAE_LAMBDA * next_non_term * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]
            next_v             = self.values[t]

        # Normalise advantages across entire rollout
        adv_flat = self.advantages.ravel()
        self.advantages = ((self.advantages - adv_flat.mean())
                           / (adv_flat.std() + 1e-8))

    def get_minibatches(self, device: torch.device):
        """Yield shuffled minibatches as tensors on `device`."""
        total = self.T * self.N
        idx   = np.random.permutation(total)

        # Flatten T×N -> total
        obs_f      = self.obs.reshape(total, *self.obs.shape[2:])
        act_f      = self.actions.ravel()
        lp_f       = self.log_probs.ravel()
        ret_f      = self.returns.ravel()
        adv_f      = self.advantages.ravel()

        for start in range(0, total, MINIBATCH_SIZE):
            mb = idx[start:start + MINIBATCH_SIZE]
            yield (
                torch.from_numpy(obs_f[mb]).to(device),
                torch.from_numpy(act_f[mb]).to(device),
                torch.from_numpy(lp_f[mb]).to(device),
                torch.from_numpy(ret_f[mb]).to(device),
                torch.from_numpy(adv_f[mb]).to(device),
            )

    def reset(self):
        self.ptr = 0


# ===============================================================================
#  4.  PPO TRAINER
# ===============================================================================

class PPOTrainer:
    def __init__(self, device: torch.device):
        self.device   = device
        self.model    = ActorCriticCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.envs     = VecSnakeEnv(NUM_ENVS)
        self.buffer   = RolloutBuffer(ROLLOUT_STEPS, NUM_ENVS,
                                      (4, CELL_COUNT, CELL_COUNT))

        self.total_steps_done = 0
        self.update_count     = 0

        # Metrics
        self.ep_reward_buf  = deque(maxlen=200)
        self.ep_length_buf  = deque(maxlen=200)
        self.ep_score_buf   = deque(maxlen=200)
        self._ep_rewards    = np.zeros(NUM_ENVS, dtype=np.float32)
        self._ep_lengths    = np.zeros(NUM_ENVS, dtype=np.int32)

        # Kick off
        self.obs = self.envs.reset()   # [N, 4, H, W]

    # -- LR annealing ----------------------------------------------------------
    def _update_lr(self):
        frac = max(0.0, 1.0 - self.total_steps_done / TOTAL_STEPS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = LR * frac

    # -- Collect one rollout ---------------------------------------------------
    def collect_rollout(self):
        self.model.eval()
        self.buffer.reset()

        for _ in range(ROLLOUT_STEPS):
            obs_t   = torch.from_numpy(self.obs).to(self.device)
            actions, log_probs, values = self.model.act(obs_t)

            next_obs, rewards, dones, infos = self.envs.step(actions)

            self.buffer.add(self.obs, actions, log_probs, rewards, dones, values)

            self._ep_rewards += rewards
            self._ep_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    self.ep_reward_buf.append(float(self._ep_rewards[i]))
                    self.ep_length_buf.append(int(self._ep_lengths[i]))
                    self.ep_score_buf.append(infos[i]["score"])
                    self._ep_rewards[i] = 0
                    self._ep_lengths[i] = 0

            self.obs = next_obs
            self.total_steps_done += NUM_ENVS

        # Bootstrap last value
        with torch.no_grad():
            _, last_values = self.model(torch.from_numpy(self.obs).to(self.device))
        self.buffer.compute_gae(last_values.cpu().numpy())

    # -- PPO update ------------------------------------------------------------
    def update(self):
        self.model.train()
        self._update_lr()

        total_loss_v, total_pg_v, total_ent_v, total_clip_frac = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for _ in range(PPO_EPOCHS):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in \
                    self.buffer.get_minibatches(self.device):

                logits, values = self.model(obs_b)
                dist           = Categorical(logits=logits)
                new_lp         = dist.log_prob(act_b)
                entropy        = dist.entropy().mean()

                ratio          = torch.exp(new_lp - old_lp_b)
                surr1          = ratio * adv_b
                surr2          = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
                pg_loss        = -torch.min(surr1, surr2).mean()
                v_loss         = 0.5 * (values - ret_b).pow(2).mean()
                loss           = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss_v   += loss.item()
                total_pg_v     += pg_loss.item()
                total_ent_v    += entropy.item()
                total_clip_frac += ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
                n_batches      += 1

        self.update_count += 1
        return {
            "loss":      total_loss_v  / n_batches,
            "pg_loss":   total_pg_v    / n_batches,
            "entropy":   total_ent_v   / n_batches,
            "clip_frac": total_clip_frac / n_batches,
        }

    # -- Save / load -----------------------------------------------------------
    def save(self, path: str = CHECKPOINT_PATH):
        torch.save({
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps_done,
            "updates":     self.update_count,
        }, path)
        print(f"  [ckpt saved -> {path}]")

    def load(self, path: str = CHECKPOINT_PATH):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps_done = ckpt.get("total_steps", 0)
        self.update_count     = ckpt.get("updates", 0)
        print(f"  [ckpt loaded ← {path}  steps={self.total_steps_done:,}]")


# ===============================================================================
#  5.  PYGAME EVAL  (single episode, rendered at human speed)
# ===============================================================================

class LiveSnake:
    """
    Thin wrapper that drives a SnakeEnv with the trained policy
    and exposes a render() method for pygame.
    """

    def __init__(self, env: SnakeEnv, model: ActorCriticCNN, device: torch.device):
        self.env    = env
        self.model  = model
        self.device = device
        self.obs    = env.reset()
        self.done   = False
        self.score  = 0

    def step(self):
        obs_t         = torch.from_numpy(self.obs[None]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
        action        = int(logits.argmax(dim=-1).item())   # greedy
        self.obs, reward, self.done, info = self.env.step(action)
        if reward == 1.0:
            self.score = info["score"]

    def render(self, surface):
        env = self.env
        # Fruit
        fx, fy = env.fruit
        pygame.draw.circle(surface, RED,
                           (fx * GRID_SIZE + GRID_SIZE // 2,
                            fy * GRID_SIZE + GRID_SIZE // 2),
                           GRID_SIZE // 2)
        # Body
        for idx, (bx, by) in enumerate(env.body):
            px = bx * GRID_SIZE + GRID_SIZE // 2
            py = by * GRID_SIZE + GRID_SIZE // 2
            if idx == 0:
                pygame.draw.circle(surface, GREEN, (px, py), int(GRID_SIZE / 1.6))
                # Eyes
                ddx, ddy = DIRS[env.dir_idx]
                if ddx == 0:  # vertical
                    pygame.draw.circle(surface, BLACK, (px + 4, py), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (px - 4, py), GRID_SIZE // 4)
                else:
                    pygame.draw.circle(surface, BLACK, (px, py + 4), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (px, py - 4), GRID_SIZE // 4)
            else:
                t   = idx / max(len(env.body), 1)
                col = (
                    int(10  + (1 - t) * (BLUE[0] - 10)),
                    int(100 + (1 - t) * (BLUE[1] - 100)),
                    int(200 + (1 - t) * (BLUE[2] - 200)),
                )
                pygame.draw.circle(surface, col, (px, py), GRID_SIZE // 2)


def draw_panel(screen, trainer: PPOTrainer, metrics: dict, snake: LiveSnake):
    px = WIDTH + 8
    pygame.draw.rect(screen, PANEL_COLOR, (px, 0, PANEL_W, HEIGHT + 8))
    pygame.draw.line(screen, ACCENT, (px, 0), (px, HEIGHT + 8), 1)

    f_title = pygame.font.SysFont("consolas", 14, bold=True)
    f_body  = pygame.font.SysFont("consolas", 13)
    f_small = pygame.font.SysFont("consolas", 11)

    y = 16
    screen.blit(f_title.render("SNAKE  PPO", True, ACCENT), (px + 12, y))
    y += 24
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    def row(label, val, color=OFF_WHITE):
        nonlocal y
        lbl = f_body.render(label, True, (120, 120, 140))
        v   = f_body.render(str(val), True, color)
        screen.blit(lbl, (px + 12, y))
        screen.blit(v,   (px + PANEL_W - v.get_width() - 12, y))
        y += 20

    ep_sc = list(trainer.ep_score_buf)
    avg_score = f"{sum(ep_sc)/len(ep_sc):.2f}" if ep_sc else "-"
    max_score = f"{max(ep_sc)}"                 if ep_sc else "-"

    row("Updates",    trainer.update_count,                    YELLOW)
    row("Total steps",f"{trainer.total_steps_done/1e6:.2f}M", YELLOW)
    row("Eval score", snake.score,                             GREEN)
    row("Avg score",  avg_score)
    row("Max score",  max_score,                               GREEN)
    row("Loss",       f"{metrics.get('loss', 0):.4f}")
    row("PG loss",    f"{metrics.get('pg_loss', 0):.4f}")
    row("Entropy",    f"{metrics.get('entropy', 0):.3f}",      ACCENT)
    row("Clip frac",  f"{metrics.get('clip_frac', 0):.3f}")
    row("LR",         f"{trainer.optimizer.param_groups[0]['lr']:.2e}")
    row("Envs",       NUM_ENVS)

    y += 8
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    # Score history sparkline
    screen.blit(f_small.render("SCORE HISTORY", True, (100, 100, 120)), (px + 12, y))
    y += 14
    history = list(trainer.ep_score_buf)[-40:]
    if len(history) > 1:
        gw, gh = PANEL_W - 24, 60
        gx, gy = px + 12, y
        pygame.draw.rect(screen, (25, 25, 40), (gx, gy, gw, gh))
        max_v = max(history) if max(history) > 0 else 1
        pts   = [(gx + int(i / (len(history) - 1) * gw),
                  gy + gh - int((v / max_v) * gh))
                 for i, v in enumerate(history)]
        if len(pts) > 1:
            pygame.draw.lines(screen, ACCENT, False, pts, 1)
        y += gh + 8

    status_color = GREEN if not snake.done else RED
    screen.blit(f_body.render("ALIVE" if not snake.done else "DIED", True, status_color),
                (px + 12, y))
    screen.blit(f_small.render("SPACE=skip  F=speed  Q=quit", True, (70, 70, 90)),
                (px + 12, HEIGHT - 20))


# ===============================================================================
#  6.  MAIN
# ===============================================================================

def run_eval_episode(trainer: PPOTrainer, screen, surface, clock,
                     metrics: dict, speed_idx_ref: List[int]) -> bool:
    """
    Play one episode visually with the current policy.
    Returns True if the user wants to quit entirely.
    """
    SPEEDS = [5, 10, 20, 40, 80]

    env   = SnakeEnv()
    snake = LiveSnake(env, trainer.model, trainer.device)

    trainer.model.eval()

    while not snake.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
                elif event.key == pygame.K_SPACE:
                    return False
                elif event.key == pygame.K_f:
                    speed_idx_ref[0] = (speed_idx_ref[0] + 1) % len(SPEEDS)

        snake.step()

        surface.fill(GREY)
        for gx in range(0, WIDTH, GRID_SIZE):
            for gy in range(0, HEIGHT, GRID_SIZE):
                pygame.draw.circle(surface, (40, 40, 40),
                                   (gx + GRID_SIZE // 2, gy + GRID_SIZE // 2), 1)
        snake.render(surface)

        screen.fill(DARK_GREY)
        screen.blit(surface, (0, 0))
        draw_panel(screen, trainer, metrics, snake)
        pygame.display.flip()

        spd = SPEEDS[speed_idx_ref[0]]
        clock.tick(spd + math.ceil(snake.score / 5) * 2)

    pygame.time.wait(600)
    return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer = PPOTrainer(device)

    # Optional: resume from checkpoint
    import os
    if os.path.exists(CHECKPOINT_PATH):
        ans = input(f"Found {CHECKPOINT_PATH} - load? [y/N] ").strip().lower()
        if ans == "y":
            trainer.load()

    # Init pygame (used only during eval)
    pygame.init()
    total_w = WIDTH + 8 + PANEL_W
    screen  = pygame.display.set_mode((total_w, HEIGHT + 8))
    pygame.display.set_caption("Snake - PPO CNN")
    surface = pygame.Surface((WIDTH + 8, HEIGHT + 8)).convert()
    clock   = pygame.time.Clock()

    speed_idx  = [1]   # mutable ref so run_eval_episode can modify it

    print(f"\nSnake PPO  |  envs={NUM_ENVS}  T={ROLLOUT_STEPS}  "
          f"batch={NUM_ENVS*ROLLOUT_STEPS:,}  device={device}")
    print(f"Conv 4->32->64->64  Flatten  Linear->512  Actor(3)+Critic(1)")
    print("-" * 60)

    t0 = time.time()

    while trainer.total_steps_done < TOTAL_STEPS:
        # -- Collect ----------------------------------------------------------
        trainer.collect_rollout()

        # -- Update -----------------------------------------------------------
        last_metrics = trainer.update()

        u   = trainer.update_count
        fps = trainer.total_steps_done / (time.time() - t0 + 1e-9)

        ep_sc  = list(trainer.ep_score_buf)
        avg_sc = sum(ep_sc) / len(ep_sc) if ep_sc else 0.0
        max_sc = max(ep_sc)              if ep_sc else 0

        print(f"Update {u:5d} | "
              f"steps {trainer.total_steps_done/1e6:5.2f}M | "
              f"fps {fps:6.0f} | "
              f"score avg {avg_sc:5.2f} max {max_sc:3d} | "
              f"loss {last_metrics['loss']:.4f} | "
              f"ent {last_metrics['entropy']:.3f} | "
              f"clip {last_metrics['clip_frac']:.3f}")

        # -- Pygame events (keep window alive between evals) ---------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                trainer.save()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                trainer.save()
                pygame.quit()
                sys.exit()

        # -- Periodic eval ------------------------------------------------
        if u % EVAL_INTERVAL == 0:
            quit_requested = run_eval_episode(
                trainer, screen, surface, clock, last_metrics, speed_idx)
            if quit_requested:
                trainer.save()
                pygame.quit()
                sys.exit()

        # -- Periodic checkpoint ------------------------------------------
        if u % SAVE_INTERVAL == 0:
            trainer.save()

    print("\nTraining complete.")
    trainer.save()
    pygame.quit()


if __name__ == "__main__":
    main()