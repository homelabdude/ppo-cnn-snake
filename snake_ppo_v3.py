"""
snake_ppo_v3.py  -  PPO + CNN + Clean Curriculum
=====================================================================
Changes from v2
  - No distance shaping at all - pure sparse reward throughout
  - Revisit penalty: 0.002 * (visits**1.2) * max(1, length/5)
    Stops looping without teaching unintended movement preferences
  - 6-stage curriculum with gentler grid transitions:
      8×8 -> 10×10 -> 15×15 -> 20×20 -> 30×30 -> 40×40
  - Graduation thresholds scaled to grid size
  - Same model architecture throughout (weights transfer at each graduation)
  - Rollback checkpoints saved at every graduation

Reward    : +1.0 eat   −1.0 die   revisit penalty (scales with length)
            No distance shaping - tiny starting grid bootstraps signal naturally

Architecture (identical to v1/v2 - never changes)
  Input  : [B, 4, 40, 40]  float32  (active grid padded to full size)
  Trunk  : Conv2d 4->32->64->64  k=3 pad=1
           Flatten -> Linear(64·40·40, 512) -> ReLU
  Heads  : actor Linear(512,3)   critic Linear(512,1)

PPO      : LR=3e-5  CLIP=0.1  EPOCHS=2  MINIBATCH=512  (stable from v2)
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
FULL_CELLS = WIDTH // GRID_SIZE   # 40 - model input always this size
PANEL_W    = 260

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
ENTROPY_COEF    = 0.05
LR              = 3e-5
MAX_GRAD_NORM   = 0.5
PPO_EPOCHS      = 2
MINIBATCH_SIZE  = 512
TOTAL_STEPS     = 200_000_000
EVAL_INTERVAL   = 5
SAVE_INTERVAL   = 100
CHECKPOINT_PATH = "snake_ppo_v3_ckpt.pt"

# -- Anti-looping --------------------------------------------------------------
MAX_STEPS_NO_EAT = 200

# -- Revisit penalty -----------------------------------------------------------
REVISIT_COEF = 0.002   # base coefficient - scales with visits and length

# -- Curriculum ----------------------------------------------------------------
CURRICULUM_STAGES = [
    {"grid":  8, "graduate_score": 6.0,  "label": "Stage 0 -  8×8"},
    {"grid": 10, "graduate_score": 5.0,  "label": "Stage 1 - 10×10"},
    {"grid": 15, "graduate_score": 6.0,  "label": "Stage 2 - 15×15"},
    {"grid": 20, "graduate_score": 8.0,  "label": "Stage 3 - 20×20"},
    {"grid": 30, "graduate_score": 10.0, "label": "Stage 4 - 30×30"},
    {"grid": 40, "graduate_score": None, "label": "Stage 5 - 40×40"},
]
GRAD_WINDOW = 100


# ===============================================================================
#  1.  ENVIRONMENT
# ===============================================================================

class SnakeEnv:
    """
    Single Snake env parameterised by grid_size.
    Torus wrapping throughout all stages.
    Observation always padded to [4, FULL_CELLS, FULL_CELLS].

    Reward:
      +1.0  eat fruit
      -1.0  die (collision or starvation)
       0    otherwise, minus revisit penalty if cell was visited before
    """

    def __init__(self, grid_size: int = 40):
        self.cell_visits = None
        self.done = None
        self.score = None
        self.fruit = None
        self.steps_no_eat = None
        self.dir_idx = None
        self.length = None
        self.body = None
        self.grid_size = grid_size
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

    def reset(self):
        g  = self.grid_size
        cx = random.randint(2, g - 3)
        cy = random.randint(2, g - 3)
        self.body          = deque([(cx, cy)])
        self.length        = 2
        self.dir_idx       = random.randint(0, 3)
        self.fruit         = self._new_fruit()
        self.steps_no_eat  = 0
        self.score         = 0
        self.done          = False
        self.cell_visits   = {(cx, cy): 1}
        return self._obs()

    def step(self, action: int):
        assert not self.done, "call reset() after done"

        self.dir_idx = (self.dir_idx + (action - 1)) % 4
        dx, dy       = DIRS[self.dir_idx]
        hx, hy       = self.body[0]
        g            = self.grid_size
        new_head     = ((hx + dx) % g, (hy + dy) % g)

        # Collision check - allow tail tip (it will be removed)
        body_set = set(self.body)
        if len(self.body) >= self.length:
            body_set.discard(self.body[-1])
        if new_head in body_set:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        self.body.appendleft(new_head)
        self.steps_no_eat += 1

        # Revisit penalty - scales super-linearly with visits and snake length
        visits = self.cell_visits.get(new_head, 0)
        if visits > 0:
            penalty = REVISIT_COEF * (visits ** 1.2) * max(1.0, self.length / 5.0)
            reward  = -penalty
        else:
            reward = 0.0
        self.cell_visits[new_head] = visits + 1

        if new_head == self.fruit:
            self.length       += 1
            self.score        += 1
            self.steps_no_eat  = 0
            # Decay visit counts on eat - historical path still costs something
            self.cell_visits   = {k: max(1, int(v * 0.5))
                                  for k, v in self.cell_visits.items()}
            self.fruit         = self._new_fruit()
            reward             = 1.0   # eat overrides revisit penalty
        else:
            if len(self.body) > self.length:
                self.body.pop()

        if self.steps_no_eat > MAX_STEPS_NO_EAT:
            self.done = True
            return self._obs(), -1.0, True, {"score": self.score}

        return self._obs(), reward, False, {"score": self.score}

    def _obs(self) -> np.ndarray:
        """
        [4, FULL_CELLS, FULL_CELLS] float32.
        Active grid occupies top-left corner - consistent across all stages.
        """
        g   = self.grid_size
        obs = np.zeros((4, FULL_CELLS, FULL_CELLS), dtype=np.float32)

        hx, hy = self.body[0]
        obs[0, hy, hx] = 1.0

        for bx, by in list(self.body)[1:]:
            obs[1, by, bx] = 1.0

        fx, fy = self.fruit
        obs[2, fy, fx] = 1.0

        obs[3, :g, :g] = self.dir_idx / 3.0

        return obs


class VecSnakeEnv:
    """Synchronous vectorised wrapper over N SnakeEnv instances."""

    def __init__(self, n: int, grid_size: int):
        self.n         = n
        self.grid_size = grid_size
        self.envs      = [SnakeEnv(grid_size) for _ in range(n)]

    def set_grid_size(self, grid_size: int):
        self.grid_size = grid_size
        self.envs      = [SnakeEnv(grid_size) for _ in range(self.n)]

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
#  2.  CURRICULUM MANAGER
# ===============================================================================

class CurriculumManager:
    def __init__(self):
        self.stage        = 0
        self.score_window = deque(maxlen=GRAD_WINDOW)

    @property
    def current_stage(self):
        return CURRICULUM_STAGES[self.stage]

    @property
    def grid_size(self):
        return CURRICULUM_STAGES[self.stage]["grid"]

    @property
    def label(self):
        return CURRICULUM_STAGES[self.stage]["label"]

    @property
    def avg_score(self):
        if not self.score_window:
            return 0.0
        return sum(self.score_window) / len(self.score_window)

    def record_episode(self, score: int) -> bool:
        """Returns True if graduation just happened."""
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
#  3.  MODEL  (identical to v1/v2)
# ===============================================================================

class ActorCriticCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * FULL_CELLS * FULL_CELLS, 512), nn.ReLU(),
        )
        self.actor  = nn.Linear(512, 3)
        self.critic = nn.Linear(512, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor):
        f = self.trunk(x)
        return self.actor(f), self.critic(f).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


# ===============================================================================
#  4.  ROLLOUT BUFFER  (identical to v1/v2)
# ===============================================================================

class RolloutBuffer:
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
        self.ptr        = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones.astype(np.float32)
        self.values[self.ptr]    = values
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

    def get_minibatches(self, device):
        total = self.T * self.N
        idx   = np.random.permutation(total)
        obs_f = self.obs.reshape(total, *self.obs.shape[2:])
        act_f = self.actions.ravel()
        lp_f  = self.log_probs.ravel()
        ret_f = self.returns.ravel()
        adv_f = self.advantages.ravel()
        for start in range(0, total, MINIBATCH_SIZE):
            mb = idx[start:start + MINIBATCH_SIZE]
            yield (torch.from_numpy(obs_f[mb]).to(device),
                   torch.from_numpy(act_f[mb]).to(device),
                   torch.from_numpy(lp_f[mb]).to(device),
                   torch.from_numpy(ret_f[mb]).to(device),
                   torch.from_numpy(adv_f[mb]).to(device))

    def reset(self):
        self.ptr = 0


# ===============================================================================
#  5.  PPO TRAINER
# ===============================================================================

class PPOTrainer:
    def __init__(self, device: torch.device):
        self.device    = device
        self.model     = ActorCriticCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.curriculum = CurriculumManager()
        self.envs      = VecSnakeEnv(NUM_ENVS, self.curriculum.grid_size)
        self.buffer    = RolloutBuffer(ROLLOUT_STEPS, NUM_ENVS,
                                       (4, FULL_CELLS, FULL_CELLS))

        self.total_steps_done = 0
        self.update_count     = 0

        self.ep_score_buf  = deque(maxlen=200)
        self.ep_reward_buf = deque(maxlen=200)
        self._ep_rewards   = np.zeros(NUM_ENVS, dtype=np.float32)
        self._ep_lengths   = np.zeros(NUM_ENVS, dtype=np.int32)

        self.obs = self.envs.reset()

    def _update_lr(self):
        frac = max(0.0, 1.0 - self.total_steps_done / TOTAL_STEPS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = LR * frac

    def collect_rollout(self):
        self.model.eval()
        self.buffer.reset()
        graduated = False

        for _ in range(ROLLOUT_STEPS):
            obs_t                      = torch.from_numpy(self.obs).to(self.device)
            actions, log_probs, values = self.model.act(obs_t)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            self.buffer.add(self.obs, actions, log_probs, rewards, dones, values)
            self._ep_rewards += rewards
            self._ep_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    ep_score = infos[i]["score"]
                    self.ep_score_buf.append(ep_score)
                    self.ep_reward_buf.append(float(self._ep_rewards[i]))
                    self._ep_rewards[i] = 0
                    self._ep_lengths[i] = 0
                    if self.curriculum.record_episode(ep_score) and not graduated:
                        graduated = True

            self.obs = next_obs
            self.total_steps_done += NUM_ENVS

        if graduated:
            self._on_graduate()

        with torch.no_grad():
            _, last_values = self.model(torch.from_numpy(self.obs).to(self.device))
        self.buffer.compute_gae(last_values.cpu().numpy())

    def _on_graduate(self):
        prev_stage = self.curriculum.stage - 1
        ckpt       = f"snake_ppo_v3_ckpt_stage{prev_stage}.pt"
        self.save(ckpt)
        print(f"\n{'='*60}")
        print(f"  GRADUATION -> {self.curriculum.label}")
        print(f"  Rollback checkpoint: {ckpt}")
        print(f"{'='*60}\n")
        self.envs.set_grid_size(self.curriculum.grid_size)
        self.obs = self.envs.reset()

    def update(self):
        self.model.train()
        self._update_lr()
        tl, tpg, tent, tcf, nb = 0.0, 0.0, 0.0, 0.0, 0

        for _ in range(PPO_EPOCHS):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in \
                    self.buffer.get_minibatches(self.device):
                logits, values = self.model(obs_b)
                dist           = Categorical(logits=logits)
                new_lp         = dist.log_prob(act_b)
                entropy        = dist.entropy().mean()
                ratio          = torch.exp(new_lp - old_lp_b)
                surr1          = ratio * adv_b
                surr2          = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_b
                pg_loss        = -torch.min(surr1, surr2).mean()
                v_loss         = 0.5 * (values - ret_b).pow(2).mean()
                loss           = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

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
        }, path)
        print(f"  [ckpt -> {path}]")

    def load(self, path: str = CHECKPOINT_PATH):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps_done    = ckpt.get("total_steps", 0)
        self.update_count        = ckpt.get("updates", 0)
        self.curriculum.stage    = ckpt.get("curriculum_stage", 0)
        self.envs.set_grid_size(self.curriculum.grid_size)
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

    def step(self):
        obs_t = torch.from_numpy(self.obs[None]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
        action = int(logits.argmax(dim=-1).item())
        self.obs, reward, self.done, info = self.env.step(action)
        if reward == 1.0:
            self.score = info["score"]

    def render(self, surface, cell_px: int):
        env = self.env
        pygame.draw.rect(surface, (50, 50, 70),
                         (0, 0, env.grid_size * cell_px,
                          env.grid_size * cell_px), 1)
        fx, fy = env.fruit
        pygame.draw.circle(surface, RED,
                           (fx * cell_px + cell_px // 2,
                            fy * cell_px + cell_px // 2),
                           max(3, cell_px // 2))
        for idx, (bx, by) in enumerate(env.body):
            px = bx * cell_px + cell_px // 2
            py = by * cell_px + cell_px // 2
            if idx == 0:
                pygame.draw.circle(surface, GREEN, (px, py),
                                   max(4, int(cell_px / 1.6)))
                ddx, ddy = DIRS[env.dir_idx]
                eye_off  = max(2, cell_px // 4)
                eye_r    = max(1, cell_px // 4)
                if ddx == 0:
                    pygame.draw.circle(surface, BLACK, (px + eye_off, py), eye_r)
                    pygame.draw.circle(surface, BLACK, (px - eye_off, py), eye_r)
                else:
                    pygame.draw.circle(surface, BLACK, (px, py + eye_off), eye_r)
                    pygame.draw.circle(surface, BLACK, (px, py - eye_off), eye_r)
            else:
                t   = idx / max(len(env.body), 1)
                col = (
                    int(10  + (1-t) * (BLUE[0]-10)),
                    int(100 + (1-t) * (BLUE[1]-100)),
                    int(200 + (1-t) * (BLUE[2]-200)),
                )
                pygame.draw.circle(surface, col, (px, py),
                                   max(2, cell_px // 2))


def draw_panel(screen, trainer: PPOTrainer, metrics: dict, snake: LiveSnake):
    px = WIDTH + 8
    pygame.draw.rect(screen, PANEL_COLOR, (px, 0, PANEL_W, HEIGHT + 8))
    pygame.draw.line(screen, ACCENT, (px, 0), (px, HEIGHT + 8), 1)

    f_title = pygame.font.SysFont("consolas", 14, bold=True)
    f_body  = pygame.font.SysFont("consolas", 13)
    f_small = pygame.font.SysFont("consolas", 11)

    y = 16
    screen.blit(f_title.render("SNAKE  PPO  v3", True, ACCENT), (px + 12, y))
    y += 24
    pygame.draw.line(screen, (40,40,60), (px+8, y), (px+PANEL_W-8, y), 1)
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
    avg_sc = f"{sum(ep_sc)/len(ep_sc):.2f}" if ep_sc else "-"
    max_sc = f"{max(ep_sc)}"                if ep_sc else "-"
    thresh = cur.current_stage["graduate_score"]
    thr_s  = f"{thresh:.1f}" if thresh else "-"

    row("Stage",       cur.label,                              ORANGE)
    row("Grid",        f"{cur.grid_size}×{cur.grid_size}",     ORANGE)
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

    # Graduation progress bar
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

    # Score sparkline with graduation threshold line
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

    env   = SnakeEnv(g)
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

        snake.step()

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
        ans = input(f"Found {CHECKPOINT_PATH} - load? [y/N] ").strip().lower()
        if ans == "y":
            trainer.load()

    pygame.init()
    total_w = WIDTH + 8 + PANEL_W
    screen  = pygame.display.set_mode((total_w, HEIGHT + 8))
    pygame.display.set_caption("Snake - PPO CNN v3 (Clean Curriculum)")
    surface = pygame.Surface((WIDTH + 8, HEIGHT + 8)).convert()
    clock   = pygame.time.Clock()

    speed_idx    = [1]
    t0           = time.time()

    stages_str = " -> ".join(str(s["grid"]) for s in CURRICULUM_STAGES)
    print(f"\nSnake PPO v3  |  envs={NUM_ENVS}  T={ROLLOUT_STEPS}  "
          f"batch={NUM_ENVS*ROLLOUT_STEPS:,}  device={device}")
    print(f"Curriculum : {stages_str}  (performance-based, window={GRAD_WINDOW})")
    print(f"Reward     : sparse +1/-1  +  revisit penalty (no distance shaping)")
    print(f"Conv 4->32->64->64  Flatten  Linear->512  Actor(3)+Critic(1)")
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