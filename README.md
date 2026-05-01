# ppo-cnn-snake
A PPO-based CNN with a gradient flow that evolves to play Snake on a 40x40 wrapping grid

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If you are running on an Nvidia GPU, make sure you have installed the CUDA Wheel of torch by running
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```
and you should see something like below
```
2.6.0+cu124
True
12.4
```

## Run

```bash
python snake_ppo_v7.py
```

## Versions

| Version | Key Change | Avg Score | Max Score |
|---------|-----------|-----------|-----------|
| v1 | Baseline PPO CNN - sparse reward, 40×40 from start | ~0 | 2 |
| v2 | Curriculum (10→20→40) + distance shaping | ~55 | 86 |
| v3/v4 | Fine 12-stage curriculum (8→40 in 2-row steps) | ~57 | 86 |
| v5 | Torus convolutions - fixes wrap-collision deaths | ~65 | 91 |
| v6 | Frame stacking × 4 - worse, dropped | ~47 | 86 |
| v7 | Flood fill reward + gradient body + attention + LSTM | ~52 | 95 |
| [**GA**](https://github.com/homelabdude/genetic-ai-snake) | **Hand-crafted raycasts + BFS inputs** | **~60** | **110+** |
## Read the write-up

Full devlog with results and commentary [here](https://homelabdude.com/posts/ai-learns-to-play-snake-part-2/)
