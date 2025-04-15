# 🧠 Tic Tac Toe AI with Q-Learning (PyTorch)

This project implements a simple reinforcement learning (Q-learning) agent that learns to play Tic Tac Toe using a neural network built with PyTorch.

After training, you can play against the AI in the terminal!

---

## 📦 Features

- 3x3 Tic Tac Toe board
- Q-learning with a neural network
- Experience replay
- Epsilon-greedy strategy
- Trains over 10,000 episodes
- Playable CLI game vs. AI

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch numpy
```

### 2. Run the Game

```bash
python aitictactoe.py
```

- If no model exists, it will train over **10,000 episodes**.
- Once training is done, the model is saved to `tic_tac_toe_ai.pth`.
- Next time you run it, it will load the saved model automatically.

---

## 🎮 How to Play

After the AI makes its first move:

- You'll be prompted to enter a number from `0-8` representing a spot on the board:
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

Example:
```
Enter your move (0-8): 4
```

---

## 🧠 How It Works

- The game environment (`TicTacToe`) simulates the board, rules, and rewards.
- A Q-network approximates the best move using 3 fully connected layers.
- It uses an epsilon-greedy policy (random moves with 10% chance).
- Rewards:
  - Win = `+1`
  - Lose = `-1`
  - Draw or ongoing = `0`
- Experience replay is used to train from random past experiences for stability.

---

## 📁 Files

- `tic_tac_toe_ai.py` – main script for training and playing
- `tic_tac_toe_ai.pth` – saved trained model (auto-generated)

---

## 🛠️ Notes

- Training takes a few seconds to a couple of minutes (depending on your system).
- You can change the training duration by modifying `episodes=10000` in the code.

---

## 📜 License

This is licensed under the MIT license 
