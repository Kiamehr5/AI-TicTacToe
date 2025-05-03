import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# --- Game Environment ---
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def get_valid_actions(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]

    def take_action(self, action, player):
        if self.done:
            return False  # Don't allow moves if game is already over

        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = player
            if self.check_win(player):
                self.done = True
                self.winner = player
            elif self.is_full():
                self.done = True
                self.winner = 0  # Draw
            return True
        return False

    def check_win(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == player and self.board[1, 1] == player and self.board[2, 2] == player:
            return True
        if self.board[0, 2] == player and self.board[1, 1] == player and self.board[2, 0] == player:
            return True
        return False

    def is_full(self):
        return not (self.board == 0).any()

# --- Neural Network ---
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Training ---
def train(model, episodes=50000):
    env = TicTacToe()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9997
    epsilon_min = 0.1

    for episode in range(episodes):
        env.reset()
        state = torch.tensor(env.get_state(), dtype=torch.float32)
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice(env.get_valid_actions())
            else:
                with torch.no_grad():
                    q_values = model(state)
                    valid_actions = env.get_valid_actions()
                    q_values = q_values[valid_actions]
                    action = valid_actions[torch.argmax(q_values).item()]

            prev_state = state.clone()
            valid = env.take_action(action, 1)
            next_state = torch.tensor(env.get_state(), dtype=torch.float32)

            if env.done:
                if env.winner == 1:
                    reward = 1
                elif env.winner == 0:
                    reward = 0.5
                else:
                    reward = -1
            else:
                reward = 0

            with torch.no_grad():
                target = model(prev_state).clone()
                if env.done:
                    target[action] = reward
                else:
                    future_q = model(next_state)
                    target[action] = reward + gamma * torch.max(future_q)

            output = model(prev_state)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            done = env.done

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        

# --- AI Move ---
def ai_move(model, env):
    with torch.no_grad():
        state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        valid_actions = env.get_valid_actions()
        q_values = q_values[0, valid_actions]
        action = valid_actions[torch.argmax(q_values).item()]
    return action

# --- GUI ---
class TicTacToeGUI:
    def __init__(self, model):
        self.env = TicTacToe()
        self.model = model
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - Q Learning AI")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_board()

    def create_board(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.window, text='', font=('Arial', 60), width=3, height=1,
                                   command=lambda row=i, col=j: self.player_move(row, col))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

    def player_move(self, row, col):
        if self.env.done:
            return
        action = row * 3 + col
        if self.env.board[row, col] == 0:
            self.env.take_action(action, 1)
            self.update_board()
            if not self.env.done:
                self.window.after(500, self.ai_move)
            else:
                self.check_game_result()

    def ai_move(self):
        action = ai_move(self.model, self.env)
        self.env.take_action(action, -1)
        self.update_board()
        if self.env.done:
            self.check_game_result()

    def update_board(self):
        for i in range(3):
            for j in range(3):
                value = self.env.board[i, j]
                if value == 1:
                    self.buttons[i][j]['text'] = 'X'
                    self.buttons[i][j]['fg'] = 'blue'
                elif value == -1:
                    self.buttons[i][j]['text'] = 'O'
                    self.buttons[i][j]['fg'] = 'red'
                else:
                    self.buttons[i][j]['text'] = ''

    def check_game_result(self):
        if self.env.winner == 1:
            self.show_result("You win!")
        elif self.env.winner == -1:
            self.show_result("AI wins!")
        else:
            self.show_result("It's a draw!")

    def show_result(self, message):
        result = tk.Label(self.window, text=message, font=('Arial', 24))
        result.grid(row=3, column=0, columnspan=3)

    def run(self):
        self.window.mainloop()

model_file = 'tic_tac_toe_ai.pth'

if os.path.exists(model_file):
    model = QNetwork()
    print("Loading trained model...")
    model.load_state_dict(torch.load(model_file))
    model.eval()  # evaluation mode
else:
    model = QNetwork()
    print("Training the model...")
    # Train the AI model
    train(model)
    torch.save(model.state_dict(), model_file)
    print(f"Model trained and saved to {model_file}")

# --- Run Everything ---
if __name__ == "__main__":
    gui = TicTacToeGUI(model)
    gui.run()
