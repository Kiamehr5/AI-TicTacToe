import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os 

# defining the Neural Network model for Q learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Tic Tac Toe game environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None  # 1 for player X, -1 for AI O, 0 for draw

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()

    def is_full(self):
        return not np.any(self.board == 0)

    def check_win(self, player):
        for row in self.board:
            if np.all(row == player):
                return True
        for col in self.board.T:
            if np.all(col == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def take_action(self, action, player):
        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = player
            if self.check_win(player):
                self.done = True
                self.winner = player
            elif self.is_full():
                self.done = True
                self.winner = 0
            return True
        return False

    def get_state(self):
        return self.board.flatten()

    def get_valid_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def render(self):
        print(self.board)

#training 
def train_model(model, optimizer, episodes=10000, epsilon=0.1, gamma=0.9, batch_size=64):
    env = TicTacToe()
    experience_replay = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # epsilon greedy action selection
            if random.random() < epsilon:
                valid_actions = env.get_valid_actions()
                action = random.choice(valid_actions)
            else:
                # get Q values for the valid actions
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_tensor).squeeze(0)  # get the Q values for all actions
                
                valid_actions = env.get_valid_actions()
                valid_q_values = q_values[valid_actions]  # filter Q values to only valid actions
                
                # ensure valid_q_values is not empty
                if len(valid_q_values) == 0:
                    raise ValueError("No valid actions available.")
                
                action = valid_actions[torch.argmax(valid_q_values).item()]
            
            #  get next state and reward
            player = 1 if len(experience_replay) % 2 == 0 else -1  # swap between player and AI
            valid_action_taken = env.take_action(action, player)
            
            if not valid_action_taken:
                continue  # skipping invalid actions 
            
            next_state = env.get_state()
            reward = 1 if env.winner == player else -1 if env.winner == -player else 0
            done = env.done
            
            experience_replay.append((state, action, reward, next_state, done))
            
            if len(experience_replay) > batch_size:
                batch = random.sample(experience_replay, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # convert to tensor
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                
                # Compute Q-values
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state

# function to play against the trained AI
def play_against_ai(model):
    env = TicTacToe()
    print("You are 'X' and the AI is 'O'.")
    
    
    with torch.no_grad():
        state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        valid_actions = env.get_valid_actions()
        q_values = q_values[0, valid_actions]
        action = valid_actions[torch.argmax(q_values).item()]
    
    env.take_action(action, -1)  # AI move
    print("AI makes the first move:")
    env.render()

    
    while not env.done:
    
        action = int(input("Enter your move (0-8): "))
        while not env.take_action(action, 1):
            action = int(input("Invalid move. Enter again (0-8): "))
        
    
        env.render()
        
        if env.done:
            print("You win!" if env.winner == 1 else "It's a draw!")
            break
        
    
        with torch.no_grad():
            state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            valid_actions = env.get_valid_actions()
            q_values = q_values[0, valid_actions]
            action = valid_actions[torch.argmax(q_values).item()]
        
        env.take_action(action, -1)  # AI move
        print("AI makes its move:")
        env.render()  
        
        if env.done:
            print("AI wins!" if env.winner == -1 else "It's a draw!")
            break


# initialize q network 
state_size = 9  # 3x3 board flattened
action_size = 9  # 9 possible actions
model = QNetwork(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# check if model file exists (if does not exist will take some time to train) (so it means if you have trained the model before and you're rerunning )
model_file = 'tic_tac_toe_ai.pth'

if os.path.exists(model_file):
    print("Loading trained model...")
    model.load_state_dict(torch.load(model_file))
    model.eval()  # evaluation mode
else:
    print("Training the model...")
    # Train the AI model
    train_model(model, optimizer, episodes=10000)
    torch.save(model.state_dict(), model_file)
    print(f"Model trained and saved to {model_file}")

#func to play against ai 
play_against_ai(model)
