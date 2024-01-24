import torch
import random
import tensorflow as tf
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

    def get_state1(self, game):
        board = np.zeros((25, 33))
        # itself
        for pt in game.snake:
            x = pt.x//20
            y = pt.y//20
            board[int(y),int(x)] = 1
        
        board[game.food.y//20,game.food.x//20] = 10
        board = np.reshape(board, (25*33,))
        return board

    def get_state(self, game):
        head = game.snake[0]
        food = game.food
        circle_straight = False
        circle_right = False
        circle_left = False
        check_x_food = False
        check_y_food = False
        # check x food
        if food.x < head.x:
            for x in range(int(food.x), int(head.x) + 1, 20):
                if game.is_collision(Point(x, head.y)):
                    check_x_food = True
                    break
        else:
            for x in range((int(head.x)), int(food.x) + 1, 20):
                if game.is_collision(Point(x, head.y)):
                    check_x_food = True
                    break
        
        # check y food
        if food.y < head.y:
            for y in range(int(food.y), int(head.y) + 1, 20):
                if game.is_collision(Point(head.x, y)):
                    check_y_food = True
                    break
        else:
            for y in range(int(head.y), int(food.y) + 1, 20):
                if game.is_collision(Point(head.x, y)):
                    check_y_food = True
                    break

        # 1 cell next
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # 2 cell next
        point_l_2 = Point(head.x - 40, head.y)
        point_r_2 = Point(head.x + 40, head.y)
        point_u_2 = Point(head.x, head.y - 40)
        point_d_2 = Point(head.x, head.y + 40)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # Danger straight 2
            (dir_r and game.is_collision(point_r_2)) or 
            (dir_l and game.is_collision(point_l_2)) or 
            (dir_u and game.is_collision(point_u_2)) or 
            (dir_d and game.is_collision(point_d_2)),

            # Danger right 2
            (dir_u and game.is_collision(point_r_2)) or 
            (dir_d and game.is_collision(point_l_2)) or 
            (dir_l and game.is_collision(point_u_2)) or 
            (dir_r and game.is_collision(point_d_2)),

            # Danger left 2
            (dir_d and game.is_collision(point_r_2)) or 
            (dir_u and game.is_collision(point_l_2)) or 
            (dir_r and game.is_collision(point_u_2)) or 
            (dir_l and game.is_collision(point_d_2)),
            
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # check x, y food
            check_x_food,
            check_y_food
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, q_values):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        move = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            
        else:
            prediction = q_values
            move = np.argmax(prediction.numpy()[0])

        return move
    
    def get_action2(self, q_values):
        # random moves: tradeoff exploration / exploitation
        move = 0
        prediction = q_values
        move = np.argmax(prediction.numpy()[0])

        return move