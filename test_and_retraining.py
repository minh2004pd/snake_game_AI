import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Input
from keras.losses import MSE
from keras.optimizers import Adam
from game import SnakeGameAI, Direction, Point
from agent import Agent
import helper

tf.random.set_seed(helper.SEED)

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

state_size = 16
num_actions = 3

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

# Create the Q-Network
q_network = keras.saving.load_model('snake1.keras')

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),  
    Dense(256, activation='relu'),
    Dense(256, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(num_actions, activation = 'linear'),
    ### END CODE HERE ###
    ])

### START CODE HERE ### 
optimizer = Adam(learning_rate = ALPHA)
### END CODE HERE ###

def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ### 
    y_targets = rewards + gamma * (max_qsa * (1-done_vals))
    ### END CODE HERE ###
    
    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    ### START CODE HERE ### 
    loss = MSE(y_targets, q_values)
    ### END CODE HERE ### 
    
    return loss

@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """
    
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    helper.update_target_network(q_network, target_q_network)

def train():
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1500

    total_point_history = []
    record = 0

    num_p_av = 100 # number of total point to use for averaging
    epsilon = 1.0 # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    agent = Agent()
    game = SnakeGameAI()

    for i in range(num_episodes):
        
        # reset the environment to the initial state and get the inital state
        game.reset()
        state = agent.get_state(game)
        total_points = 0
        t = 0
        
        while True:
            
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            
            # get move
            action = agent.get_action2(q_values)

            # perform move and get new state
            reward, done, score = game.play_step(action)
            next_state = agent.get_state(game)
            
            memory_buffer.append(experience(state, action, reward, next_state, done))
            
            update = helper.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                
                experiences = helper.get_experiences(memory_buffer)
                
                agent_learn(experiences, GAMMA)
            
            state = next_state.copy()
            total_points += reward
            t += 1
            
            if done:
                game.reset()
                agent.n_games += 1
                if score > record:
                    record = score
                    
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                break

    #             plot_scores.append(score)
    #             total_score += score
    #             mean_score = total_score / agent.n_games
    #             plot_mean_scores.append(mean_score)
    #             plot(plot_scores, plot_mean_scores)
        
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 800:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            q_network.save('snake.h5')
            break

    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

if __name__ == '__main__':
    train()