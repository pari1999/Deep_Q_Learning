# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
from env import ContinuousMazeEnv # Importing the custom environment
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
import time
import argparse
import numpy as np

#argument parsing

parser = argparse.ArgumentParser(description='DQN Training and Testing')
parser.add_argument('--train', action='store_true', help='Train the DQN agent')
parser.add_argument('--test', action='store_true', help='Test the DQN agent')
parser.add_argument('--render', action='store_true', help='Render the environment during training/testing')
args = parser.parse_args()



# User definitions:
# -----------------
train_dqn = args.train
test_dqn = args.test
render = args.render

#! Define env attributes (environment specific)
dim_actions = 4
dim_states = 2


# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 10_000
max_steps = 500

test = 9

# Main:
# -----
if train_dqn:
    env = ContinuousMazeEnv(render_mode="human")


    #! Initialize the Q Net and the Q Target Net
    q_net = Qnet(dim_actions=dim_actions, 
                 dim_states=dim_states)
    q_target = Qnet(dim_actions=dim_actions, 
                    dim_states=dim_states)
    
    q_target.load_state_dict(q_net.state_dict())

    #! Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(),
                           lr=learning_rate)

    rewards = []
    epsilon_decay=[]

    success_counter = 0
    training_complete = False


    for n_epi in range(num_episodes):
        #! Epsilon decay (Please come up with your own logic)
        epsilon = max(0.1, 1.0 - (n_epi*0.4) * 0.001
                      )  # ! Linear annealing from 8% to 1%

        s, _ = env.reset()
        done = False

        #! Define maximum steps per episode, here 10,000
        for _ in range(max_steps):
            #! Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _, _ = env.step(a)
            if render:
                env.render()

            done_mask = 0.0 if done else 1.0

            #! Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0
        epsilon_decay.append(epsilon)

        #! Define a stopping condition for the game:
        if r >= 10.0:  # only goal gives +10.0 at the end
            success_counter += 1
            print(f"[EP {n_epi}] Goal reached! Success count = {success_counter}")
        else:
            success_counter = 0

        if success_counter >= 100:
            print(f"\n Training complete after {n_epi} episodes!")
            print(f"Epsilon decay at the end of the training is : {epsilon}")
            training_complete = True

        if training_complete:
            print("Saving the model and exiting...")
            break

    env.close()

    #! Save the trained Q-net
    torch.save(q_net.state_dict(), f"dqn_{test}.pth")

    rewards_np = np.array(rewards)
    epsilon_decay_np = np.array(epsilon_decay)

    #! Save the rewards and epsilon decay
    np.save(f"rewards{test}.npy", rewards_np)
    np.save(f"epsilon_decay{test}.npy", epsilon_decay_np)

    #! Plot the training curve
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

    #plotting the epsilon decay

    plt.plot(epsilon_decay, label='Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.savefig("epsilon_decay.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    
    env = ContinuousMazeEnv(render_mode="human")

    dqn = Qnet(dim_actions=dim_actions, 
               dim_states=dim_states)
    dqn.load_state_dict(torch.load(f"dqn_{test}.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            #! Completely exploit while testing
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _, _ = env.step(action.argmax().item())
            env.render()
            time.sleep(0.1)  # Slow down rendering for better visualization
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
