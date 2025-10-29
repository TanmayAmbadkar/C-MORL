from a2c_ppo_acktr.model import Policy
import pickle
import torch
import mo_gymnasium
import imageio
import numpy as np
import matplotlib.pyplot as plt
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility
import pandas as pd
import os

# Pareto front calculation (robust and correct for maximization)
def pareto_front(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if all(points[j] >= points[i]) and any(points[j] > points[i]):
                is_efficient[i] = False
                break
    return is_efficient

env = mo_gymnasium.make("minecart-v0")
all_rewards = []
hypervolumes = []
sparsities = []
expected_utilities = []

seeds = os.listdir("results/minecart/cmorl-ipo/")

for seed in seeds:
    policies = os.listdir(f"results/minecart/cmorl-ipo/{seed}/final/")
    policies = sorted([p for p in policies if p.startswith("EP_policy_")])
    print(f"Processing seed {seed} with {len(policies)} policies")
    current_rewards = []
    for i in range(len(policies)):
        policy_file = torch.load(open(f"results/minecart/cmorl-ipo/{seed}/final/EP_policy_{i}.pt", "rb"), weights_only=True)
        env_params = pickle.load(open(f"results/minecart/cmorl-ipo/{seed}/final/EP_env_params_{i}.pkl", "rb"))
        # print("Environment parameters:", vars(env_params['ob_rms']))
        ob_rms = env_params['ob_rms']
        policy = Policy(
            obs_shape=env.observation_space.shape,
            action_space=env.action_space,
            # base_kwargs=dict(recurrent=False, use_critic=True, use_gae=True, use_proper_time_limits=True),
            obj_num = 3
        )
        policy.load_state_dict(policy_file)
        rendered_frames = []
        obs, info = env.reset(seed=42)
        done = False
        rewards = np.zeros(3)
        gamma = 1
        while not done:
            obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
            _, action, _, _ = policy.act(torch.Tensor(obs.reshape(1, -1)), None, None, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0].detach().numpy())
            rewards += gamma * reward
            done = terminated or truncated
            # rendered_frames.append(env.render())
            gamma *= 1.0

        current_rewards.append(rewards)
        print(f"Episode {i} rewards: {rewards}")

    all_rewards += current_rewards
    
    current_rewards = np.array(current_rewards)
    mask = pareto_front(current_rewards)
    front = current_rewards[mask]
    dominated = current_rewards[~mask]
    
    sample_weights = torch.distributions.dirichlet.Dirichlet(torch.ones(3)).sample((1000,)).numpy()
    hypervolumes.append(hypervolume(ref_point = np.array([-1, -1, -200]), points=front))
    sparsities.append(sparsity(front))
    expected_utilities.append(expected_utility(front, sample_weights))

    
    



all_rewards = np.array(all_rewards)
mask = pareto_front(all_rewards)
front = all_rewards[mask]
dominated = all_rewards[~mask]

moppo_files = pickle.load(open("eval_results_minecart-v0.pkl", "rb"))
front_moppo = np.array(moppo_files['pareto_front'])

plt.scatter(front[:, 0], front[:, 1], color='blue', label='C-MORL', s = 100)
plt.scatter(front_moppo[:, 0], front_moppo[:, 1], color='red', label='MOPPO', s = 100)


# plt.scatter(dominated[:, 0], dominated[:, 1], color='red', label='Dominated Points')

# for file in ["ant2d_CAPQL.csv", "ant2d_GPI-LS.csv", "ant2d_PG-MORL.csv"]:
#     data = pd.read_csv(file)
#     front_file = pareto_front(data.values)
#     front_file = data.values[front_file]
#     print(front_file)
#     plt.scatter(front_file[:, 0], front_file[:, 1], label=file[6:-4], s=100)
plt.legend()
plt.xlabel("X Velocity")
plt.ylabel("Y Velocity")
plt.title("Pareto Front for Minecart")
# plt.grid()
plt.savefig("minecart_rewards.png")


# sample_weights = torch.distributions.dirichlet.Dirichlet(torch.ones(2)).sample((1000,)).numpy()
# print("Hypervolume", hypervolume(ref_point = np.array([-100, -100]), points=front))
# print("Sparsity", sparsity(front))
# print("Expected Utility", expected_utility(front, sample_weights))

print("Hypervolumes:", np.mean(hypervolumes), np.std(hypervolumes))
print("Sparsities:", np.mean(sparsities), np.std(sparsities))
print("Expected Utilities:", np.mean(expected_utilities), np.std(expected_utilities))