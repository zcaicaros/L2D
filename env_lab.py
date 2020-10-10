import numpy as np
from JSSP_Env import SJSSP
from uniform_instance_gen import uni_instance_gen
from Params import configs
import time

n_j = 200
n_m = 50
low = 1
high = 99
SEED = 11
np.random.seed(SEED)
env = SJSSP(n_j=n_j, n_m=n_m)


# rollout env random action
t1 = time.time()
data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
dur = np.array([[83, 65,  3],
               [69, 42, 64],
               [27, 27, 18]])
mch = np.array([[3, 2, 1],
                [1, 2, 3],
                [2, 1, 3]])
# data = (dur, mch)
print('Dur')
print(data[0])
print('Mach')
print(data[-1])
print()
_, _, omega, mask = env.reset(data)
# print('Init end time')
# print(env.LBs)
# print()
rewards = [- env.initQuality]
while True:
    action = np.random.choice(omega[~mask])
    # print('action:', action)
    adj, _, reward, done, omega, mask = env.step(action)
    rewards.append(reward)
    # print('ET after action:\n', env.LBs)
    # print(fea)
    # print()
    if env.done():
        break
t2 = time.time()
makespan = sum(rewards) - env.posRewards
# print(makespan)
# print(env.LBs)
print(t2 - t1)
# np.save('sol', env.opIDsOnMchs // n_m)
# np.save('jobSequence', env.opIDsOnMchs)
# np.save('testData', data)
# print(env.opIDsOnMchs // n_m + 1)
# print(env.step_count)
# print(t)
# print(np.concatenate((fea, data[1].reshape(-1, 1)), axis=1))
# print()
# print(env.adj)


'''# rtools solution
from ortools_baseline import MinimalJobshopSat
data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
# print(data)
times_rearrange = np.expand_dims(data[0], axis=-1)
machines_rearrange = np.expand_dims(data[1], axis=-1)
data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
result = MinimalJobshopSat(data.tolist())
print(result)'''

'''# run solution to test env
from ortools_baseline import MinimalJobshopSat
np.random.seed(SEED)
data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
times_rearrange = np.expand_dims(data[0], axis=-1)
machines_rearrange = np.expand_dims(data[1], axis=-1)
data2ortools = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
opt_val, sol = MinimalJobshopSat(data2ortools.tolist())
steps_basedon_sol = []
for i in range(n_m):
    get_col_position_unsorted = np.argwhere(data[-1] == (i+1))
    get_col_position_sorted = get_col_position_unsorted[sol[i]]
    sol_i = sol[i] * n_m + get_col_position_sorted[:, 1]
    steps_basedon_sol.append(sol_i)

steps_basedon_sol = np.asarray(steps_basedon_sol).tolist()
for m in range(n_m):
    steps_basedon_sol[m].append(0)

c = 0
adj, fea, omega, mask = env.reset(data)
rewards = [- env.initQuality]
while not env.done():
    for m in range(n_m):
        for t in range(steps_basedon_sol[m][-1], n_j):
            if steps_basedon_sol[m][t] in env.omega:
                adj, fea, reward, done, omega, mask = env.step(steps_basedon_sol[m][t])
                rewards.append(reward)
                steps_basedon_sol[m][-1] += 1
                c += 1
            else:
                break
print(rewards)
makespan = sum(rewards) - env.posRewards
print(makespan)
print(opt_val)'''

'''# Test network
from mb_agg import *
import torch
from agent_utils import select_action
from agent_utils import greedy_select_action
from models.actor_critic import ActorCritic

torch.manual_seed(configs.torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(configs.torch_seed)
device = torch.device(configs.device)

# define network
actor_critic = ActorCritic(n_j=n_j,
                           n_m=n_m,
                           num_layers=configs.num_layers,
                           learn_eps=False,
                           neighbor_pooling_type=configs.neighbor_pooling_type,
                           input_dim=configs.input_dim,
                           hidden_dim=configs.hidden_dim,
                           num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                           num_mlp_layers_actor=configs.num_mlp_layers_actor,
                           hidden_dim_actor=configs.hidden_dim_actor,
                           num_mlp_layers_critic=configs.num_mlp_layers_critic,
                           hidden_dim_critic=configs.hidden_dim_critic,
                           device=device)

# calculate g_pool for each step
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
                         n_nodes=n_j * n_m,
                         device=device)

data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
adj, fea, omega, mask = env.reset(data)
rewards = [- env.initQuality]
while True:
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    adj_tensor = torch.from_numpy(np.copy(adj)).to(device)
    candidate_tensor = torch.from_numpy(np.copy(omega)).to(device)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
    with torch.no_grad():
        pi, _ = actor_critic(x=fea_tensor,
                             graph_pool=g_pool_step,
                             padded_nei=None,
                             adj=adj_tensor,
                             candidate=candidate_tensor.unsqueeze(0),
                             mask=mask_tensor.unsqueeze(0))
        # action, _ = select_action(pi, omega, None)
        _, indices = pi.squeeze().cpu().max(0)
        action = omega[indices.numpy().item()]
        adj, fea, reward, done, omega, mask = env.step(action.item())
        rewards.append(reward)
        if env.done():
            break
makespan = sum(rewards) - env.posRewards
print(makespan)
print(env.posRewards)
print(env.opIDsOnMchs)'''

'''# Test random instances
for _ in range(3):
    times, machines = uni_instance_gen(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
    print(times)'''