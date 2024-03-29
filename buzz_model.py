import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import count
import math
import click
import matplotlib.pyplot as plt

from util.game_env import GameEnv
from util.helper_classes import ReplayMemory, Transition
from util.helper_functions import load_checkpoint_buzz, load_best_model, save_checkpoint , load_checkpoint_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Double DQN
class DDQNQuizBowlPlayer(nn.Module):
    def __init__(self, inp_state_dim, opp_state_dim, n_actions):
        super(DDQNQuizBowlPlayer, self).__init__()

        self.hidden_opp = 10
        self.hidden_inp = 128
        self.hidden_common = 128
        self.n_actions = n_actions
        self.inp_state_dim = inp_state_dim
        self.opp_state_dim = opp_state_dim

        self.__build_model()

    def __build_model(self):
        self.model1 = torch.nn.Sequential(
                  torch.nn.Linear(self.inp_state_dim, self.hidden_inp),
                  torch.nn.ReLU(),
                )

        self.model2 = torch.nn.Sequential(
                  torch.nn.Linear(self.opp_state_dim, self.hidden_opp),
                  torch.nn.ReLU(),
                )

        self.common_net = torch.nn.Sequential(
                  torch.nn.Linear(self.hidden_opp + self.hidden_inp, self.hidden_common),
                  torch.nn.ReLU(),
                  torch.nn.Linear(self.hidden_common, self.n_actions)
                )


    def forward(self, x):
        x1 = x[..., :self.inp_state_dim]
        x2 = x[..., self.inp_state_dim:]

        x1 = self.model1(x1)
        x2 = self.model2(x2)

        x = torch.cat((x1, x2), dim = -1)
        x = self.common_net(x)

        return x

def select_action_test(state, policy_net):
    with torch.no_grad():
        return policy_net(state).max(0)[1]

def select_action(state, policy_net, learn_start, steps_done, eps_start, eps_end, eps_decay, n_actions):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * (max(0, steps_done - learn_start)) / eps_decay)
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(0)[1]#.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def QLearn(memory, optimizer, policy_net, target_net, batch_size, gamma):
    if len(memory) < batch_size:
        return
        
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]).cuda()
    
    # print(type(batch.action), len(batch.action), len(batch.action[0]), type(batch.action[0]))
    state_batch = torch.stack(batch.state).cuda()
    # print(state_batch.type(), state_batch.size())
    action_batch = torch.tensor(batch.action).view(-1, 1).cuda()    
    reward_batch = torch.stack(batch.reward).cuda()

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    
    best_actions_policy_net = policy_net(non_final_next_states).max(1)[1].view(-1, 1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions_policy_net).squeeze().detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig("figures/episode_durations_" + str(len(durations_t)) + ".png")

def validate(policy_net, env, split = 1):
    
    with torch.no_grad():
        epoch_size = env.loader.inputs[split][0].size(0)
        epoch_eps_len = 0
        num_buzzes = 0
        epoch_reward = 0
        epoch_buzz_pos = 0
        
        with click.progressbar(range(epoch_size)) as game_inds:
            for i_game in game_inds:
                state, reward, terminal = env.new_game(split)
                state = state.cuda()

                for t in count():
                    # Select and perform an action
                    action = select_action_test(state, policy_net)
                    
                    next_state, reward, terminal = env.step(action.item())

                    epoch_reward += reward
                    next_state = next_state.cuda()
                    reward = torch.tensor([reward], dtype = torch.float, device=device)

                    state = next_state

                    if terminal:
                        epoch_eps_len += t + 1
                        if action.item() == 1:
                            num_buzzes += 1
                            epoch_buzz_pos += t + 1
                        break


        xx = 0
        if num_buzzes:
            xx = epoch_buzz_pos / num_buzzes

        return epoch_reward/epoch_size, xx, epoch_eps_len / epoch_size, num_buzzes/epoch_size


def run(hyperparams, content_model, loader, restore, checkpoint_file = None, only_validate = True):
    env = GameEnv(content_model, loader)

    n_actions = env.n_actions
    inp_state_dim = env.inp_state_dim
    opp_state_dim = env.opp_state_dim

    logger = [{'avg_reward' : [], 'avg_episode_length' : [], 'frac_buzzes':[], 'avg_buzz_pos' : []} for i in range(3)]
    steps_done = 0
    min_val_reward = 99999999999
    start_game_ind = 0

    policy_net = DDQNQuizBowlPlayer(inp_state_dim, opp_state_dim, n_actions)
    target_net = DDQNQuizBowlPlayer(inp_state_dim, opp_state_dim, n_actions)
    
    if only_validate:
        policy_net = load_checkpoint_policy(policy_net, "checkpoints/buzz/best_model.pth")
        policy_net.cuda()
        print(validate(policy_net, env, split = 1))
        return

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    replay_memory_size = hyperparams['replay_memory_size']
    memory = ReplayMemory(replay_memory_size)


    if restore and checkpoint_file:
        # remove env
        hyperparams, policy_net, optimizer, memory, logger, \
            start_game_ind, steps_done, min_val_reward = load_checkpoint_buzz(hyperparams, policy_net, \
                                                                            optimizer, memory, logger, \
                                                                            checkpoint_file)

    gamma = hyperparams['gamma']
    eps_start = hyperparams['eps_start']
    eps_end = hyperparams['eps_end']
    eps_decay = hyperparams['eps_decay']
    target_update = hyperparams['target_update']
    replay_memory_size = hyperparams['replay_memory_size']
    num_episodes = hyperparams['num_episodes']
    learn_start = hyperparams['learn_start']
    update_freq = hyperparams['update_freq']

    batch_size = loader.batch_size

    policy_net.cuda()
    target_net.cuda()

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # episode_durations = []
    epoch_size = loader.inputs[0][0].size(0)
    start_epoch = int(start_game_ind / epoch_size)

    num_games = num_episodes * epoch_size
    epoch_reward = 0
    epoch_buzz_pos = 0
    num_buzzes = 0
    epoch_eps_len = 0

    print(num_games)
    
    with click.progressbar(range(start_game_ind, num_games)) as game_inds:
        for i_game in game_inds:
            # Initialize the environment and state
            state, reward, terminal = env.new_game(0) # split = 0 for train
            state = state.cuda()
            

            for t in count():
                # Select and perform an action
                action = select_action(state, policy_net,  learn_start, steps_done, eps_start, eps_end, eps_decay, n_actions)
                # print(action)
                # print(action.type())
                steps_done += 1

                next_state, reward, terminal = env.step(action.item())
                epoch_reward += reward
                next_state = next_state.cuda()
                reward = torch.tensor([reward], dtype = torch.float, device=device)


                if terminal:
                    next_state = None
                    # memory.push(state.cuda(), action.cuda(), None, reward.cuda())
                # else:
                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if steps_done > learn_start and steps_done % update_freq:
                # Perform one step of the optimization (on the target network)
                    QLearn(memory, optimizer, policy_net, target_net, batch_size, gamma)

                if terminal:
                    # episode_durations.append(t + 1)
                    # plot_durations(episode_durations)
                    
                    if action.item() == 1:
                        num_buzzes += 1
                        epoch_buzz_pos += t + 1
                    epoch_eps_len += t + 1
                    break

            # Update the target network, copying all weights and biases in DQN
            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            
            if (i_game + 1) % epoch_size == 0:
                epoch = int(i_game / epoch_size)
                print('\nOn training set : Epoch:  %d | avg_reward: %.4f | avg_buzz_pos : %.2f | avg_episode_length : %.2f | frac_buzzes : %.2f'
                      %(epoch, epoch_reward/epoch_size, epoch_buzz_pos/num_buzzes, epoch_eps_len / epoch_size, num_buzzes/epoch_size)) 

                logger[0]['avg_reward'].append(epoch_reward / epoch_size)
                logger[0]['avg_buzz_pos'].append(epoch_buzz_pos / num_buzzes)
                logger[0]['avg_episode_length'].append(epoch_eps_len / epoch_size)
                logger[0]['frac_buzzes'].append(num_buzzes / epoch_size)

                val_reward, val_buzz_pos, val_eps_len, val_frac_buzzes = validate(policy_net, env, split = 1)
                
                is_best = False
                if val_reward < min_val_reward : 
                    min_val_reward = val_reward
                    print("Best Model Found")
                    is_best = True

                print('\nOn Validation set : Epoch:  %d | avg_reward: %.4f | avg_buzz_pos : %.2f | avg_episode_length : %.2f | frac_buzzes : %.2f'
                    %(epoch, val_reward, val_buzz_pos, val_eps_len, val_frac_buzzes)) 

                logger[1]['avg_reward'].append(val_reward)
                logger[1]['avg_buzz_pos'].append(val_buzz_pos)
                logger[1]['avg_episode_length'].append(val_eps_len)
                logger[1]['frac_buzzes'].append(val_frac_buzzes)

                save_checkpoint({'hyperparams': hyperparams,
                    'state_dict': policy_net.state_dict(),
                    'memory' : memory,
                    'start_game_ind' : (i_game + 1),
                    'steps_done' : steps_done,
                    'learn_start' : learn_start,
                    'logger': logger,
                    'min_val_reward' : min_val_reward,
                    'optimizer' : optimizer.state_dict()}, False, checkpoint_file)

                save_checkpoint({'state_dict' : policy_net.state_dict()}, is_best, \
                                        "checkpoints/buzz/policy.pth")

                epoch_reward = 0
                epoch_buzz_pos = 0
                epoch_eps_len = 0
                num_buzzes = 0

    
    best_policy_net = load_best_model(policy_net, filename = 'checkpoints/buzz/best_model.pth')
    test_reward, test_buzz_pos, test_eps_len, test_frac_buzzes = validate(best_policy_net, env, split = 2)
    
    print('\nOn Test set(Best from validation set) avg_reward: %.4f | avg_buzz_pos : %.2f | avg_episode_length : %.2f | frac_buzzes : %.2f'
        %(test_reward, test_buzz_pos, test_eps_len, test_frac_buzzes)) 

    logger[2]['avg_reward'].append(test_reward)
    logger[2]['avg_buzz_pos'].append(test_buzz_pos)
    logger[2]['avg_episode_length'].append(test_eps_len)
    logger[2]['frac_buzzes'].append(test_frac_buzzes)

    return logger
