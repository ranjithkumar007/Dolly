from game_env import GameEnv
from transitions import ReplayMemory, ParallelTable
import torch
import torch.nn as nn

class DQNQuizBowlPlayer(nn.Module):
	def __init__(self, inp_state_dim, opp_state_dim, n_actions):
		super(DQNQuizBowlPlayer, self).__init__()

		self.hidden_opp = 10
		self.hidden_inp = 128
		self.hidden_common = 128
		self.n_actions = n_actions
		self.inp_state_dim = inp_state_dim
		self.opp_state_dim = opp_state_dim


		__build_model()

	def __build_model(self):
		model1 = torch.nn.Sequential(
		          torch.nn.Linear(self.inp_state_dim, self.hidden_inp),
		          torch.nn.ReLU(),
	        	)

		model2 = torch.nn.Sequential(
		          torch.nn.Linear(self.opp_state_dim, self.hidden_opp),
		          torch.nn.ReLU(),
	        	)

		self.common_net = torch.nn.Sequential(
		          torch.cat(ParallelTable(model1, model2), dim = 1),
		          torch.nn.Linear(self.hidden_opp + self.hidden_inp, self.hidden_common),
		          torch.nn.ReLU(),
		          torch.nn.Linear(self.hidden_common, self.n_actions)
	        	)

	def forward(self, x):
		x1 = x[:inp_state_dim]
		x2 = x[inp_state_dim:]
		return self.common_net([x1, x2])


def select_action(state):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def QLearn():
	

def run(hyperparams, content_model, loader):
	gamma = hyperparams['gamma']
	eps_start = hyperparams['eps_start']
	eps_end = hyperparams['eps_end']
	eps_decay = hyperparams['eps_decay']
	target_update = hyperparams['target_update']

	env = GameEnv(content_model, loader)
	n_actions = env.n_actions
	inp_state_dim = env.inp_state_dim
	opp_state_dim = env.opp_state_dim

	policy_net = DQNQuizBowlPlayer(inp_state_dim, opp_state_dim, n_actions)
	target_net = DQNQuizBowlPlayer(inp_state_dim, opp_state_dim, n_actions)

	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(1000)

	steps_done = 0




	
