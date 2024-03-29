import torch

from .evaluate import get_payoff

class GameEnv:
	def __init__(self, content_model, loader, hist_len = 2):
		self.content_model = content_model
		self.actions = ["WAIT", "BUZZ"]
		self.n_actions = len(self.actions)
		self.user_stats = loader.user_stats
		self.loader = loader
		self.split = -1
		self.hist_len = hist_len
		self.opp_state_dim = 3
		self.inp_state_dim = self.hist_len * self.content_model.n_outputs + 1 + 3 
		self.state_dim =  self.opp_state_dim + self.inp_state_dim
		

	def reset(self):
		self.n_step = 0
		self.tot_buzzes = 0
		self.game_ptr = self.loader.batch_size
		self.buzz_ptr = 0
		
	def new_game(self, split):
		# if split != 0:
		# 	print("New game")
		self.reset()
		self.split = split

		self.load_next_buzz()
		self.buzzed = False	# whether model buzzed
		self.player_buzzed = False	# whether opponent buzzed

		self.get_state = None
		if self.split == 0:
			self.get_state = self.get_state_train
		else:
			self.get_state = self.get_state_test

		return self.get_state()

	def load_next_buzz(self):
		# if self.split != 0:
		# 	print("Load nbuzz")
		if self.buzz_ptr >= self.tot_buzzes:
			self.load_next_question()

		player_buzz = None
		if self.split != 0:
			player_buzz = self.buzzes[self.buzz_ptr]
			self.buzz_ptr = self.buzz_ptr + 1
		else:
			self.buzz_ptr = self.tot_buzzes
			player_buzz = self.buzzes[torch.randint(len(self.buzzes), (1, 1))]

		self.player_id = player_buzz[0]
		self.player_buzz_pos = int(player_buzz[1])
		self.player_correct = (player_buzz[2] == '1')

	def load_next_question(self):
		# print("Load next question", self.game_ptr)
		self.game_ptr = self.game_ptr + 1
		if self.game_ptr >= self.loader.batch_size:
			self.load_next_batch()

		self.buzzes = self.mb_buzzes[self.game_ptr]
		self.buzz_ptr = 0
		self.n_step = 1 # number of word currently running
		self.tot_buzzes = len(self.buzzes)
		self.max_step = self.all_mask[self.game_ptr].sum()

	def load_next_batch(self):
		# print("Load next batch")
		mb_X, mb_y, mb_len, mb_buzzes, all_mask, last_mask = self.loader.load_next_batch(self.split, buzz_info = True)
		self.mb_len = mb_len

		with torch.no_grad():
			self.outputs = self.content_model(mb_X, mb_len) # un_normalised log probs
			self.outputs = self.outputs.view(self.loader.batch_size, self.loader.max_seq_len, -1)
			self.ans_prob = self.outputs.softmax(dim = 2)
			_, self.ans_pred = torch.max(self.outputs, dim = 2)
			self.ans_target = mb_y

		self.game_ptr = 0
		self.mb_buzzes = mb_buzzes
		self.all_mask = all_mask
		self.last_mask = last_mask # might remove


	def get_state_train(self):
		terminal = False
		reward = None
		state = None

		if self.buzzed:
			buzz_pos = self.n_step - 1
			correct = (self.ans_pred[self.game_ptr][buzz_pos - 1] == self.ans_target[self.game_ptr])

			try:
				assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
			except Exception as e:
				print(buzz_pos, self.player_buzz_pos, self.player_correct)
				assert(0)

			# if self.n_step <= self.max_step:
			# 	print("YO, I buzzed not at last")
			terminal = True
			reward = get_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)
		elif self.n_step > self.player_buzz_pos:
			self.player_buzzed = True

			if self.player_correct:
				terminal = True
				reward = -10
			elif self.n_step <= self.max_step:
				terminal = False
				correct = (self.ans_pred[self.game_ptr][self.n_step - 2] == self.ans_target[self.game_ptr])
				if correct:
					reward = -1
				else:
					reward = 1
			else:
				print('Error! Buzz should have happened at last word!!')
				exit(1)
		else:

			terminal = False

			if self.n_step == 1:
				reward = 0
			else:
				correct = (self.ans_pred[self.game_ptr][self.n_step - 2] == self.ans_target[self.game_ptr])
				if correct:
					reward = -1
				else:
					reward = 1


		ts = min(self.n_step, self.max_step)
		state = self.fill_state(ts)

		return state, reward, terminal

	def get_state_test(self):
		terminal = False
		reward = None
		state = None

		if self.buzzed:
			buzz_pos = self.n_step - 1
			correct = (self.ans_pred[self.game_ptr][buzz_pos - 1] == self.ans_target[self.game_ptr])

			try:
				assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
			except Exception as e:
				print(buzz_pos, self.player_buzz_pos, self.player_correct)
				assert(0)

			terminal = True
			reward = get_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)

		elif self.n_step > self.player_buzz_pos:
			self.player_buzzed = True

			if self.player_correct:
				terminal = True
				reward = -10
			elif self.n_step <= self.max_step:
				terminal = False
				reward = 0
			else:
				print('Error! Buzz should have happened at last word!!')
				exit(1)
		else:
			terminal = False
			reward = 0

		ts = min(self.n_step, self.max_step)
		state = self.fill_state(ts)

		# print(ts, self.n_step, self.max_step, self.player_buzz_pos, self.player_correct, reward, terminal)

		return state, reward, terminal


	def fill_state(self, ts):
		state = torch.zeros(self.state_dim)
		i = ts - 1
		j = max(0, ts - self.hist_len)
		ind = 0
		while i >= j:
			state[ind:(ind+self.content_model.n_outputs)] = self.ans_prob[self.game_ptr][i]
			ind += self.content_model.n_outputs

			if i == 0:
				break
			i -= 1

		ind = self.hist_len * self.content_model.n_outputs
		state[ind] = ts - 1

		ind += 1
		if self.player_buzzed:
			if self.player_correct:
				state[ind] = 1
			else:
				state[ind + 1] = 1
		else:
			state[ind + 2] = 1

		ind += 3

		if (self.split != 0 and self.player_id in self.user_stats) or self.split == 0:
			state[ind] = self.user_stats[self.player_id]['overall_acc']
			state[ind + 1] = self.user_stats[self.player_id]['mean_frac']
			state[ind + 2] = self.user_stats[self.player_id]['total_ques']
		# else:
		# 	print("Missing data of ", self.player_id)

		ind += 3
		assert(ind == self.state_dim)

		return state
		

	def step(self, action):
		# 'BUZZ' -> 1 and 'WAIT' -> 0
		if action == 1 or self.n_step == self.max_step:
			self.buzzed = True
		self.n_step = self.n_step + 1

		return self.get_state()
		