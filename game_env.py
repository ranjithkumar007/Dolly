class GameEnv:
	def __init__(self, content_model, loader, hist_len = 2):
		self.content_model = content_model
		self.actions = ["BUZZ", "WAIT"]
		self.batch_ind = 0
		self.user_stats = loader.user_stats
		self.step = 0
		self.loader = loader
		self.tot_buzzes = 0
		self.game_ptr = self.loader.batch_size
		self.buzz_ptr = 0
		self.hist_len = hist_len
		self.opp_state_dim = 3
		self.inp_state_dim = self.hist_len * self.content_model.n_outputs + 1 + 3 
		self.state_dim =  self.opp_state_dim + self.inp_state_dim
		
	def new_game(self, split):
		self.split = split

		load_next_buzz()
		self.buzzed = False	# whether model buzzed
		self.player_buzzed = False	# whether opponent buzzed
		self.buzz_ptr = 0 # step to be executed next

		return get_my_state()

	def load_next_buzz(self):
		if self.buzz_ptr >= self.tot_buzzes:
			load_next_question()

		player_buzz = None
		if self.test:
			self.buzz_ptr = self.buzz_ptr + 1
			player_buzz = self.buzzes[self.buzz_ptr]
		else:
			self.buzz_ptr = len(self.buzzes)
			player_buzz = self.buzzes[torch.randint(0, len(self.buzzes))]

		self.player_id = player_buzz[0]
		self.player_buzz_pos = player_buzz[1]
		self.player_correct = (player_buzz[2] == 1)

	def load_next_question(self):
		if self.game_ptr >= self.loader.batch_size:
			load_next_batch()

		self.buzzes = self.mb_buzzes[self.game_ptr]
		self.buzz_ptr = 0
		self.step = 1
		self.tot_buzzes = len(self.buzzes)
		self.max_step = self.all_mask[self.game_ptr].sum()
		self.game_ptr = self.game_ptr + 1

	def load_next_batch(self):
		mb_X, mb_y, mb_len, mb_buzzes, all_mask, last_mask = self.loader.load_next_batch(self.split)
		self.outputs = self.content_model(mb_X, mb_len) # un_normalised log probs
		self.outputs = self.outputs.view(self.batch_size, self.loader.max_seq_len, -1)
		self.game_ptr = 0
		self.mb_buzzes = mb_buzzes
		self.all_mask = all_mask
		self.last_mask = last_mask # might remove
		self.ans_prob = self.outputs.softmax(dim = 2)


	def get_state_train(self):
		terminal = False
		reward = None
		state = None

		if self.buzzed:
			buzz_pos = self.step - 1
			correct = (self.ans_pred[buzz_pos] == self.ans_target)

			assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
			terminal = True
			reward = get_one_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)

		elif self.step > self.player_buzz_pos:
			self.player_buzzed = True

			if self.player_correct:
				terminal = True
				reward = -10
			elif self.step <= self.max_step:
				terminal = False
				correct = (self.ans_pred[self.step - 1] == self.ans_target)

				if correct:
					reward = -1
				else:
					reward = 1
			else:
				print('Error! Buzz should have happened at last word!!')
				exit(1)
		else:
			terminal = False

			if self.step == 1:
				reward = 0
			else:
				correct = (self.ans_pred[self.step - 1] == self.ans_target)
				if correct:
					reward = -1
				else:
					reward = 1


		ts = min(self.step, self.max_step)
		state = fill_state(ts)

		return state, reward, terminal

	def get_state_test(self):
		terminal = False
		reward = None
		state = None

		if self.buzzed:
			buzz_pos = self.step - 1
			correct = (self.ans_pred[buzz_pos] == self.ans_target)

			assert(buzz_pos <= self.player_buzz_pos or not self.player_correct)
			terminal = True
			reward = get_one_payoff(buzz_pos, correct, self.player_buzz_pos, self.player_correct)

		elif self.step > self.player_buzz_pos:
			self.player_buzzed = True

			if self.player_correct:
				terminal = True
				reward = -10
			elif self.step <= self.max_step:
				terminal = False
				reward = 0
			else:
				print('Error! Buzz should have happened at last word!!')
				exit(1)
		else:
			terminal = False
			reward = 0

		ts = min(self.step, self.max_step)
		state = fill_state(ts)

		return state, reward, terminal


	def fill_state(self, ts):
		state = torch.zeros(self.state_dim)
		i = ts - 1
		j = max(0, ts - self.hist_len)
		ind = 0
		while i >= j:
			state[ind] = self.ans_prob[self.game_ptr][i]
			ind += self.content_model.n_outputs

		ind = self.hist_len * self.content_model.n_outputs
		state[ind] = ts - 1

		ind += 1
		if player_buzzed:
			if player_correct:
				state[ind] = 1
			else:
				state[ind + 1] = 1
		else:
			state[ind + 2] = 1

		ind += 3

		state[ind] = self.user_stats[self.player_id]['overall_acc']
		state[ind + 1] = self.user_stats[self.player_id]['mean_frac']
		state[ind + 2] = self.user_stats[self.player_id]['total_ques']

		ind += 3
		assert(ind == self.state_dim)

		return state
		

	def step(action):
		if action == 'BUZZ' or self.step == self.max_step:
			self.buzzed = True
		self.step = self.step + 1

		if self.split != 0: # not train
			return get_state_test()
		else
			return get_state_train()
