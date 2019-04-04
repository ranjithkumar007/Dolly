
class GameEnv:
	def __init__(self, content_model):
		self.content_model = content_model
		self.actions = ["BUZZ", "WAIT"]
		self.user_stats = user_stats
		self.batch_ind = 0
		
	def new_game(self, test):
		self.test = test

		load_next_buzz()
		self.buzzed = False	# whether model buzzed
		self.player_buzzed = False	# whether opponent buzzed
		self.step = 1 # step to be executed next

		return get_my_state()

	def load_next_buzz(self):
		if self.buzz_ptr >= len(self.buzzes):
			load_next_question()

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
		if self.game_ptr >= self.batch_size:
			load_next_batch()

		self.game_ptr = self.game_ptr + 1


	def load_next_batch(self):

	    end_ind = (self.batch_ind+1)*batch_size
        if (self.batch_ind+1)*batch_size > self.train_X.size(0):
            end_ind = train_X.size(0)
        start_ind = end_ind - batch_size

        mb_X = train_X[start_ind:end_ind]
        mb_len = train_seq_len[start_ind:end_ind]

        all_mask, last_mask = sequence_masks(sequence_length=mb_len, max_len=train_X.size(1))
        all_mask = all_mask.flatten().float()
        last_mask = last_mask.flatten().float()

        mb_y = train_y[start_ind:end_ind]
        mb_y = mb_y.view(-1, 1).repeat(1, train_X.size(1)).flatten()






