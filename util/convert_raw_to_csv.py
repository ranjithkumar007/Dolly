import pandas as pd
import os
import numpy as np

np.random.seed(5)

def convert2csv(data_dir, file_name, buzz):
	file_name = os.path.join(data_dir, file_name)
	data_train = []
	data_cv = []
	data_test = []
	buzz_d=[]

	with open(file_name, 'r') as f:
		lines = f.read().splitlines() 
		for line in lines:
			slots = line.split('|||')
			temp = None

			if not buzz:
				assert(len(slots) == 5)
				qid, cat, ans, split, qtext = slots

				temp = (qtext.strip(), ans.strip(), cat.strip())
			else:
				assert(len(slots) == 6)
				qid, cat, ans, split, qtext, buzzes = slots
				qlen = len(qtext.split(' '))

				buzzes_split = buzzes.split('|')
				pos_b=[]
				neg_b=[]
				for b in buzzes_split:
					uid, pos, cor = b.split('-')
					if cor == '0':
						neg_b.append(int(pos))
					else:
						assert(cor=='1')
						pos_b.append(int(pos))
					buzz_d.append((qid, uid, int(pos), int(cor), cat, float(pos)/float(qlen)))
				temp = (qtext.strip(), ans.strip(), cat.strip(), buzzes)

			split = split.strip()
			if split == "train":
				data_train.append(temp)
			elif split == "dev":
				data_cv.append(temp)
			else:
				assert(split == "test")
				data_test.append(temp)

	def write_to_csv(d, split):
		df = pd.DataFrame(d)
		df['split']=split
		print(split, df.shape)
		df.to_csv(os.path.join(data_dir, str(split) + '.csv'), header = False, index = False)

	data_train = np.array(data_train)
	data_cv = np.array(data_cv)
	data_test = np.array(data_test)
	buzz_df = np.array(buzz_d)

	write_to_csv(data_train, "train")
	write_to_csv(data_cv, "val")
	write_to_csv(data_test, "test")
	buzzes_df = pd.DataFrame(buzz_df, columns = ["que_id", "user", "ans_position", "correct", "category", "frac_ans_position"])
	buzzes_df['split']=split
	print(split, buzzes_df.shape)
	buzzes_df.to_csv(os.path.join(data_dir, "user_stats_on_"+str(split) + '.csv'), header = True)

