#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from rnn.arch_custommazegrayz32 import RNN
import argparse
import numpy as np
import os

ROOT_DIR_NAME = './data/'
SERIES_DIR_NAME = './data/custom3dmaze_series_gray2/'


def get_filelist(N):
    filelist = os.listdir(SERIES_DIR_NAME)
    filelist = [x for x in filelist if (x != '.DS_Store' and x!='.gitignore')]
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N


def random_batch(filelist, batch_size):
	N_data = len(filelist)
	indices = np.random.permutation(N_data)[0:batch_size]

	z_list = []
	action_list = []
	rew_list = []
	done_list = []

	for i in indices:
		try:
			new_data = np.load(SERIES_DIR_NAME + filelist[i], allow_pickle=True)

			

			mu = new_data['mu']
			log_var = new_data['log_var']
			action = new_data['action']
			reward = new_data['reward']
			done = new_data['done']

			reward = np.expand_dims(reward, axis=1)
			done = np.expand_dims(done, axis=1)

			s = log_var.shape

			z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

			z_list.append(z)
			action_list.append(action)
			rew_list.append(reward)
			done_list.append(done)

		except Exception as e:
			print(e)

			

	z_list = np.array(z_list)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	done_list = np.array(done_list)

	return z_list, action_list, rew_list, done_list


def batch(filelist, batch_size, step):
	sorted_indices = np.argsort(filelist)  
	start_idx = step * batch_size
	end_idx = (step + 1) * batch_size
    
    
	if end_idx > len(sorted_indices):
		end_idx = len(sorted_indices)
    
	indices = sorted_indices[start_idx:end_idx]
	
	z_list = []
	action_list = []
	rew_list = []
	done_list = []

	for i in indices:
		try:
			new_data = np.load(SERIES_DIR_NAME + filelist[i], allow_pickle=True)

			

			mu = new_data['mu']
			log_var = new_data['log_var']
			action = new_data['action']
			reward = new_data['reward']
			done = new_data['done']

			reward = np.expand_dims(reward, axis=1)
			done = np.expand_dims(done, axis=1)

			s = log_var.shape

			z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

			z_list.append(z)
			action_list.append(action)
			rew_list.append(reward)
			done_list.append(done)

		except Exception as e:
			print(e)

			

	z_list = np.array(z_list)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	done_list = np.array(done_list)

	return z_list, action_list, rew_list, done_list


def main(args):
	
	new_model = args.new_model
	N = int(args.N)
	steps = int(args.steps)
	batch_size = int(args.batch_size)
	epochs=int(args.epochs)

	rnn = RNN() #learning_rate = LEARNING_RATE

	if not new_model:
		try:
			rnn.set_weights(f'./rnn/weights_custom3dmaze_gray2z32k3conv1g5h256.{steps}.{batch_size}.{epochs}.ckpt')
		except:
			print(f"Either set --new_model or ensure ./rnn/weights_custom3dmaze_gray2_randz32k3conv1g5h256.{steps}.{batch_size}.{epochs}.ckpt exists")
			raise


	filelist, N = get_filelist(N)

	for epoch in range(epochs):

		for step in range(steps):
			print('STEP ' + str(step),'epoch'+str(epoch))

			z, action, rew ,done = batch(filelist, batch_size, step)
			# z, action, rew ,done = random_batch(filelist, batch_size)
			
			# print("z shape:", z.shape)
			# print("action shape:", action.shape)
			
			# print(rew.shape)
			
			rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1,:], rew[:, :-1, :]], axis = 2)
			
			rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :]], axis = 2) #, done[:, 1:, :]
			
			if step == 0:
				np.savez_compressed(ROOT_DIR_NAME + f'rnn_files/rnn_files_custom3dmaze_gray2z32k3conv1g5h256.{steps}.{batch_size}.{epochs}.ckpt.npz', rnn_input = rnn_input, rnn_output = rnn_output)

			rnn.train(rnn_input, rnn_output)

			if step % 10 == 0:

				# rnn.model.save_weights(f'./rnn/weights{steps}.{batch_size}.h5')
				rnn.model.save_weights(f'./rnn/rnn_weights/weights_custom3dmaze_gray2z32k3conv1g5h256.{steps}.{batch_size}.{epochs}.ckpt')
		# rnn.model.save_weights(f'./rnn/weights{steps}.{batch_size}.h5')

	rnn.model.save_weights(f'./rnn/rnn_weights/weights_custom3dmaze_gray2z32k3conv1g5h256.{steps}.{batch_size}.{epochs}.ckpt')




if __name__ == "__main__":
		parser = argparse.ArgumentParser(description=('Train RNN'))
		parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
		parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
		parser.add_argument('--steps', default = 4000, help='how many rnn batches to train over')
		parser.add_argument('--batch_size', default = 100, help='how many episodes in a batch?')
		parser.add_argument('--epochs', default = 1)
		args = parser.parse_args()

		main(args)
