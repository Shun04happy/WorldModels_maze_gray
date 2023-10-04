#python 03_generate_rnn_data.py

from vae.arch_custommazegrayz32k3conv4 import VAE
import argparse
import config_custom2dmaze
import numpy as np
import os

ROOT_DIR_NAME = "./data/"
ROLLOUT_DIR_NAME = "./data/custom2dmaze_gray/"
SERIES_DIR_NAME = "./data/custom2dmaze_series_grayconv4/"


def get_filelist(N):
    filelist = os.listdir(ROLLOUT_DIR_NAME)
    
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N

def encode_episode(vae, episode):

    obs = episode['obs']#(300, 64, 64, 3)
    action = episode['action']
    reward = episode['reward']
    done = episode['done']

    done = done.astype(int)  
    reward = np.where(reward>0, 1, 0) * np.where(done==1, 1, 0)
    
    action = action[:, np.newaxis]
    mu, log_var, _ = vae.encoder.predict(obs)#(300,32)
    
    initial_mu = mu[0, :]#(1,32)
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)



def main(args):

    N = args.N

    vae = VAE()

    try:
      # vae.set_weights('./vae/weights.h5')
      vae.set_weights('./vae/vae_weights/weights_custommaze_gray5000.10z32k3conv4r100000.ckpt')
    except Exception as e:
      print(e)
      print("./vae/weights_custommaze_roll35000.5z32.ckpt does not exist - ensure you have run 02_train_vae.py first")
      raise


    filelist, N = get_filelist(N)

    file_count = 0

    initial_mus = []
    initial_log_vars = []

    for file in filelist:
      try:
      
        rollout_data = np.load(ROLLOUT_DIR_NAME + file)
        mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_episode(vae, rollout_data)

        np.savez_compressed(SERIES_DIR_NAME + file, mu=mu, log_var=log_var, action = action, reward = reward, done = done)
        initial_mus.append(initial_mu)
        initial_log_vars.append(initial_log_var)

        file_count += 1

        if file_count%50==0:
          print('Encoded {} / {} episodes'.format(file_count, N))

      except Exception as e:
        print(e)
        print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_mus = np.array(initial_mus)
    initial_log_vars = np.array(initial_log_vars)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z_custommaze_gray5000.10z32k3conv4r100000.ckpt.npz', initial_mu=initial_mus, initial_log_var=initial_log_vars)#(10000, 32)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  args = parser.parse_args()

  main(args)
