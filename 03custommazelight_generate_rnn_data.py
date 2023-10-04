#python 03_generate_rnn_data.py

from vae.arch_custommazelightz16k3conv1 import VAE
import argparse
import config_custom2dmaze
import numpy as np
import os

ROOT_DIR_NAME = "./data/"
ROLLOUT_DIR_NAME = "./data/custom2dmaze_light1/"
SERIES_DIR_NAME = "./data/custom2dmaze_series_light1/"


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
    z = vae.encoder.predict(obs)#(300,32)
    
    initial_z = z[0, :]#(1,32)
    

    return (z, action, reward, done, initial_z)



def main(args):

    N = args.N

    vae = VAE()

    try:
      # vae.set_weights('./vae/weights.h5')
      vae.set_weights('./vae/vae_weights/weights_custommaze_light5000.10z16k3conv1r100000.ckpt')
    except Exception as e:
      print(e)
      print("./vae/weights_custommaze_roll35000.5z32.ckpt does not exist - ensure you have run 02_train_vae.py first")
      raise


    filelist, N = get_filelist(N)

    file_count = 0

    initial_zs = []
    

    for file in filelist:
      try:
      
        rollout_data = np.load(ROLLOUT_DIR_NAME + file)
        z, action, reward, done, initial_z,  = encode_episode(vae, rollout_data)

        np.savez_compressed(SERIES_DIR_NAME + file, z=z, action = action, reward = reward, done = done)
        initial_zs.append(initial_z)
        

        file_count += 1

        if file_count%50==0:
          print('Encoded {} / {} episodes'.format(file_count, N))

      except Exception as e:
        print(e)
        print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_zs = np.array(initial_zs)
    

    print('ONE Z SHAPE = {}'.format(z.shape))
    print('INITIAL Z SHAPE = {}'.format(initial_zs.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z_custom2dmaze_light5000.10z16k3conv1r100000.ckpt.npz', initial_z=initial_zs)#(10000, 32)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  args = parser.parse_args()

  main(args)
