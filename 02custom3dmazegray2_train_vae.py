#python 02_train_vae.py --new_model

from vae.arch_custom3dmazegray2z32conv1 import VAE
import argparse
import numpy as np
import config
import os
import datetime
import gc
from vae.arch import K
import time
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# DIR_NAME = './data/rollout/'
DIR_NAME = './data/custom3dmaze_gray2/'

SCREEN_SIZE_X = 80
SCREEN_SIZE_Y = 240
GLAY_SCALE = 1

# SCREEN_SIZE_X = 64
# SCREEN_SIZE_Y = 64


def import_data(load, M,i):
  filelist = os.listdir(DIR_NAME)
  filelist = [x for x in filelist if x != '.DS_Store']
  filelist.sort()
  length_filelist = len(filelist)


  if length_filelist > load:
    filelist = filelist[i*load:(i+1)*load]
    # filelist = filelist[:N]
    # filelist = filelist[N:2*N]
    # filelist = filelist[2*N:3*N]
    # filelist = filelist[3*N:4*N]
    # filelist = filelist[4*N:5*N]
    # filelist = filelist[5*N:6*N]
    # filelist = filelist[6*N:7*N]
    # filelist = filelist[7*N:8*N]
    # filelist = filelist[8*N:9*N]
    # filelist = filelist[9*N:10*N]
    # filelist = filelist[10*N:11*N]
    # filelist = filelist[11*N:12*N]
    # filelist = filelist[12*N:13*N]
    # filelist = filelist[13*N:14*N]
    # filelist = filelist[9800:10000]
  if length_filelist < load:
    load = length_filelist
  # print(filelist)
  data = np.zeros((M*load, SCREEN_SIZE_X, SCREEN_SIZE_Y, GLAY_SCALE), dtype=np.float32)
  idx = 0
  file_count = 0
  # M_obstotal=0
  
  # for file in filelist:
  
  #       new_data = np.load(DIR_NAME + file)['obs']
  #       M_obstmp =new_data.shape[0]
  #       M_obstotal += M_obstmp 

  # data = np.zeros((M_obstotal*load, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)
  
  for file in filelist:
      try:
        new_data = np.load(DIR_NAME + file,allow_pickle=True)['obs']
        # M_obs =new_data.shape[0]
        data[idx:(idx + M), :, :, :] = new_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
          print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, load, idx))
      except Exception as e:
        print(e)
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, load, idx))

  return data, load,file_count



def main(args):

  new_model = args.new_model
  N = int(args.N)
  M = int(args.time_steps)
  epochs = int(args.epochs)
  load=int(args.load) #load episodenum
  loadnum=int(N/load)
  print(f'loadnum:{loadnum}')
  vae = VAE()
  checkpoint_path=f'./vae/vae_weights/weights_custom3dmaze_gray2{N}.{epochs}z32k3conv1r100000lr0001.ckpt'
  for epoch in range(epochs):
    print('EPOCH ' + str(epoch))
    for i in range(loadnum):
      if not new_model:
        try:
         
          # vae.set_weights(f'./vae/weights_custommaze{N}.{epochs}z40.ckpt')
          vae.set_weights(f'./vae/vae_weights/weights_custom3dmaze_gray2{N}.{epochs}z32k3conv1r100000lr0001.ckpt')
          
        except:
          print("Either set --new_model or ensure ./vae/weights3.h5 exists")
          raise
      # else:
      #   print("newcheck")
    
      try:
        data, load ,filecount= import_data(load, M,i)
        
      
      except:
        print('NO DATA FOUND')
        raise
          
      print('DATA SHAPE = {}'.format(data.shape))
      
      if filecount==0:
        print(f"filecount 0 :{filecount}")
        break
        
       
      print(f"number of loading:{i+1}/{loadnum}, epoch:{epoch+1}/{epochs}")
      vae.train(checkpoint_path,data)
      vae.save_weights(f'./vae/vae_weights/weights_custom3dmaze_gray2{N}.{epochs}z32k3conv1r100000lr0001.ckpt')
     
      
      K.clear_session()
      gc.collect()
      
      # time.sleep(2)
    
  vae.save_weights(f'./vae/vae_weights/weights_custom3dmaze_gray2{N}.{epochs}z32k3conv1r100000lr0001.ckpt')
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--time_steps', type=int, default=1000,
                        help='how many timesteps at start of episode?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  parser.add_argument('--load',help='number of episode loading at onetime')
  args = parser.parse_args()

  main(args)
