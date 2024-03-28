import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from options.train_options import TrainOptions
import models.networks as models
import utils.test_utils as utils
import nibabel as nib
import pickle
from tqdm import tqdm
from models.networks import ReflectionPadding2D, ReflectionPadding3D
from keras.models import load_model


# Predict porosity volume
def predict_porosity_3d(model, data, sqrt=False):
  """
  Return entire volume 256^3 of porosity computed from MRI input.

  :param model: generator model.
  :param data: dataset to compute porosity from.
  :sqrt: set to True if model was trained to predict the square root of the porosity.
  :returns: generated porosity volume.
  """

  X, M = data['x'], data['m']

  temp_vol = np.zeros((256,256,256,1))
  nb_sums = np.zeros((256,256,256,1))
  
  blocks_per_axis = int((256 - 64) / 32 + 1)
  s = 32 # side of a block
  
  for i in tqdm(range(blocks_per_axis)):
    for j in range(blocks_per_axis):
      for k in range(blocks_per_axis):
        block = model.predict(X[:,i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s])
        if sqrt:
          block = np.square(block)
        temp_vol[i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s] += block[0]
        nb_sums[i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s] += 1
          
  averaged_vol = np.divide(temp_vol, nb_sums)
  gen_vol = np.multiply(averaged_vol - 1, M[0]) + 1
  
  gen_vol = (gen_vol + 1) / 2
  
  return gen_vol

        
PATH = os.getcwd() 
model_path = PATH + '/checkpoints/ctmask_l1_l2_3d'
data_path  = PATH + '/datasets/ctmask_nosqrt_3d'
nifti_path = PATH + '/inference/'


print( '*** Model path: ' + model_path )
print( '*** Trained weights: ' + data_path )
print( '*** Data path: ' + nifti_path )


# Data Preprocessing
print("*** Preprocessing...")
with open(os.path.join(data_path, "dataset_info"), "rb") as file:
  dataset_info = pickle.load(file)

# retrieve scaler for MR images
mm_t1 = dataset_info["mm_t1"]
# retrieve scaler for masks
mm_mask = dataset_info["mm_mask"]


# load model
print("*** Loading model...")
model_name = "ctmask_l1_l2_3d_g.h5"
model = load_model(
    os.path.join(model_path, model_name), 
    compile = False, 
    custom_objects = {"ReflectionPadding2D": ReflectionPadding2D, "ReflectionPadding3D": ReflectionPadding3D}
)

# Retrieving nifti files
print("*** Retrieving nifti files to convert...")
subjects_list = []
for file in os.listdir(nifti_path):
  print(file)
  # if os.path.isdir(file):
  subjects_list.append(file)

# inference loop
print("*** Starting inference...")
# progress_bar = tqdm(total=len(subjects_list), desc="Processing")

for subject in tqdm(subjects_list):

 #  progress_bar.set_description(f"Processing: {subject}")

  path1 = os.path.join(nifti_path, subject, "t1.nii")
  print( 'T1 path = ' , path1)
  
  t1 = nib.load(path1)
  mask = nib.load(os.path.join(nifti_path, subject, "mask.nii"))

  t1 = t1.get_fdata()
  mask = mask.get_fdata()

  t1 = t1.reshape(256,256,256,1)
  # print( '*** t1 shape = ' , t1.shape )
  mask = mask.reshape(256,256,256,1)
  # print( '*** mask shape = ' , mask.shape )
  
  t1_mm = mm_t1.transform(np.expand_dims(t1[...,0].flatten(),-1))
  t1 = np.reshape(t1_mm, t1.shape)
  mask = np.where(mask > 0, 1, 0)

  t1 = t1.reshape(1,256,256,256,1)
  mask = mask.reshape(1,256,256,256,1)

  input = {"x": t1, "m": mask}
  output = predict_porosity_3d(model, input)

  # save generated volume as a nifti
  output_nifti = nib.Nifti1Image(output, affine=np.eye(4))

  # Save the NIfTI image to a file
  nib.save(output_nifti, os.path.join(nifti_path, subject, 'poro.nii.gz'))

  # progress_bar.update(1)
  
# progress_bar.close()
print("*** Done.")
