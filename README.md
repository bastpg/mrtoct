# mrtoct
mrtoct is a robust DL network developed by Bastien Guerin and Matthieu Dagommer at the MGH A. A. Martinos Center for Biomedical Imaging for estimation of cortical bone porosity from T1-weighted MR images. Robustness was enforced in the implementation, so the tool should work for a wide range of T1-weighted images and acquisition parameters (although there are no guarantees of the accuracy of results). To improve robustness of the estimation process, the network was trained by focusing backpropagation to voxels located inside a mask of the skull. This mask is estimated using SAMSEG, which is itself a robust tool (reference: Puonti, Oula, Juan Eugenio Iglesias, and Koen Van Leemput. "Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling." NeuroImage 143 (2016): 235-249, webpage: https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg). Obviously, the skull mask needs to be 'liberal' so as not to reject edge skull voxels, which is the reason why we perform a thickening operation in the preprocessing step. The output of the program is the distribution of bone porosity within the mask, which is in [0;1]. The porosity is a useful metric to estimate subject-specific acoustic parameters for transcranial focused ultrasound applications.

# Dependencies
The followed are utilized in this program: Freesurfer, iso2mesh, tensorflow, tensorflow-addons, nibabel.

# Step 0) Create the mrtoct Python virtual environment
Upon using mrtoct for the first time, create a new environment by typing:
```
conda create --name mrtoct_env
source activate mrtoct_env
```

Then install the following dependencies:
```
conda install tensorflow
pip install tensorflow-addons
pip install nibabel
```

Finally quit the environment:
```
conda deactivate
```

Check that the new environment indeed exists using the command:
```
conda env list
```
![image](https://github.com/parkerkotlarz/mrtoct/assets/157265957/1af89de8-6f1e-4521-aac8-c14c67e77f36)

# Step 1) Data preprocessing

In this step, we transform the T1 volume into the standard Freesurfer 1x1x1 mm^3 volume and we segment it using SAMSEG (both actions are done at the same time). You need to have Freesurfer installed. 

In a **terminal**, type:
```
samseg --t1w T1_image_name --o ./ --threads 10 --refmode t1w % Replace the input T1 volume by the correct name (can have any format known to Freesurfer and change the number of threads if necessary
```

Then, we create the skull mask. In the **terminal**, type:
```
mri_convert input/t1w/r001.mgz t1.nii
mri_convert seg.mgz seg.nii
```

Then in **MATLAB**, type:
``` MATLAB
header = load_nifti( 'seg.nii' );
smask = double( header.vol == 165);  % create skull mask from SAMSEG segmentation output
smask = imclose( smask , strel('sphere',5) );  % close the skull mask
smask = thickenbinvol( smask , 3 );  % make sure the mask covers a bit more than the actual skull (liberal mask), this step requires the iso2mesh open-source Matlab package
header.vol = smask;
save_nifti( header , 'mask.nii' );  % save skull mask as NIFTI to be used as input to mrtoct
```

# Step 2) Move data in the correct location 
The files to be processed should be placed in the inference/ folder in the main mrtoct/ directory. Each subject has its own folder, which can have any name you want. There are two files per subject/folder: t1.nii and mask.nii. Those file names are fixed and are created in Step 1).

# Step 3) Run
Activate the Python environment by typing 
```
conda activate mrtoct_env
```

Then simply type 
```
python3 inference.py
```

















