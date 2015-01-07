Image Processing
==================================
Date: 1/7/15

### Initial notes
- freesurfer changes data to 8 pt int
- watch out with fslmaths, might output int from decimal input!

Data storage & other details
-----------------------------------------------------------
### Normalize first or after running stats?
- FSL normalizes later, after stats...why?
  - native 64 x 64 x 36 image --> much greater dimensions in MNI
    - keep smaller image size (i.e., native space) longer in FSL to save space
- in SPM convert to standard space sooner, and then run stats after

### Talairaich vs. MNI
- French alcoholic woman's brain vs. avg over 305 individuals
- diff coordinate spaces
- auto-talairach in AFNI is really in MNI!!

Spatial transformations
-----------------------------------------------------------
- usually automated (unless using landmarks)

### **affine transforms** (linear, parallel lines remain parallel)
- translate, rotate in same plane, then resample
- estimate parameters:
  - 2 params for translation, 1 for rotation
- 3-12 params, pretty fast to apply
- Types of affine transforms:
  - translate (x+trans), rotate (x*cos(theta)-y*sin(theta)), scale (x*s), shear (x + y*shear)
- 6 DOF transform
  - translate & rotate in x, y, z
  - not change the shape
  - good for within subj alignment
- 7 DOF
  - same as 6, but also scale
  - use to go from T2* to T1 image (voxel size might be off), with boundary based registration
- full affine = 12 DOF (4 types * 3 dim)
  - common for registering between subjs, even if shape a little off
  - use to go from native T1 to MNI152

### **nonlinear transforms**
- more params, take longer, overfitting issues...
- more localizer transformation
- e.g., polynomial basis functions
- regularize to avoid overfitting

How to carry out transformations
-----------------------------------------------------------
- e.g., from subj T1 to MNI152

### **Cost functions** (how similar are the 2 images?)
- **Least squares**
  - If have similar images, with similar values and image intensities it works (e.g., 2 similar T2s)
- **Normalized correlation** (default in FSL for motion correction)
  - how well correlated are the values?
  - Great for comparing 2 images of the same type, like 2 T2s, but not good for between T1 and T2...
  - ok for motion correction, but large functional activations could also affect this and lead to errors during motion correction
- **Mutual Information**
  - how well can you predict the values in one image from the values in another?
  - entropy! how much randomness is in the signal
  - joint entropy: relative value of each of the images, plot values against each other, and should see a line
  - we want to minimize the joint entropy
  - Mutual information: entropy of each minus joint entropy...so we want to maximize the mutual information
  - Normalized MI = sum of both entropies / joint entropy
  - Correlation ratio (Default in FSL for going between T2 and T1)
    - nonlinearity in relationship between T2 and T1
    - penalize variability in each bin
    
### **Optimization**
- **grid search**
  - ok if you want to drain power
- **gradient descent**
  - watch out for local minima
  - regularize!
    - penalize complicated nonlinear warps
    - usually first do just affine, and then add in nonlinearities
- **multiscale optimization**

### **Reslicing & Interpolation**
- **Nearest neighbor**
  - match with the best voxel
  - lose resolution, but good for when transformed image values need to match the original image
  - blockier looking
  - ok for masks, or atlases
- **Linear interpolation**
  - take weighted average of transformed voxels
  - integrates over nearest 8 voxels in 3D
  - might blur the image
- Higher order interpolations
  - **Sinc interpolation**
    - windowed sine, e.g., sin(x)/x between some x bounds
  - [**Spline interpolation**](http://en.wikipedia.org/wiki/Spline_interpolation)

Fourier Analysis
-----------------------------------------------------------
- remove certain frequencies from the data ()
- highpass filtering
  - get rid of low frequency noise
  - calculate power from design
- lowpass filtering
  - done for resting state, theoretically removes physio noise
