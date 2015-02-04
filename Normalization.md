Normalization
==================================
Date: 2/4/15

### Initial notes
- Spatial normalization: putting people into some type of standard space

Talairach space
-----------------------------------------------------------
- 3d Cartesian coordinate space
- Based on anatomical landmarks
  - AC, PC, midline sag, bounding box, coronal plane to create multiple boxes
  - shove brain into box aligned across brains
  - modify each box independently, piecewise linear transform
- Talairach atlas is based on 1 person's brain, so not great to fit many people

Atlas vs. Template
-----------------------------------------------------------
### Atlas
- Show anatomical locations in a coordinate space
- Examples
  - Harvard Oxford, AAL Atlas, etc.
- Talairach atlas
  - postmortem drawings?

### Template
- Example: MNI152 (152 people registered together)
  - Can align images to this target template
  - Supplies a coordinate system
- MNI305
  - 9 dof affine registration
  - align the 305 images to Talairach atlas
- ICBM 152/MNI152
  - Brains from around the world
  - Register high res images to MNI 305 template
  - different from MNI305!
  - nonlinear registration?

Preprocessing anatomical images
-----------------------------------------------------------
## Bias field correction
- gradient in brightness across image
- HP filter and/or bias field correction
- Can impact anatomical images & influence registration

## Brain extraction
- Freesurfer (make_mask, or fs.ApplyMask for nipype) works better than BET (FSL)
- BET (FSL)
  - starts with a ball in center of the brain, iteratively expands

## Tissue segmentation
- Separate gray, white matter & CSF
- Could save as regressors for CSF, etc.
- Make sure do bias field correction
- Can't threshold images to separate
  - distributions of image intensities overlap
  - voxels might be on the boundaries
- Technique:
  - Unified segmentation (Ashburner & Friston, 2005)
  - Put brain into MNI space, and have priors on each voxel for whether gray, white or CSF


Normalization
-----------------------------------------------------------
## When?
- prior to stats (SPM) -- normalize data
- post stats (FSL) -- normalize stats
- Pros/Cons:
  - File sizes are smaller if normalize post-stats
  - Can do different types of normalization once you have your stats quickly

## Which images are used?
- Can register EPI to EPI template, but not great since not much contrast in EPI
- Or, 2-step procedure, register functional to anat, and anat to T1-weighted, and then concatenate both
  - this limits the accumulation of interpolation error
- 3 step procedure
  - collect coplanar to cover same slice as func, and register to that, then to anat, etc.
  - bbregister is better

## How do you realign?
- Landmark based
  - need anatomy expert
- Volume-based registration
  - MI (SPM) or normalized correlation (FSL)
- Diffeomorphic transformation
  - Treat brain like viscous fluid registration
  - Penalizes major warps
- Warpfield (nonlinear registration), ANTS
  - Take brain on square grid, show how voxels get pushed around, done in 3D space or on the surface in 2D space
- Surface based methods
  - Rely on sulci and gyri
  - Automated, but check for handles and donut holes!
  - Register to surface atlas
  - Map fMRI data to surface space
  - Appears to be more accurate
  - Fine, but only good for cortical surface
  - CIFTI files: stores surface and subcortical gray matter (grayordinates)
    - could do analysis on all of the data
- Which to use?
  - Maybe nonlinear is better, some examples where FNIRT shows more activation than FLIRT
  - Klein 2009 says nonlinear
  - ANTS might be the best, but annoying to install/use

## QA
- T1-normalization
  - View individual subjects normalization on top of template
- Average normalized brains (e.g., all T1s)
  - Should look like a blurry version of the breal brain
  - Good for large datasets where time/brain is limited
- Is orientation correct?
- Double check that brain extraction/skull stripping got everything

## Different ages
- Children
  - Pediatric templates
    - only good if have a single age group
    - common methods are robust to age differences, at least for 7+
- Elderly
  - Decrease in gray matter, increase in CSF
  - Create custom templates
- Lesions
  - Account for lesions in cost function
    - leave the lesion spots out of warp
    - might impact surrounding structure too
