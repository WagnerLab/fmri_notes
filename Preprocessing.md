Preprocessing
==================================
Date: 1/14/15

Stream
-----------------------------------------------------------
### Basic pipeline (check data between steps)
- Reconstructed image data
- Sometimes do fieldmaps (be careful if it helps or hurts)
- Motion correction
- Slice timing correction (optional)
- Spatial smoothing (optional; on volume or surface)
- Stats

- not including normalization as preproc step

### Quality control
- Use eyes to look at data!
- Possible problems:
  - **Missing data** from half the brain?
  - **Spikes** in fMRI data
    - signal at very specific frequency
    - problem in k-space
  - **Ghosting**
    - offset in phase between different lines in k-space in EPI acq
    - could be worse in MUX
- How to:
  - Create **movie** in fslview
  - **Model free analysis**
    - ICA, locate artifacts in the data
      - e.g., melodic in FSL
      - find independent non-Gaussian things
        - physio artifact (periodic)
        - movement (component response spike)
          - good for non-rigid body movement, esp that where movement happens between 2 sequential TRs
      - Remove from data
        - Using melodic just pull out the components
        - Add component time series to GLM
          - not ideal if a lot of components
          - Check version of FSL, might do unnecessary matrix inversion and use a ton of RAM

### Distortion Correction
- Air+tissue boundary causes inhomogeneity
  - e.g., ears, sinuses
  - miss parts of brain, and stretches some areas
    - due to phase encoding, can combine a bunch of voxels in real space into one in image space
- When?
  - at beginning/end, or interleaved between runs
  - the more you average together the better off you are
- How?
  - Collect field maps, perhaps multiple
    - register field maps
  - map out inhomogeneity in magnetic field, and then fix it
  - map of magnitude and relative phase between 2 echoes
  - unwrapping

### Slice Timing Correction
- Slices are collected in sequence, top to bottom, interleaved, etc.
  - interleaved
    - can't make perfect square, so collecting slice B also includes some info from A and C.
- worst for ER designs, not block, since each second matters more
- Resample dataset back to one point in time
- Caveats (esp with shorter TRs (<2s) & interleaved acq)
  - If have spike, could smear across data with sinc interpolation
  - Should tweak techniques for MUX!
- Could add temporal derivatives to the model
  - help with slice timing problems to soak up that variance

### Motion correction/realignment
- Retrospective (after the fact)
- Prospective: Siemens can do this during the scan
  - with cameras to track head motion, and then update the slices
- Doesn't do a good job with:
  - physio noise
    - to get rid of that, can do cardiac gating, e.g., if imaging brain stem
    - record heart beat and respiration and use those to remove that artifact
    - can get aliased signal since stuff like respiration is relatively fast, can get low frequency in images
- Bulk motion
  - head movement, where whole head moves
  - artifact near edges of brain, and near ventricles
- Spin history effects
  - spin history of voxel will be in new area after movement
  - might be able to correct with SPM
- Might be hard to break apart signal from motion
  - But timing matters! motion should happen immediately, and then brain activation later
- How to fix: Realignment
  - Rigid-body, 6 param registration
    - x,y,z translation, rotation
  - How to visualize:
    - Try plotting the derivatives, so not looking at slow drifts that are plotting in absolute diff
    - Or plot framewise displacement, to collapse across 6 params
    - Also plot intensity changes over time
  - Use middle image from each run as the target image
    - first image could be messed up
    - could use mean image, but is more blurry, and extra calculation
  - Cost function:
    - FSL: normalized correlation error
    - caveat! if signal is strong, might look like motion
      - could use mutual information instead
  - Interpolation
    - linear: more spatial smoothing, but fast
    - Higher order methods:
      - sinc, spline, fourier-based
- Other
  - relationships between motion and susceptibility artifacts
    - rigid-body motion correction won't fix
  - motion or slice-timing correction first?
    - if slice first, could move voxels into different slices
    - if motion first, could be errors in motion intensity spread out over time
    - nipype tool 4drealign does both at the same time! but takes a long time. should work in multiband
  - When to throw out data
    - maybe not pick a given threshold, unless too many corrupted timepoints
    - 1/2 voxel
  - 
