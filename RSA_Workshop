RSA Workshop
========================

Distance measures
------------------------

### What does RSA measure?

- GLM to get beta for each voxel x condition
- e.g., OLS fitting

### Euclidean distance

- vector of values for each voxel, for each condition
- some point in n-dim space (sum of squares of components)
- Difference = sum of square differences between points in voxel space

### Pearson correlation distance

- Distance of points from origin = univariate effect, difference from baseline
  - Could just subtract the mean from each vector, and move the voxels to plane
- Some voxels might be relatively less active compared to others
  - z transform --> divisive normalization of voxel variance
  - project to circle?
  - 1 - cos(angle between A and B) = distance

### Issues w/ Euclidean distance & Pearson correlation

- baseline shift can change angle between 2 points, but not Euclidean distance
  - try subtracting out mean pattern (but makes patterns anticorrelated!)
- change the pattern scaling (make B slightly stronger)
  - changes Euclidean distance, but not correlation

### Reliability of dissimilarity measures

- split-half reliability analysis
  - similar reliability between 2 metrics

Noise Normalization for fMRI
------------------------
- Some voxels are noiser than others
- Could weight voxels by noisier (e.g., using residuals from epsilon matrix) ~ t-stat
- Voxel by voxel covariance matrix, variance on matrix, weight $\beta$ = $\beta$/$\sigma_{voxel}$
- Could normalize by whole covariance $\Sigma$ matrix --> huge increase in reliability
  - u = beta * $\Sigma^{1/2}$

CV distance estimates
------------------------
- Noise drives distances further apart (although ranks of distances are preserved)
- Multiple runs in the experiment
- Each estimated voxel pattern = true pattern + noise
  -  run specific noise, that is independent between runs
- estimated (u_a - u_b) from run 1 * (u_a - u_b)^T from run 2
  - since noise is independent between runs, can call CV distance the true distance
- average across folds if can have more folds (e.g., leave 1 out)
  - 1/4(d_1 (d2+d2 + d4)/3^T)
- CV Mahalanobis distance has better split-half reliability than multivariate noise normalization when SNR is low
- Multivariate noise normalization > CV MD > distance!

- For each cell in RDM, random effects t-test (vs. 0) across subjects, then FDR correct, threshold
- Fixed effects within subject, can get Linear discriminate t-value, by normalizing the distance by standard error
  - ratio between values is no longer preserved, because SD might differ between diff contrasts

### Distances measures vs. Pattern classifiers

- Instead of RDM, do pairwise classification accuracies, and insert into matrix
- But pattern classifiers are quantized (broken into bins)
- If doing binary classification, then fine, but not for investigating representation


RDM Practical
------------------------
- 3 main steps:
  1. compute/visualize RDMs from different brain regions
  2. compare brain and model RDMs
  3. statistical inference

### `DEMO1_RSA_ROI_simulatedAndRealData.m`

  - vectorized RDM is way to convert symmetric RDM to vector (N(N-1))/2
    - squareRDMs() to convert back to full RDM
  - Naming convention for RDMs:
    - structure = RDM
      - field: RDM = dissimilarity matrix
      - field: name = 'hIT | BE | Session: 1' (separate info w/vertical bars)
      - field: color = [0 0 1]
    - averageRDMs_subjectSession(RDMs, 'session'), can avg over name id
  - **MultiDimensional Scaling (MDS)**
    - show each condition with plot, and specify color of the plot
    - project data into 2D space, and visualize on 2D circle
    - Shepard plot: shows similarity vs. disparity
    - Pearson & Spearman rank correlations
    - `MDSConditions(cell name (has name, color, RDM))`
    - Dendrogram: avg RDMs across subjects, and can visualize categorial stucture
      - hierarchical clustering (exploratory!)
      - plots smalled colored dots that match up with MDS plot
      - `dendrogramConditions()`
  - **Comparing RDMs (brain vs. model)**
    - Create RDM that is dissimilarity between separate RDMs
    - How to quantify correlation
      - Between values: Pearson, rank correlation (e.g., Spearman, Kendall's tau a)
      - Between correlation matrices: Kendall's $\tau$ a is the correlation measure for 2nd order correlation matrix
        - Pearson assumes linear, but should use rank correlation, e.g., how similar are the ranks
        - Spearman correlation = correlation of the ranks (rank dissimilarities, and then compute pearson corr)
          - doesn't penalize
        - Kendall's tau a gives more bonus to the more specific predictions
          - within RDM get 2 dissimilarities, see if one is larger than the other, repeat for the other matrix
          - If the 2 RDMs match, then give a rank of 1, otherwise 0. Then, get proportion of 1s for all pairs of dissimilarities for each cell
          - allow for detailed predictions (.7 in 1 condition, .8 in the other)
    - Correlation between RDMs: `pairwiseCorrelateRDMs(cell array {models})`
    - MDS for RDM-RDM similarities: `MDSRDMs()`
    - 'subjectRFXsignedRank'
    - `plotpValues = '='` or `'*'` (write pval, or * for significance)
    - Height of bars is average correlation across subjects
    - Threshold for multiple comparisons
      - FDR, or Bonferroni (N; 2)
    - When comparing 2 RDMs, and not enough evidence for base RDM, could do bootstrapping
      - `userOptions.candRDMdifferencesTest = 'conditionRFXbootstrap'`
    - Noise ceiling: Compute threshold (gray) for maximum correlations for this dataset
      - Upper bound
        - best you can do is at the mean of all the subjects (group mean of average RDMs is the best)
        - for RDMs rarely use Euclidean distance, but rather pearson correlation (or spearman, etc.)
          - but 1-r = $\sqrt{2}$ * Euclidean distance
          - for spearman, do the same thing as pearson, but rank transform first
          - Kendall tau a --> gradient descent (maximum correlation)
      - Lower bound
        - Leave 1-subject out, average RDMs for n-1, and repeat for all folds, then average
        - Underfit (so lower bound on ceiling), but still pretty good
      - If model is between lower and upper bound, doing pretty well.
      - But, what if data noisy?
        - Might be useful to report the value of the ceiling, since can't really do much better, given the noise.
        - Upper bound statistic: `stats_p_r.ceiling(2)`

### Simulate fMRI data
- `DEMO1_RSA_ROI_sim.m`
- Edit `betaCorrespondence.m`
  - Might want to use t statistics, rather than betas


Multivariate noise estimation & LD-*t*
------------------------
- Example:
  - finger representations in M1
    - thumb, index... pinky
    - patterns are unique(ish) for each finger, but within subject
    - pattern is different in subject 2 (r between subjects is low)
    - *but* relationship in peaks is somewhat consistent across subjs

### Compute cross validated distance metric

- $(u^1-u^1)(u^2-u^2)^T$
- 1 ROI, 2 hemispheres, 2 hands, 6 subjects, 10 conditions, 8 runs
- Estimate betas, residuals, and multivariate noise normalization
- `rsa_noiseNormalizeBeta(Y,SPM)`, where Y: timeseries after preprocessing
  - Estimate beta coefficients and residuals from preproc time series
      - ordinary least squares estimate of beta_hat = inv(X'*X)*X'*Y
      - residuals: res  = Y - X*beta
  - Run-wise multivariate noise normalization
    - For each run, estimate the residual variance for each voxel (Sw_hat)
      - `covdiag(res(indices_run_i,:))`
      - ~100 time points per 1050 voxels; if just cov, then rank deficient, and need to be able to invert
      - this applies shrinkage estimation, so covariance matrix of resids (voxel to voxel), and optimal estimation of shrinkage
    - Mutlivariate noise normalization
      - `beta_hat(indices_run_i,:)*Sw_hat(:,:,run_i)^(-1/2)`
- Calculate the CV squared Euclidian distances
  - Mahalanobis distance (LDC)
  - Compute pairwise distances
  - Leave 1-run out, and cross validate
    - Take inverse of X, and apply to partition (10xnumber_voxels)
      - `A(:,:,i) = pinv(Xa)*Ya; B = pinv(Xb)*Yb;`
    - Compute the difference between these in each partition
      - `d(i,:) = sum((C*A(:,:,i)).*(C*B),2)';` (1x45 vector --> 10 x 10 matrix)
  - Take average across CV partitions
    - `d = sum(d)./numPart;`
- Show dissimilarity matrix, rank-transformed, scaled (0,1)
  - if rank transformed, then highlights small differences; analyses should be done on raw values

### Compute linear discriminant t value (LDt)

- t-value that tells us how likely patterns are truly dissimilar
- Normalize contrast by $SE_{LDC}$
  - Take diagonal of covariance matrix of noise (variance), and multiply by training w (u_A-u_B)
  - `A(:,:,i) = pinv(Xa)*Ya;
    B        = pinv(Xb)*Yb;
    w=(C*A(:,:,i));
    d(i,:)   = sum(w.*(C*B),2)';
    BCovb=SPM.xX.Bcov(partition==part(i),partition==part(i));
    sigma=sum(w*Sw(:,:,i).*w,2);
    se = sqrt(sigma.*diag(C*BCovb*C'))';
    d(i,:)=d(i,:)./se;`
- Can easily threshold matrix w/p-value for visualization purposes

Weighted Representational Component Models
------------------------
(Jorn Diedrichsen, UCL)

### Motivation

- How are different features of a stimulus (e.g., color, shape) observed in neural activity?
  - How many voxels are activated by a given feature?
  - Approach:
    - weighting of different features
    - for instance, red might not be coded for, but blue a lot

### Covariances & distances

- $/beta$ matrix: k conditions X p voxels; each row, u, is a pattern of voxels
- Compute distances (d) between patterns, u = inner product of $u_i-u_j$
  - u_{i}u_i^T _ u_{i}u_{i}^T - 2u_{i}u_{j}^T
  - Pattern "covariance": G = U{i}U{j}^T, related to pattern distance: D = G + G - G - G
    - Can go from covariance/inner product matrix to the Mahalanobis-distance matrix, just need the origin
    - Make sure sum of rows/columns = 0
    - If you can prove something for covariance matrix, applies to distance matrix too

### Features & Representational components

- Each pattern ($u$) is a sum of different features ($f$) times a feature pattern ($w_h$)
  - As feature weighting changes, how does RDM change?
- $G = UU^T$ = $\Sigma f_hw_h \Sigma w_h^Tf_h^T$
  - if feature patterns are independent, then $w_i w_j^T = 0$
  - = $\Sigma f_h W_h W_h^Tf_h^T$, since cross sums cancel, if weight (w) inner product = 0
    - strength with which feature is represented
    - how stimuli differ from eachother (f_h f_h^T)
  - G = $\Sigma \omega_h G_h$ ($\omega = w_h w_h^T$)
  - Distance matrix (d) = $\Sigma \omega_h D_h$
- How well does a neural area represent part of the model, rather than estimating weight for each feature
  - could group features into representational components
  - sometimes we want to think of the features as dependent, loading onto a "component"
  - beta matrix = sum of F_h (k conditions x q features) * w_h (q features x p voxels)
  - Example:
    - 5 different stimuli (conditions 1-5, linearly increasing brightness) = F_h
    - Features correlation, V_h = 1
    - Covariance, G_h - F_hV_hF
    - Distance component = D = G_i+G_j-2G_{ij}
  - Example 2:
    - Faces, places, different features
    - Correlation matrix = [1 0; 0 1]
    - Covariance: faces related to each other, not places, vice versa
    - Distance matrix = reverse of covariance matrix
- Features don't really matter, its the component matrices (covariance & distance) that you get out
  - The exact values of features doesn't matter too much, many different weightings of feature correlations give rise to same component matrices
- ** Estimation**
  - D = $\Sigma \omega_h D_h$
  - linear regression!
  - Vectorize distances, then build component matrix X [d_1 d_2]
  - $\omega = (X^T X)^{-1} X^T d$

### Factorial Models (MANOVA)

- Integrated vs. independent encoding of features?
  - How to test?
    - Vary both factorially in design, and test for interaction
    - Where is factor A encoded? where is B encoded?
- Example:
  - A: grasp, pinch, grip; B: see, do
  - Start with features:
    - Factor A: 6 x 3 ([1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1])
    - Factor B: 6 x 2 ([1 0; 1 0; 1 0 ...])
    - Interaction: A * B
  - Representational components
    - component matrix for each feature
  - Determine weights for each component
    - string representational components together into component design matrix (X)
  - Get data
    - Y --> betas --> pattern estimates --> D (RDM)
  - Bring representational components and D together w/linear regression
    - $\omega = (X^T X)^{-1} X^T vec(\hat{D})$
  - $(X^T X)^{-1} X^T$ yields interesting matrices
    - Similar to cross "decoding": train classifier on see-grasp, test on do-pinch/grasp
  - Get component weights for each factor (simulate over many iterations)
    - if weights around 0, then doesn't code for that factor.
- Models for BOLD signals adding
  - patterns engage independent neural subpopulations
  - combine linearly to determine firing rate
  - relationship between neural activity and BOLD is approximately linear
    - NOT true! Log
  - BUT, if within the range of a normal task, then work around that range (small % signal change), then its small variation around signal, and linear approximation of nonlinear function is fine

### Linear representational models

- **Coding of action sequences** (Yokoi et al., in prep)
  - Set up
    - sequences consisting of chunks
    - where are sequences encoded in motor areas?
    - what is the nature of chunks?
      - finger presses, chunks, 2nd level chunks, entire sequence?
  - From features, create representational component matrices
    - feature 1: finger transitions for each sequence --> covariances --> distances
    - feature 2: chunks (8), sequence # (11) x 8 chunks --> covariances --> distances
    - feature 3: 2nd order chunks ...
    - feature 4: sequences
    - **Note**: Component matrices are not orthogonal, so need to worry about that, but try to maximize non-orthoginality in design
  - OLS
    - assume observations are independent (I)
    - equal variance (I)
    - normally distributed (D)
    - If this is violated, then have an unbiased estimator, but not the best linear unbiased estimator (BLUE)
      - not the one with lowest variance
      - If distances are non-zero, variance/covariance go up. If add noise, variance/covariance goes up
        - variance($\hat{d}$) = true variance of $dd^t$ + distance dependent variance * distance ($\delta \delta^T$) + constant dependent on noise
        - related to how you do cross validation
          - usually maximize & exhaust CV
          - If low SNR, independent noise dominates; otherwise other term dominates
  - Can do iterated reweighted least squares (IRLS; since variances are dep on true probabilities)
    - start with some guess of component weights ($\omega$)
    - predict distances: $\hat{d} = X\omega$
    - calculate variance-covariance of d
    - use this in estimation, OLS, get best estimate
    - repeat til convergence2
    - moderate improvements in SD, but depends on your model how much it'll improve estimates over OLS

### Nonlinear representational models

- Sometimes model components are non-linear functions of parameters
- Example:
  - area might have narrow tuning curve, or wider (tuning widths change)
  - can express as feature vectors
  - change between distances and feature matrices are nonlinearly related
  - How to solve:
    - could have linear approximation (model 1st derivatives), like AR-estimation in 1st level SPMs
    - estimate nonlinear parameters to optimize log-likelihood, e.g., log p(d|$\theta$)

Advanced Topics in RSA
------------------------
### Calculating the noise ceiling

- when to search for better model (not at noise ceiling yet), or better data (in noise ceiling)
- Correlation distance proportional to squared Euclidean distance (for normalized patterns)
- Comparing RDMs
  - use correlations, not distances, mostly since want a bigger bar for better model
- Norm pattern vectors, so length = 1, with angle $\alpha$
  - d is Euclidean distance between them; cos($\alpha$) = r, distance is 1-r
  - d^2 = 2(1-r)
- Max = group mean RDM
- Min = leave one out RDMs

### Kendall's tau a

- Important when you have categorical models that predict tied dissimilarities (e.g., if predict 1 for within and 0 for between categories)
- Have values for x and y
- compute 2 difference matrices: a (differences between all points on x), b (diff between all points on y)
- coeff = normalized inner product of a and b (just Pearson)
- Kendall -> signed differences (concordant or discordant pairs, so 1 if both go up/down, and 0 if one does up, one goes down)
- Spearman --> ranked differences
- Using pearson or spearman, true model can lose to simplified step model, but using Kendall $\tau_a$, true model can't lose

### Remixing/reweighting representational features

- Might not know how brain-activity measurements mix and weight representational dimensions
  - Local averaging w/voxels in fMRI
  - sparse, biased neuronal sample in cell recordings


Practical: Weighted representational component models
------------------------
- Toolboxes:
  - rsatoolbox.v.2a
  - DataFrame
    - toolboxes from Jorn's lab for graphing
- MANOVA design
  - Define how conditions related to the features
  - Make representational model matrices
    - `rsa_modelRDMsMANOVA(F_A, F_B)`
  - Visualize pseudo-inverse matrices for visualization purposes (what's going into regression)
  - Get prewhitened data from ROIs
    - get $\beta$s: `rsa_noiseNormalizeBetas()`
    - get CV distances: `rsa_ldc()`
      - NOTE! Distances should be LDC. This might not hold for correlations, LD-t, etc.
  - Plot for each subject for visualization
    - don't want to do rank transform
    - Can apply nonlinear function to data before plotting: `rsa_figureRDMs('transformFcn', 'fnc_name')`
      - e.g., 'ssqrt' -> sign preserving sqrt
  - Plot average RDMs across subjects
    - `rsa_flattenRDMs(RDMs)`
    - R-like function to average over subjects: `T=tapply(Data,{'regType','regSide'},{'RDM'})`
    - Put into RDM format: `avrgRDMs=rsa_foldRDMs(T);`
    - Plot: `rsa_figureRDMs(avrgRDMs,'rankTransform',0,'transformFcn','ssqrt');`
  - Estimate component weights w/normal OLS regression
    - Basically multiply each RDM by the 3 M RDM models
    - `rsa_fitRepModelRegress(M,RDMs,'method','OLS')`
      - Arguments: Model, and RDMs (subject x condition x ROI),
      - Returns: omegas, 1 per Model (e.g., 448 x 3)
      - Flatten Model if necessary (# models X 36)
      - Y = (448 X 36)
      - OLS
        - `omega = Y*pinv(X);`

- Hierarchical design/Linear model
  - Making models from features
    - Features:
      - Rows = sequences, and columns = features from sequences
      - Each sequence might be represented
        - e.g., I-matrix (size 8, or # of conditions)
      - Each chunk might be represented
        - sequence #1 has chunks 2, 3, and shares chunk with 5
    - `rsa_features2ModelRDMs()`
  - Load in data in RDMs
  - Calculate variance-covariance of distances
    - given from `rsa_LDCsigma`; just save Sig into data structure
  - Use variance-covariance in estimation of $\omega$ estimates
  - Repeat update of $\omega$ through convergence

Practical: Anatomically-guided (surface-based) searchlight
------------------------
- Where in the brain is something represented?
  - for each voxel in the brain, and a sphere of voxels around it, and calculate RDM
- Issues
  - If interested in one region, sphere around center voxel might go into other areas of non-interest
- Solutions
  - If cortical areas of regions, you could use anatomically guided surface searchlight (surface-based)
  - Could draw a mask for anatomical ROI and constrain searchlight to ROI
  - Have spheres that cut off at boundaries (e.g., just within cerebellum)
    - could specify a minimum number of voxels to include
    - If want to look at cortex and other areas, can force it so that only one region can claim a voxel
- Example:
  - Location of representations of skilled finger movements
- Generate searchlight indices
  - `rsa_defineSearchlight({M},[],'sphere',[40 160]);`
    - M = functional mask, [] means just the functional mask, look everywhere
    - at center node, give all voxels within 40 mm, but at least 160 voxels
    - to get fixed radius, just [40]
  - Returns: structure with fields LI, voxels, voxmin, etc.
    - every row is one searchlight (linear indices to center voxels of searchlight)
    - 80613 total in tutorial
    - L.LI(1) --> 160 voxels
    - L.structure --> all 1s, since only looking at 1 region
- Run searchlight
  - Define contrast vector for RDM:
    - `D = load(fullfile(Opt.rootPath,Opt.spmDir,'SPM_info.mat'));`
    - `Opt.conditionVec = (D.hand == 1 & D.stimType==0) .* D.digit;`
    - Which trials are trials that below to right hand, active movements, code by digit (5 conditions)
  - Actually run the searchlight:
    - `rsa_runSearchlightLDC(L,Opt)`
    - Returns:
      - EPI size x 10 (5 x 5 RDM (10 unique distances))
  - Take mean of distances at each voxel (collapse over all 10 values, so just mean distance)
- Generate searchlight indices from multiple regions (functional and anat)
  - `L = rsa_defineSearchlight(M(2:3),M{1},'sphere',[40 160]);`
  - Smaller # of indices, just for 2 masks, within functional mask
- Surface-based searchlight
  - define searchlight on the surface (e.g., lh.pial, lh.white)
  - Gifti is like nifti, but for surface
  - Convert FS files to .gii
  - `rsa_readSurf({white},{pial});`
  - `rsa_defineSearchlight(S,mask,'sphere',[20 80]);`
