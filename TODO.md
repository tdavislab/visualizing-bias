# TODOs

### Bug fixes
- [x] Change debiasing from `inplace` to `copy`
- [x] PCA on A + B only, then apply the projection matrix on the third.

### Different debiasing techniques
- [x] Linear Debiasing
- [ ] OSCaR - Orthogonal Subspace Correction and Rectification

### UI changes
- [x] Dim other points on hover
- [x] Align axis ranges for PCA and Two-means
- [x] Remove PCA picture
- [x] Toggle mean-line and purple points
- [ ] Remove points without removing the visualization of that word
- [ ] 3D PCA, future stuff
- [ ] Make the mean direction always horizontal and mean 1 on right and mean 2 on left


### Animation Tasks
- [x] Animate collapse to two-means
- [ ] Animate PCA as rotation and scaling (but from what to what?)
- [ ] Mean point in debiasing (right)
- [ ] Animate between one debiasing method to another to show difference between methods

### Miscellaneous
- Assumption on gender of name
- Projection on normal to max margin plane
- PCA with maximal alignment
- Check Viz for debiasing for what types of techniques they've used  

----

- Google slide deck shared among everyone 
- Protected class/seed sets (bias direction) may have more than 2 inputs
- Subspace method selection on the seedset line
- Saved screenshot of Vivek's interface 
- Oscar takes two subspaces and orthogonalizes it, so seed-set pair 1 and seed-set pair 2.
- Classification - linear classifier, take normal to separating hyperplane as gender direction.
  
- Linear projection with 4 subspace methods:
    1. Two-means (two seed inputs)
    2. PCA (one input, csv)
    3. PCA-paired (hypen + csv)
    4. Classification (two seed inputs)
- Evaluation set

- Hard debiasing, again with 4 subspace methods
    - Evaluation set
    - Equalize set, similar to PCA-paired
    - Hidden large neutral set - just a label that it is there
    - [Optional] Generating set
    
- INLP
    1. Classification set, two boxes
    2. Evaluation set
    3. Slider for number of iterations, between 1-40, default is 35
    4. Show which step you are at in the SVG
    
- OSCaR
    1. Space 1 - same 4 subspace methods
    2. Space 2 - same 4 subspace methods
    3. Evaluation set
    
- QOL: Linked zoom and pan

- Take dot product with subspace direction, that will be x coordinate
- X - gender direction = X', and then PCA of X', which will have orthogonal components to gender direction
- Color theme accessible to color-blind people, plus shape encoding, triangle or square
- Hover over point, click to retain highlighting across all views
- Classifier method - sample grid, check sign of classifier, and color accordingly
- WEAT score and residual bias measures

- Submit the slide deck