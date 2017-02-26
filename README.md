# FaceRecognition_2013
MATLAB code for face recognition in the frequency (DCT) domain.
The attached code is the implementation of the system proposed in DOI: 10.1109/NCVPRIPG.2013.6776153

The code implements a face recognition system using 
  - Discrete Cosine Transform as the Feature Extractor
  - Dual Objective Feature Selector (DOI: 10.1109/NCVPRIPG.2013.6776153) as the Feature Selector
  - A Scaled version (to exploit within class information) of the Euclidean classifier as the classifier.
  
  - FERET.m is the main file. It is configured to run on the attached FERET database.
  - ScaleNorm.m contains the function that performs scale normalization (pre-processing)
  - SelectorFunc.m contains the function that performs feature selection.
