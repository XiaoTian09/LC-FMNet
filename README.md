# LC-FMNet
A location-constrainted FCN model for mechanism determination

Introduction： A new focal mechanism determination method based on a fully convolutional neural network is proposed for accurately and efficiently inverting source mechanisms. Our approach incorporates aligned P-wave data, azimuth, and take-off angle as inputs, with Gaussian distributions of strike, dip, and rake as training labels. The model trained on the numerical datasets, is successfully applied to field data. 


'train.py' : Mechanism determination network based on FCN. The input is aligned P-wave data, azimuth, and take-off angle with the size of 1x64x128x3. The output size is 1x128x3x1.

'Traindata_demo.mat' is the training data of 2000 samples with the size of 2000x64x128x3.

'Trainlabel_demo.mat' is corresponding to the training label of 2000 samples with the size of 2000x128x3.

'Plot_kagan.m and mapping.m' is a script to draw the prediction.


The traning data demo (2000 samples) are accessible on the Jianguoyun via https://www.jianguoyun.com/p/DckcF9IQov3zBxjqx84FIAA

Any questions, please contact: tianx@ecut.edu.cn
