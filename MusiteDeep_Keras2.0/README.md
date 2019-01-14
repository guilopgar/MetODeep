# MusiteDeep: a Deep-learning Framework for General and Kinase-specific Phosphorylation Site Prediction 

This folder contains a Keras2.0 and Tensorflow version of MusiteDeep, downloaded from https://github.com/duolinwang/MusiteDeep/tree/master/MusiteDeep_Keras2.0, on November 5th, 2018. Several modifications were made afterwards:

* The code of https://github.com/guilopgar/MetODeep/blob/master/MusiteDeep_Keras2.0/MusiteDeep/methods/multiCNN.py was refactored, following the template method design pattern, with the goal of implementing other deep models in future work.

* The files https://github.com/guilopgar/MetODeep/blob/master/MusiteDeep_Keras2.0/MusiteDeep/predict.py, https://github.com/guilopgar/MetODeep/blob/master/MusiteDeep_Keras2.0/MusiteDeep/train_general.py and https://github.com/guilopgar/MetODeep/blob/master/MusiteDeep_Keras2.0/MusiteDeep/train_kinase.py were modified in order to include more input arguments, such as the value of the learning-rate and transfer-leayer hyperparameters, and a string indicating which deep learning model is used.

* The input arguments of https://github.com/guilopgar/MetODeep/blob/master/MusiteDeep_Keras2.0/MusiteDeep/methods/Bootstrapping_allneg_continue_val.py were changed, including, for example, two new arguments to specify the deep model and the optimizer used during training, respectively.
