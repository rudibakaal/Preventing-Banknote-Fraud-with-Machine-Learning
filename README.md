# Preventing-Banknote-Fraud-with-Machine-Learning

## Motivation
Neural network used to evaluate banknote authenticity based on data extracted from genuine and forged banknote-like images with the Wavelet Transform tool.

For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained[1].

All features were standardised via sklearn's StandardScaler class.

## Neural Network Topology and Results Summary

The binary-crossentropy loss function was leveraged along with the Adam optimizer for this classification problem.

![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

After 75 epochs, the training and validation set binary classifiers reach 99% and 93% accuracy, respectively, in the predicting banknote authenticity. 

![bank-note](https://user-images.githubusercontent.com/48378196/97141706-5c4bfa00-17b3-11eb-8b51-79c927c250f5.png)

## License
MIT

## References
[1]  Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)
