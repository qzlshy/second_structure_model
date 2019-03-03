# second_structure_model

1. sim_cnn(Simple convolutional neural network): 

	A simple convolutional neural network with twelve hidden layers and one output layer was constructed.  The kernel size was 3x1 and the number of channel was 256 in hidden layers. The SAME padding was utilized for each layer. 

2. bilstm(bidirectional LSTM neural networks):

	The network was constructed with a bidirectional LSTM layer, two hidden layers and an output layer. The number of units in LSTM was 256. 1024 units were in the first and 512 units were in the second hidden layer.

3. dcnf(Convolutional neural network with Conditional Neural Fields layer): 

	The difference between Deep Convolutional Neural Fields (DeepCNF) and this implementation was that we used the relu activation function instead of sigmoid for hidden layers and we trained the network with Adam method instead of L-BFGS. Additionally, Wang trained the network layer by layer, but we trained the whole network directly. L2 regularization was not used here. 
	reference：Protein secondary structure prediction using deep convolutional neural fields

4. inception(inception-inside-inception(Deep3I)): 

	In our implementation no dropout layer was used and the input features were PSSM alone. In the original work physio-chemical properties of amino acids and HHBlits profiles were used as input features besides PSSM.
	reference：MUFOLD-SS: New deep inception-inside-inception networks for protein secondary structure prediction

5. double_bilstm(double bidirectional LSTM neural networks):

	Similar to 4), in contrast to the original work, the network was constructed without dropout layers. Physio-chemical properties of amino acids and HHBlits profiles were not used as input features. 
	reference：Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility
	
6. cnn_lstm(Convolutional neural network with bidirectional LSTM layer):

	A network with five convolution layers, followed by one bidirectional LSTM layer and one output layer was constructed. The kernel size was 3x1 and the channel number was 256 in convolution layers. The number of units in LSTM was 256.

7. context(Context convolutional neural network):

	First 8 layer convolutions were concatenated together and the features of different receptive field were mixed. In 5-8 layers the dilated convolution were used, strides of the dilated convolutions in layers 5-8 were 2, 4, 8 and 16 respectively. The kernel size used were 3x1, the activation function was relu in hidden layers, and the activation function in the output layer was softmax. The loss function was cross entropy.
