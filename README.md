# second_structure_model

1. Simple convolutional neural network: 

	A simple convolutional neural network with twelve hidden layers and one output layer was constructed.  The kernel size was 3x1 and the number of channel was 256 in hidden layers. The SAME padding was utilized for each layer. 

2. bidirectional LSTM neural networks:

	The network was constructed with a bidirectional LSTM layer, two hidden layers and an output layer. The number of units in LSTM was 256. 1024 units were in the first and 512 units were in the second hidden layer.

3. Convolutional neural network with Conditional Neural Fields layer \cite{wang2016protein}: 

	The difference between Deep Convolutional Neural Fields (DeepCNF) and this implementation was that we used the relu activation function instead of sigmoid for hidden layers and we trained the network with Adam method instead of L-BFGS. Additionally, Wang trained the network layer by layer, but we trained the whole network directly. L2 regularization was not used here.

4. inception-inside-inception(Deep3I) \cite{fang2018mufold}: 

	In our implementation no dropout layer was used and the input features were PSSM alone. In the original work physio-chemical properties of amino acids and HHBlits profiles were used as input features besides PSSM. 

5. double bidirectional LSTM neural networks \cite{heffernan2017capturing}:

	Similar to 4), in contrast to the original work, the network was constructed without dropout layers. Physio-chemical properties of amino acids and HHBlits profiles were not used as input features. 
	
6. Convolutional neural network with bidirectional LSTM layer:

	A network with five convolution layers, followed by one bidirectional LSTM layer and one output layer was constructed. The kernel size was 3x1 and the channel number was 256 in convolution layers. The number of units in LSTM was 256.

7. Context convolutional neural network:

	We constructed a context convolutional neural network (Contextnet). Concatenation and dilated convolution operations were utilized in the network. The detailed structure was described in the \textit{Methods} section (see also Figure 1.).
