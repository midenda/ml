
# ToDo List

### Fix
  - Identify why Convolution Backpropagation is/isn't working correctly

-----------------------------------------------------------------------

### Next on the Agenda
 -  Combine layers and convolutional layers into network object
 -  Mark members of big classes / structs as private / public
 -  Add move constructor to tensor, check large data not being copied unnecessarily eg pass into functions by reference
 -  Find decent default values for hyperparameters like learning rate etc
 -  Test on actual data

-----------------------------------------------------------------------

### Investigate
 -  Why is `RMSProp` producing such a weird pattern of losses
 -  Why is the model not overfitting

-----------------------------------------------------------------------

### Implement
 -  RNN
 -  transformers
 -  Folded-In-Time Network architecture
 -  Data preprocessing pipeline
 -  Jagged Tensors
 -  Network Visualisation module

-----------------------------------------------------------------------

### Improve
 -  Improve python graphing
 -  Mark members of big classes / structs as private / public
 -  Add move constructor to `Tensor`, check large data not being copied unnecessarily eg pass into functions by reference
 -  Mark variables as `const` unless necessarily variable

-----------------------------------------------------------------------

### Change
 -  Switch to using `Tensor` for connected layer weights, biases etc

-----------------------------------------------------------------------

### Research
 -  Find decent default values for hyperparameters like learning rate etc
 -  Research potential parallelisations: CUDA or openCL? or just CPU threads
 -  Read regularisation chapter

-----------------------------------------------------------------------

### Refactor
 -  Combine layers and convolutional layers into network object (Singleton class?) with same backpropagation algorithms/structure etc for `FullyConnectedLayers`, pass final gradient from one stage to the next
     - Rename `Network` -> `FullyConnectedLayers`
     - New Network contains:   
         - `ConvolutionalLayers`
         - `FullyConnectedLayers`
         - `RecurrentLayers`
         - ...

-----------------------------------------------------------------------
