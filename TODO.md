
# ToDo List

### Fix
  -  Identify why Convolution Backpropagation is/isn't working correctly
  -  Make elements of `const Tensor` actually behave as `const`
  -  Change gradient descent algorithms in Network so that they don't return a pointer for costs
  -  Standardise behaviour for `Tensor.dimensions`
  
-----------------------------------------------------------------------

### Next on the Agenda
 -  Add depth to `RecurrentLayer`
 -  Combine `Layer`, `RecurrentLayer` and `ConvolutionLayer` into network object
 -  Mark members of big classes / structs as private / public
 -  Add `Tensor` support to `Regression`
 -  Add move constructor to `Tensor`, check large data not being copied unnecessarily eg pass into functions by reference

-----------------------------------------------------------------------

### Investigate
 -  Is it possible to rewrite `Iterate` using a variadic template instead of run-time recursion
 -  Why is `RMSProp` producing such a weird pattern of losses
 -  Why is the model not overfitting
 -  Check training works on real data sets
 -  Find decent default values for hyperparameters like learning rate etc

-----------------------------------------------------------------------

### Test
 -  Use benchmarking to find bottlenecks
 -  Test efficiency of `Tensor`
 -  Check training works on real data sets

-----------------------------------------------------------------------

### Implement
 -  Support Vector Machine
 -  Memory allocation tracker
 -  Data preprocessing pipeline
 -  Network Visualisation module
 -  Long Short-Term Memory Network
 -  Transformers
 -  Diffusion
 -  Folded-In-Time Network architecture
 -  Word Embedding Algorithm
 -  Jagged Tensors

-----------------------------------------------------------------------

### Improve
 -  Add overview analysis to benchmarking
 -  Mark members of big classes / structs as `private` / `public`
 -  Add move constructor to `Tensor`, check large data not being copied unnecessarily eg pass into functions by reference
 -  Mark variables as `const` unless necessarily variable
 -  Improve python graphing - perform regression on loss during training to identify learning patterns

-----------------------------------------------------------------------

### Change
 -  Switch to using `Tensor` for connected layer weights, biases etc
 -  Find way of representing layers as a graph of nodes, with forwards/backwards propagation defined as edges between nodes

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
