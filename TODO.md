
# ToDo List

### Fix
  -  Identify why Convolution Backpropagation is/isn't working correctly
  -  Make elements of `const Tensor` actually behave as `const`
  -  Standardise behaviour for `Tensor.dimensions`
  
-----------------------------------------------------------------------

### Next on the Agenda
 -  Implement Batch Normalisation
 -  Add depth to `RecurrentLayer`
 -  Mark members of big classes / structs as private / public
 -  Add `Tensor` support to `Regression`

-----------------------------------------------------------------------

### Investigate
 -  Is it possible to rewrite `Iterate` using a variadic template instead of run-time recursion
 -  Why is `RMSProp` producing such a weird pattern of losses
 -  Why is the model not overfitting
 -  Check training works on real data sets
 -  Find decent default values for hyperparameters like learning rate etc

-----------------------------------------------------------------------

### Test
 -  Use benchmarking to find bottlenecks
 -  Test efficiency of `Tensor`
 -  Check training works on real data sets

-----------------------------------------------------------------------

### Implement
 -  Batch Normalisation
 -  Support Vector Machine 
 -  Activation checkpointing
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
 -  Mark variables as `const` unless necessarily variable
 -  Improve python graphing - perform regression on loss during training to identify learning patterns

-----------------------------------------------------------------------

### Change

-----------------------------------------------------------------------

### Research
 -  Find decent default values for hyperparameters like learning rate etc
 -  Research potential parallelisations: CUDA or openCL? or just CPU threads
 -  Research regularisation of biases
 -  Find way of representing layers as a graph of nodes, with forwards/backwards propagation defined as edges between nodes

-----------------------------------------------------------------------

### Refactor

-----------------------------------------------------------------------
