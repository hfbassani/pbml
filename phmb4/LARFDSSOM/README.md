# Original LARFDSSOM

### Requirements:

1. You must have a file containing all the paths to the datasets you want to use. You can follow [this](https://github.com/hfbassani/pbml/blob/master/phmb4/Parameters/inputPathsReal) example

2. You must have a parameters file:
   
   To run LARFDSSOM, we are using ten variables (parameters to the model and other stuff like seed for the rand calls):
   
   1. a_t
   2. lp
   3. dsbeta
   4. age_wins
   5. e_b
   6. e_n
   7. epsilon_ds
   8. minwd
   9. epochs
   10. seed
   
   So that you can follow [this](https://github.com/hfbassani/pbml/blob/master/phmb4/Parameters/OrigRealSeed_0) example, where the first ten lines represent the first set of parameters, lines 11 to 20 the second set and continues this way until the last set of parameters.
    
3. You must create a results folder

### Running:

1. Make sure you fully fill the requirements.
2. Open the NetbeansProject with Netbeans
3. Set the arguments for the program:

  ```
  -i: this flag is used to get the path to the file containing all the paths to the datasets to be used.
  
  -r: this flag is used to get the path to the results folder
  
  -p: this flag is used to get the path to the parameters file
  
  -s[optional]: this flag disables the subspace clustering mode. With this flag, each sample will be assigned to a single cluster.

  -f[optional]: this flag disables the noisy filtering and all samples will be assigned to a cluster.
  ```
  
  For example, to run experiments for the [these](https://github.com/hfbassani/pbml/tree/master/Datasets/Realdata) real datasets, you must use -s and -f flags, so that the arguments will be like follows:
  
  ```
  -i ../../Parameters/inputPathsReal -r teste_orig/ -p ../../Parameters/OrigRealSeed_0 -s
  ```
  
  After that, you can run yours metrics based on the results file.
  
### Parameters Generation 
  
See [params-gen](https://github.com/hfbassani/pbml/tree/master/params-gen/)
    
### Metrics Calculation

See [clustering-analysis](https://github.com/hfbassani/pbml/tree/master/clustering-analysis/)
