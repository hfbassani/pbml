# params-gen

This project was developed with the object to sample parameters between a range according to LHS (Latin Hypercube Sampling) for the LARFDSSOM

## Running:

1. Open the NetbeansProject with Netbeans
2. Set the arguments for the program:

  Datasets: [RealData](https://github.com/hfbassani/pbml/tree/master/Datasets/Realdata), [Simulated](https://github.com/hfbassani/pbml/tree/master/Datasets/Simulated)

  ```
  -f: this flag is used to get the path OF the file that will be created with the parameters
  
  -n: this flag is used to define the number of files to be created with different parameters sets
  
  -r: this flag is used to define the number of different parameters sets for each file (according to -n)
  
  -o: this flag is used to make sure that you are creating parameters for the original version of LARFDSSON rather than any other on development approaches
  
  -s[optional]: this flag is used when you want to create parameters between ranges for the simulated data (real data is the default option)
  ```
  
  Example for RealData: 
  
  ```
  -f ../phmb4/Parameters/OrigRealSeed -n 1 -r 500 -o
  ```
  
## OBS:

 if you want to change the range values or whatever else, feel free to update MyParameters.h
 
 ### You can run the .jar file without the need to open the project. Just follow this [link](https://github.com/hfbassani/pbml/tree/master/Executables).
