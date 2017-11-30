# Running executables

first of all, navigate to the directory containing the binaries.

## Creating Parameters File 

  1. Run the program:

  ```sh
  $ ./params-gen -f ../phmb4/Parameters/OrigRealSeed -n 1 -r 500 -o
  ```
  
  This example will create only one parameters file (called OrigRealSeed_0) containing different parameters sets for the original version of LARFDSSOM with ranges for real data [datasets](https://github.com/hfbassani/pbml/tree/master/Datasets/Realdata).

## Running LARFDSSOM

  1. Create a results directory for the output of the model
  2. Run the program:
  
  ```sh
  $ -i ../../Parameters/inputPathsReal -r results/ -p ../phmb4/Parameters/Parameters/OrigRealSeed_0 -s
  ```
  This example will run the LARFDSSOM using the parameters files create above, saving the outputs in the results directory that you create.

## Calculating your metrics

  1. Create a output_metrics directory for the output of the program
  2. Run the program:
  
  ```sh
  java -jar ClusteringAnalysis.jar "CE:Accuracy" ../Datasets/Realdata results output_metrics "metrics_larfdssom" -n ../phmb4/Parameters/originalParameters -r 500 -t
  ```
  
  This example will calculate the Clustering Error and Accuracy of LARFDSSOM's results, saving the output (a .csv file) in the output_metrics directory that you create.

  Opening the .csv file, you will see the main results at the first four lines for each Dataset used.
  
  
  ## OBS:
  
  All the details about the programs, parameters and usage can be found in the following links: [params-gen](https://github.com/hfbassani/pbml/tree/master/params-gen), [clustering-analysis](https://github.com/hfbassani/pbml/tree/master/clustering-analysis) and [LARFDSSOM](https://github.com/hfbassani/pbml/tree/master/phmb4/LARFDSSOM).
  
