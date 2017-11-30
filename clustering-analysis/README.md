# Clustering Analysis

## Requirements:

1. You must have a file containing all the outputs from LARFDSSOM
2. You must have a directory containing all the datasets and their respective ground truth files ([.true files](https://github.com/hfbassani/pbml/tree/master/Datasets/Realdata))
3. You must create a results folder for the output (a .csv file with all the information about the metrics and results)

## Running:

```
Usage: 'Metrics (CE:F1Measure)' 'Datafiles directory' 'Results directory' 'Output directory' 'Output file name' [-t] [-e extension] [-n 'Parameter Names File'] [-p 'Parameters File'] -r 'number of experiments'

Example: "CE:Accuracy" ../Datasets/Realdata ../phmb4/LARFDSSOM/NetbeansProject/larfdssom_results output_metrics "metrics_larfdssom" -p ../phmb4/Parameters/OrigRealSeed_0 -n ../phmb4/Parameters/parametersNameOrig -r 500 -t
```

Where:

```
'Metrics (CE:F1Measure)' is the metrics that you want to calculate
'Datafiles directory': directory containing all the datasets and their respective ground truth files
'Results directory':  directory containing all the outputs from LARFDSSOM
'Output directory' 'Output file name'
[-t]: if you ran LARDSSOM with -s flag, you need to use this -t flag to handle the relevances
[-e extension]: if you want to explicitly set your files extension
[-n 'Parameter Names File']: path to the file containing all the LARFDSSOM's parameters names (like [this](https://github.com/hfbassani/pbml/blob/master/phmb4/Parameters/parametersNameOrig))
[-p 'Parameters File']: path to the file containing the parameters set that you used running LARFDSSOM
-r 'number of experiments': the number of experiments for each dataset (number of parameters set)

```

### You can run the .jar file without the need to open the project. Just follow this [link](https://github.com/hfbassani/pbml/tree/master/Executables).
