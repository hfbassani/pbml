/**
 * Created by pedromagalhaes on 03/06/17.
 */

import i9.subspace.base.Cluster;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

import weka.clusterquality.ClusterQualityMeasure;
import weka.core.Instances;
import weka.subspaceClusterer.SubspaceClusterEvaluation;

public class ClusteringAnalysis {

    static boolean allTrue = false;
    static String extension = ".arff";
    static String measuresStr = "CE";
    static String outputDir = "";
    static String outputFileName = "";
    static ArrayList<String> parametersName = new ArrayList<>();
    static ArrayList<String> parameters = new ArrayList<>();
    static int repeat = 500;
    static boolean semi_sup = false;

    static File inputsDirectory;
    static String inputFileNames[];
    static ArrayList<Cluster> trueClusters;

    static final String INPUT_MESSAGE = "Usage: 'Metrics (CE:F1Measure)' 'Datafiles directory' 'Results directory' " +
            "'Output directory' [-t] [-e extension] [-n 'Parameter Names File'] [-p 'Parameters File'] -r 'number of experiments' -S 'Semi-Supervised";

    public static void main(String[] args) {

        if (args.length < 5) {
            System.out.println(INPUT_MESSAGE);
            return;
        }

        String inputDir = null, resultsDir = null;

        for (int i = 0 ; i<args.length ; i++) {
            if (i == 0) {
                measuresStr = args[i];
            } else if (i == 1) {
                inputDir = handleDirectoryString(args[i]);
            } else if (i == 2) {
                resultsDir = handleDirectoryString(args[i]);
                String[] dirSplit = resultsDir.split("/");
                outputFileName = dirSplit[dirSplit.length - 1];
            } else if (i == 3) {
                outputDir = handleDirectoryString(args[i]);
            } else if (args[i].compareTo("-t") == 0) {
                showErrorMessage(args, i);
                allTrue = true;
            } else if (args[i].compareTo("-n") == 0) {
                showErrorMessage(args, i);

                i++;
                parametersName.addAll(readFile(args[i]));

            } else if (args[i].compareTo("-p") == 0) {
                showErrorMessage(args, i);

                i++;
                parameters.addAll(readFile(args[i]));

            } else if (args[i].compareTo("-e") == 0) {
                showErrorMessage(args, i);

                i++;
                extension = args[i];
            } else if (args[i].compareTo("-r") == 0) {
                showErrorMessage(args, i);

                i++;
                repeat = Integer.valueOf(args[i]);
            } else if (args[i].compareTo("-S") == 0) {
                showErrorMessage(args, i);
                semi_sup = true;
            }
        }

        runTests(inputDir, resultsDir);
    }

    public static ArrayList<String> readFile (String path) {
        ArrayList<String> lines = new ArrayList<>();

        try (Stream<String> stream = Files.lines(Paths.get(path))) {
            stream.forEach(line -> lines.add(line));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return lines;
    }

    public static void showErrorMessage (String[] args, int i) {
        if (args.length < i) {
            System.out.println(INPUT_MESSAGE);
            return;
        }
    }

    public static String handleDirectoryString (String path) {
        String dir = path;
        if (!dir.endsWith("/"))
            dir = dir + "/";

        return dir;
    }

    public static File checkDirectory (String path) {
        File dir = new File(path);
        if (!dir.exists()) {
            return null;
        }

        return dir;
    }

    public static void createDirectory (String path) {
        Path folder_path = Paths.get(path);
        if (!Files.exists(folder_path)) {
            try {
                Files.createDirectory(folder_path);
                System.out.println("Directory created");
            } catch (IOException e) {
                System.out.println("Error creating directory: " + path);
            }
        }
    }

    public static void runTests(String inputPath, String resultsPath) {

        inputsDirectory = checkDirectory(inputPath);
        if(inputsDirectory == null) {
            System.out.println("Directory not found: " + inputPath);
            return;

        }

        createDirectory(outputDir);

        inputFileNames = inputsDirectory.list();
        Arrays.sort(inputFileNames, (Object o1, Object o2) -> ((String) o1).compareTo((String) o2));

        String dataFile;
        String trueClustersFile;

        List<String> outputLines = new ArrayList<>();

        String[] measureNames = measuresStr.split(":");
        ArrayList<Integer> clustersFound = new ArrayList<>();
        ArrayList<ArrayList<Double>> values = new ArrayList<>();
        ArrayList<Double> bestValues = new ArrayList<>();
        ArrayList<Double> means = new ArrayList<>();
        ArrayList<Integer> indexes = new ArrayList<>();

        for (String filename : inputFileNames) {
            if (getFileExtension(filename).compareTo(extension) == 0) {
                dataFile = inputPath + filename;
                trueClustersFile = inputPath + removeExtension(filename) + ".true";


                Instances dataInstances = null;
                try {
                    if (extension.compareTo(".arff") == 0) {
                        dataInstances = new Instances(new FileReader(dataFile));
                        // Make the last attribute be the class
                        dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
                    }

                } catch (IOException ex) {
                    Logger.getLogger(ClusteringAnalysis.class.getName()).log(Level.SEVERE, null, ex);
                    return;
                }

                trueClusters = readTrueClusters(new File(trueClustersFile));
                ArrayList<ArrayList<ClusterQualityMeasure>> measures = new ArrayList<>();
                ArrayList<Integer> clusters = new ArrayList<>();

                for (int r = 0; r < repeat; r++) {

                    String resultsFilename = resultsPath + removeExtension(filename) + "_" + r + ".results";
                    ArrayList<Cluster> results = readResults(new File(resultsFilename), allTrue);

                    //Evaluate results
                    ArrayList<ClusterQualityMeasure> measure =
                            evaluateClusters(measuresStr, results, dataInstances, trueClusters);

                    clusters.add(results.size());
                    measures.add(measure);
                }

                System.out.println("\n" + filename);

                for (int i = 0 ; i < measureNames.length ; ++i) {

                    ArrayList<Double> measureValues = new ArrayList<>();
                    for (ArrayList<ClusterQualityMeasure> measure : measures) {
                        measureValues.add(measure.get(i).getOverallValue());
                    }

                    values.add(measureValues);

                    double best = Collections.max(measureValues);
                    bestValues.add(best);
                    int bestParameterSetIndex = measureValues.indexOf(best);
                    indexes.add(bestParameterSetIndex);
                    clustersFound.add(clusters.get(bestParameterSetIndex));

                    OptionalDouble average = measureValues.stream().mapToDouble(a -> a).average();
                    double mean = average.isPresent() ? average.getAsDouble() : 0;
                    means.add(mean);

                    String output = "\nBest " + measureNames[i] + ": " + best + "\nBest Parameter Set Index: " + bestParameterSetIndex + "\nMean: " + mean + "\nClusters Found: " + clusters.get(bestParameterSetIndex);
                    outputLines.add(output);
                    System.out.println(output);
                }
            }
        }

        createFullCSV(measureNames, values, clustersFound, bestValues, indexes, means);
    }

    public static void createFullCSV(String[] measureNames, ArrayList<ArrayList<Double>> measures, ArrayList<Integer> clustersFound,
                                     ArrayList<Double> bestValues, ArrayList<Integer> indexes, ArrayList<Double> means) {
        ArrayList<String> outputLines = new ArrayList<>();

        // Headers
        String header_line = "experiment";
        int qtdInputFiles = 0;

        for (String measure : measureNames) {
            for (String fileName : inputFileNames) {
                if (getFileExtension(fileName).compareTo(extension) == 0) {
                    ++qtdInputFiles;
                    header_line += "," + removeExtension(fileName) + "_" + measure;
                }
            }
        }

        for (String att : parametersName) {
            header_line += "," + att;
        }

        qtdInputFiles /= measureNames.length;

        /**********************************************/

        List<Double> bestValuesSorted = new ArrayList<>();

        // Adding final values first
        String line = "max_value";
        for (int i = 0; i < measureNames.length; ++i) {

            int j = 0;
            while(j < qtdInputFiles) {
                double values = bestValues.get(i + (j * measureNames.length));
                bestValuesSorted.add(values);
                line += "," + Double.toString(values);
                ++j;
            }
        }
        outputLines.add(line);

        line = "clusters_found";
        for (int i = 0; i < measureNames.length; ++i) {

            int j = 0;
            while(j < qtdInputFiles) {
                line += "," + Integer.toString(clustersFound.get(i + (j * measureNames.length)));
                ++j;
            }
        }
        outputLines.add(line);

        line = "index_set";
        for (int i = 0; i < measureNames.length; ++i) {

            int j = 0;
            while(j < qtdInputFiles) {
                line += "," + Integer.toString(indexes.get(i + (j * measureNames.length)));
                ++j;
            }
        }
        outputLines.add(line);

        line = "mean_value";
        for (int i = 0; i < measureNames.length; ++i) {

            int j = 0;
            while(j < qtdInputFiles) {
                line += "," + Double.toString(means.get(i + (j * measureNames.length)));
                ++j;
            }
        }
        outputLines.add(line);

        int start = 0;
        int end = bestValues.size() / measureNames.length;
        for (int i = 0 ; i < measureNames.length ; ++i) {

            List<Double> measureValues = bestValuesSorted.subList(start, end);

            System.out.println("\n" + measureNames[i] + " mean(std):" + Double.toString(mean(measureValues)) + " (" + Double.toString(sample_stdev(measureValues)) + ")" );

            start = end;
            end += measureValues.size();
        }

        /**********************************************/

        // Adding header line

        outputLines.add("");
        outputLines.add(header_line);

        /**********************************************/


        for (int k = 0 ; k < repeat ; ++k) {

            line = Integer.toString(k);

            // Adding each measure values

            for (int i = 0; i < measureNames.length; ++i) {

                int j = 0;
                while(j < qtdInputFiles) {
                    line += "," + Double.toString(measures.get(i + (j * measureNames.length)).get(k));
                    ++j;
                }
            }

            // Adding each parameter value
            for (int i = 0 ; i < parametersName.size() ; ++i) {
                line += "," + parameters.get(i + (k * parametersName.size()));
            }

            outputLines.add(line);
        }

        saveFile(outputLines, outputDir + outputFileName + ".csv");
    }

    public static double mean (List<Double> values) {

        double mean = 0;

        for (int i =0 ; i < values.size() ; ++i) {
            mean += values.get(i);
        }

        return mean > 0 ? mean / values.size() : mean;
    }

    public static double sample_stdev (List<Double> values) {
        // Step 1:
        double mean = mean(values);
        double temp = 0;

        for (int i = 0; i < values.size(); i++)
        {
            // Step 2:
            double squrDiffToMean = Math.pow(values.get(i) - mean, 2);

            // Step 3:
            temp += squrDiffToMean;
        }

        // Step 4:
        double meanOfDiffs = temp / (values.size() - 1);

        // Step 5:
        return Math.sqrt(meanOfDiffs);
    }

    public static void saveFile(List<String> lines, String path) {
        Path file = Paths.get(path);
        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException ex) {
            Logger.getLogger(ClusteringAnalysis.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static String getFileExtension(String fileName) {
        int dotPos = fileName.lastIndexOf(".");
        if (dotPos < 0) {
            return "";
        }
        return fileName.substring(dotPos);
    }

    public static String removeExtension(String fileName) {
        int dotPos = fileName.lastIndexOf(".");
        return fileName.substring(0, dotPos);
    }

    public static ArrayList<Cluster> readTrueClusters(File file) {
        Scanner sc;
        try {
            sc = new Scanner(file);
            sc.useLocale(Locale.ENGLISH);
        } catch (IOException ex) {
            Logger.getLogger(ClusteringAnalysis.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
        ArrayList<Cluster> clusterList = new ArrayList<>();
        sc.skip("DIM=");
        sc.useDelimiter(";");
        int dim = sc.nextInt();
        sc.nextLine();

        sc.useDelimiter("\\s");
        while (sc.hasNext()) {
            boolean[] subspace = new boolean[dim];
            for (int j = 0; j < dim; j++) {
                int useDim = sc.nextInt();
                subspace[j] = useDim > 0.5;
            }

            int numInstances = sc.nextInt();
            ArrayList<Integer> objects = new ArrayList<>();

            for (int i = 0; i < numInstances; i++) {
                int value = sc.nextInt();
                objects.add(value);
            }

            Cluster cluster = new Cluster(subspace, objects);
            clusterList.add(cluster);
        }

        return clusterList;
    }

    public static ArrayList<Cluster> readResults(File file, boolean allTrue) {
        Scanner sc;
        try {
            sc = new Scanner(file);
            sc.useLocale(Locale.ENGLISH);
        } catch (IOException ex) {
            Logger.getLogger(ClusteringAnalysis.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }

        //Read clusters info
        int numClusters = sc.nextInt();
        int numAttributes = sc.nextInt();

        //Create clusters
        ArrayList<Cluster> clusterList = new ArrayList<>();
        for (int i=0; i< numClusters; i++) {
            sc.nextInt(); //skip cluster number
            double dsWs[] = new double[numAttributes];
            double average = 0;
            for (int j=0; j<numAttributes; j++) {
                dsWs[j] = sc.nextFloat();
                average+=dsWs[j];
            }
            average = average/numAttributes;


           /*
           double deviation = 0;
           for (int j=0; j<numAttributes; j++) {
               deviation += Math.abs(average-dsWs[j]);
           }
           deviation = deviation/numAttributes;
           average = average - deviation;
           /**/

            boolean[] subspace = new boolean[numAttributes];
            for (int j=0; j<numAttributes; j++) {
                if (allTrue) {
                    subspace[j] = true;
                } else {
                    subspace[j] = dsWs[j] >= average;
                }
            }

            Cluster cluster = new Cluster(subspace, new ArrayList<>());
            clusterList.add(cluster);
        }

        //Read clusters data
        while (sc.hasNext()) {
            int index = sc.nextInt();
            int cluster = sc.nextInt();

            if (semi_sup)
                sc.nextInt(); //skip class value

            if (cluster>=0)
                clusterList.get(cluster).m_objects.add(index);
        }
        /**/

        return clusterList;
    }

    public static ArrayList<ClusterQualityMeasure> evaluateClusters(String measuresStr, ArrayList<Cluster> clusterResults, Instances instances, ArrayList<Cluster> trueClusters) {
        ArrayList<ClusterQualityMeasure> eMeasures = SubspaceClusterEvaluation.getMeasuresByOptions(measuresStr);

        eMeasures.forEach((m) -> {
            m.calculateQuality(clusterResults, instances, trueClusters);
        });

        return eMeasures;
    }
}
