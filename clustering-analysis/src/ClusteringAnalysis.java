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
    static ArrayList<String> parametersName = new ArrayList<>();
    static ArrayList<String> parameters = new ArrayList<>();
    static int repeat = 500;

    static File inputsDirectory;
    static String inputFileNames[];
    static ArrayList<Cluster> trueClusters;

    static final String INPUT_MESSAGE = "Usage: 'Metrics (CE:F1Measure)' 'Datafiles directory' 'Results directory' " +
            "'Output directory' [-t] [-e extension] [-n 'Parameter Names File'] [-p 'Parameters File'] -r 'number of experiments'";

    public static void main(String[] args) {

        if (args.length < 4) {
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

    public static void runTests(String inputPath, String resultsPath) {
        inputsDirectory = checkDirectory(inputPath);
        if(inputsDirectory == null) {
            System.out.println("Directory not found: " + inputPath);
            return;

        }
        inputFileNames = inputsDirectory.list();
        Arrays.sort(inputFileNames, (Object o1, Object o2) -> ((String) o1).compareTo((String) o2));

        String dataFile;
        String trueClustersFile;

        List<String> outputLines = new ArrayList<>();

        String[] measureNames = measuresStr.split(":");
        ArrayList<ArrayList<Double>> values = new ArrayList<>();
        ArrayList<Double> bestValues = new ArrayList<>();
        ArrayList<Double> means = new ArrayList<>();
        ArrayList<Integer> indexes = new ArrayList<>();

        for (String filename : inputFileNames) {
            if (getFileExtension(filename).compareTo(extension) == 0) {
                dataFile = inputPath + filename;
                trueClustersFile = inputPath + removeExtension(filename) + ".true";

                System.out.println("dataFile: " + dataFile);
                System.out.println("trueClustersFile: " + trueClustersFile);

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

                for (int r = 0; r < repeat; r++) {

                    String resultsFilename = resultsPath + removeExtension(filename) + "_" + r + ".results";
                    ArrayList<Cluster> results = readResults(new File(resultsFilename), allTrue);

                    //Evaluate results
                    ArrayList<ClusterQualityMeasure> measure =
                            evaluateClusters(measuresStr, results, dataInstances, trueClusters);

                    measures.add(measure);
                }

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

                    OptionalDouble average = measureValues.stream().mapToDouble(a -> a).average();
                    double mean = average.isPresent() ? average.getAsDouble() : 0;
                    means.add(mean);

                    String output = filename + "\nBest " + measureNames[i] + ": " + best + "\nBest Parameter Set Index: " + bestParameterSetIndex + "\nMean: " + mean + "\n";
                    outputLines.add(output);
                    System.out.println(output);
                }

            }
        }

        createFullCSV(measureNames, values, bestValues, indexes, means);

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmm").format(Calendar.getInstance().getTime());
        saveFile(outputLines, outputDir + timeStamp + "_results.txt");
    }

    public static void createFullCSV(String[] measureNames, ArrayList<ArrayList<Double>> measures,
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

        // Adding final values first

        String line = "max_value";
        for (int i = 0; i < measureNames.length; ++i) {

            int j = 0;
            while(j < qtdInputFiles) {
                line += "," + Double.toString(bestValues.get(i + (j * measureNames.length)));
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

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmm").format(Calendar.getInstance().getTime());
        saveFile(outputLines, outputDir + timeStamp + "_results.csv");
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