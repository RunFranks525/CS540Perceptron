// This code was written by a fellow student in CS 540. Please don't distribute or share this code with anyone.
// You may use it to complete assignment 2.

import java.util.*;
import java.io.*;
import java.text.DecimalFormat;

public class Perceptron
{
  // "Main" reads in the names of the files we want to use, then reads 
  // in their examples.
  //
  public static int FeatureCount;
  
  public static ArrayList<Feature> FeatureList;
  
  public static double LearningRate;
  
  public static int numExamples;
  
  public static String positiveValue;
  
  public static String negativeValue;
  
  public static String firstOutput;
  
  public static String secondOutput;
  
  public static double threshold;
  
  public static double maxTune;
  
  public static double testAtTune;
  
  public static int epoch;
  
  public static void main(String[] args)
  {   
    if (args.length != 3)
    {
      System.err.println("You must call BuildAndTestDecisionTree as " + 
			 "follows:\n\njava BuildAndTestDecisionTree " + 
			 "<trainsetFilename> <testsetFilename>\n");
      System.exit(1);
    }    

    // Read in the file names.
    String trainset = args[0];
    String tuneset = args[1];
    String testset  = args[2];

    // Read in the examples from the files.
    ArrayList<Example> trainExamples = new ArrayList<>();
    ArrayList<Example> tuneExamples = new ArrayList<>();
    ArrayList<Example> testExamples = new ArrayList<>();

    try {
		trainExamples = readExamplesFromFile(trainset);
		tuneExamples = readExamplesFromFile(tuneset);
		testExamples = readExamplesFromFile(testset);
	} catch (Exception e) {
		System.exit(0);
	}
    LearningRate = 0.1;
    ArrayList<Input> perceptronInputs = setUpPerceptronInputs();
    threshold = 0.0;
    maxTune = Double.MIN_VALUE;
    testAtTune = 0.0;
    buildPerceptron(trainExamples, perceptronInputs);
    for(int i = 1; i <= 1000; i++){
        ArrayList<Example> trainPermutation = createPermutation(trainExamples);
        if((i % 50) == 0) {
        	//System.out.print("Epoch " + i + ": ");
            buildPerceptron(trainPermutation, perceptronInputs);
            generateReporting(i, trainPermutation, tuneExamples, testExamples, perceptronInputs);
            System.out.println();
        } else {
            buildPerceptron(trainPermutation, perceptronInputs);
        }
    }
    handlePrinting(perceptronInputs);
  }
  
  /**
   * Helper function to handle printing out results from running the perceptron
   * @param perceptronInputs
   */
  private static void handlePrinting(ArrayList<Input> perceptronInputs) {
	DecimalFormat df = new DecimalFormat("#.##");
    System.out.println("------------");
    System.out.println("The tune set was highest (" + df.format(maxTune * 100) + "% accuracy) at Epoch " + epoch + ".  Test set = " + df.format(testAtTune * 100) + "% here.");
    System.out.println("------------");
    for(Input input : perceptronInputs){
    	System.out.println("Wgt = " + input.FeatureWeight + " " + input.Feature.FeatureName);
    }
    System.out.println("----");
    System.out.println("Threshold: " + threshold);
}

  /**
   * Helper function to handle printing out results of each Epoch over running the perceptron
   * @param epochNo
   * @param train
   * @param tune
   * @param test
   * @param perceptronInputs
   */
  private static void generateReporting(int epochNo, ArrayList<Example> train, ArrayList<Example> tune, ArrayList<Example> test, ArrayList<Input> perceptronInputs) {
	usePerceptron("train", train, perceptronInputs);
	double tuneAccuracy = usePerceptron("tune", tune, perceptronInputs);
	double testAccuracy = usePerceptron("test", test, perceptronInputs);
	if(tuneAccuracy > maxTune) {
		maxTune = tuneAccuracy;
		testAtTune = testAccuracy;
		epoch = epochNo;
	}
}

  /**
   * Helper function that returns a permutated example set from a provided example set
   * @param trainExamples
   * @return
   */
  private static ArrayList<Example> createPermutation(ArrayList<Example> trainExamples) {
	ArrayList<Example> trainPermutation = new ArrayList<Example>();
	ArrayList<Example> tempList = new ArrayList<Example>();

	for(Example ex : trainExamples) {
		tempList.add(ex);
	}
	while (tempList.size() > 0){
		Random random = new Random();
		int randomIndex = random.nextInt(tempList.size());
		trainPermutation.add(tempList.get(randomIndex));
		tempList.remove(randomIndex);
	}
	return trainPermutation;
}
  
  /**
   * Uses the trained Perceptron after the Perceptron is learned over a train example set
   * @param type
   * @param examples
   * @param perceptronInputs
   * @return
   */
  private static double usePerceptron(String type, ArrayList<Example> examples, ArrayList<Input> perceptronInputs){
	  double ratio = 0.0;
	  int numCorrect = 0;
	  for(Example example : examples){
		//Classify Examples
          double weightedSum = computeWeightedSum(example, perceptronInputs);
          int hypothesis = getHypothesis(weightedSum);
          if(hypothesis == example.OutputCategory){
        	  numCorrect++;
          } 
	  }
      ratio = (double) numCorrect / examples.size();
      DecimalFormat df = new DecimalFormat("#.##");
	  System.out.print(type + " = " + (df.format(ratio * 100)) + "% ");
      //System.out.print(df.format(ratio * 100) + " ");
	  return ratio;
  }

  /**
   * Builds the perceptron, training it on a provided example set. Runs the perceptron over every example in the example set and 
   * utilizes a number of helper functions to calculate and adjust different weights and thresholds
   * @param trainExamples
   * @param perceptronInputs
   */
  private static void buildPerceptron(ArrayList<Example> trainExamples, ArrayList<Input> perceptronInputs) {
	  for (int i = 0; i < numExamples; i++)
      {
          Example currentExample = trainExamples.get(i);
          int actualCategory = currentExample.OutputCategory;
          Boolean isCorrectOutputCategory = false;
          do {
        	  double weightedSum = computeWeightedSum(currentExample, perceptronInputs);
              int hypothesis = getHypothesis(weightedSum);
              isCorrectOutputCategory = evaluatePerceptronHypothesis(actualCategory, hypothesis);
              if (!isCorrectOutputCategory)
              {
                  for (int j = 0; j < FeatureCount; j++)
                  {
                      int currentFeatureValue = currentExample.ListOfFeatures.get(j).value;
                      Input currentInputAdjusment = perceptronInputs.get(j);
                      adjustWeight(currentInputAdjusment, actualCategory, hypothesis, currentFeatureValue);
                  }
                  threshold = adjustThreshold(perceptronInputs.get(perceptronInputs.size() - 1), actualCategory, hypothesis, -1);
              }
        } while (!isCorrectOutputCategory);
      }
  }
	  
  /**
   * Helper function used to adjust the bias threshold in the perceptron given that it needs adjusting (predicted output is wrong)
   * @param input
   * @param actualCategory
   * @param hypothesis
   * @param exFeatureValue
   * @return
   */
  private static double adjustThreshold(Input input, int actualCategory, int hypothesis, int exFeatureValue) {
	  double currentWeight = input.FeatureWeight;
      double weight = currentWeight + (LearningRate * (actualCategory - hypothesis) * exFeatureValue);
      input.FeatureWeight = weight;
      return weight;
  }

  /**
   * Helper function used to adjust weights on a given input node, utilizing a basic weight adjustment formula
   * @param input
   * @param actualCategory
   * @param hypothesis
   * @param exFeatureValue
   */
  private static void adjustWeight(Input input, int actualCategory, int hypothesis, int exFeatureValue) {
	  double currentWeight = input.FeatureWeight;
      double weight = currentWeight + (LearningRate * (actualCategory - hypothesis) * exFeatureValue);
      input.FeatureWeight = weight; 
  }

  /**
   * Helper function used to compute the weighted sum activation function for the Perceptron Nodes output
   * @param currentExample
   * @param perceptronInputs
   * @return
   */
  private static double computeWeightedSum(Example currentExample, ArrayList<Input> perceptronInputs) {
	  double weightedSum = 0.0;
      for(int i = 0; i < FeatureCount; i++)
      {
          //for each feature, use feature value * weight
          Feature currentFeature = currentExample.ListOfFeatures.get(i);
          double featureWeightProduct = currentFeature.value * perceptronInputs.get(i).FeatureWeight;
          weightedSum += featureWeightProduct;
      }
      int biasValue = perceptronInputs.get(perceptronInputs.size() - 1).Feature.value;
      double biasWeight = perceptronInputs.get(perceptronInputs.size() - 1).FeatureWeight;
      weightedSum += biasValue * biasWeight;
      return weightedSum;
  }

  /**
   * Helper function to evaluation the perceptron's output hypothesis, returns if it is correct
   * @param actualCategory
   * @param hypothesis
   * @return
   */
  private static Boolean evaluatePerceptronHypothesis(int actualCategory, int hypothesis){
	  if (actualCategory == hypothesis) return true;
      return false;
  }
  
  /**
   * Helper function to actually return a prediction from the perceptron given the activation function value
   * @param weightedSum
   * @return
   */
  private static int getHypothesis(double weightedSum){
      int hypothesis = 0;
      if (weightedSum >= threshold)
      {
          hypothesis = 1;
      }
      else
      {
          hypothesis = 0;
      }

      return hypothesis;
  }

  /**
   * Builds the perceptron inputs over the provided features in an example set
   * @return
   */
  private static ArrayList<Input> setUpPerceptronInputs(){
	  ArrayList<Input> perceptronInputs = new ArrayList<Input>();
      for(int i = 0; i < FeatureCount; i++)
      {
          Input inputNode = new Input(FeatureList.get(i), 0);
          perceptronInputs.add(inputNode);
      }
      Feature biasFeature = new Feature("bias", -1);
      Input biasNode = new Input(biasFeature, 0);
      perceptronInputs.add(biasNode);
      return perceptronInputs;
  }
  
  /**
   * Reads in the examples from a file provided on the command line
   * @param trainset
   * @return
   * @throws Exception
   */
  private static ArrayList<Example> readExamplesFromFile(String exampleSet) throws Exception {

	    // Try creating a scanner to read the input file.
	    Scanner fileScanner = null;
	    try {
	      fileScanner = new Scanner(new File(exampleSet));
	    } catch(FileNotFoundException e) {
	    	throw new Exception();
	    }
	    
	    // If the file was successfully opened, read the file
	    return parse(fileScanner);
}

  /**
   * Begins parsing the examples provided from the scanner in the input file
   * @param fileScanner
   * @return
   */
  private static ArrayList<Example> parse(Scanner fileScanner) {
	  FeatureCount = Integer.parseInt(parseSingleToken(fileScanner));
  
	  parseFeatures(fileScanner);
  
	  firstOutput = parseSingleToken(fileScanner);
	  secondOutput = parseSingleToken(fileScanner);
	  numExamples = Integer.parseInt(parseSingleToken(fileScanner));
	  // Parse the expected number of examples.
	  ArrayList<Example> examples = new ArrayList<>();
	  for(int i = 0; i < numExamples; i++) {
		  String line = findSignificantLine(fileScanner);
		  Scanner lineScanner = new Scanner(line);

		  // Parse a new example from the file.
		  String name = lineScanner.next();
		  String output = lineScanner.next();
		  Integer binaryOutput = -1;
		  if(output.equals(firstOutput)){
			  binaryOutput = 0;
		  } else if (output.equals(secondOutput)){
			  binaryOutput = 1;
		  } else {
			  System.exit(1);
		  }
		  Example ex = new Example(new ArrayList<>(), name, binaryOutput);
    
		  ArrayList<Feature> Features = new ArrayList<>();
		  for(int j = 0; j < FeatureCount; j++) {
			  String featureName = "Feature" + j;
			  String featureValueString = lineScanner.next();
			  Integer featureValue = -1;
			  if(featureValueString.equals(positiveValue)){
				  featureValue = 0;
			  } else if (featureValueString.equals(negativeValue)){
				  featureValue = 1;
			  } else {
				  System.exit(0);
			  }
			  Feature feature = new Feature(featureName, featureValue.intValue());
			  ex.ListOfFeatures.add(feature);
		  }
		  // Add this example to the list.
		  examples.add(ex);
  }
  return examples;
}

  /**
   * Parses out the feature information provided from the given example set
   * @param fileScanner
   */
  private static void parseFeatures(Scanner fileScanner) {
	    // Initialize the array of features to fill.
	    FeatureList = new ArrayList<Feature>();

	    for(int i = 0; i < FeatureCount; i++) {
	      String line = findSignificantLine(fileScanner);

	      // Once we find a significant line, read the feature description
	      // from it.
	      Scanner lineScanner = new Scanner(line);
	      String name = lineScanner.next();
	      String dash = lineScanner.next();  // Skip the dash in the file.
	      positiveValue = lineScanner.next();
	      negativeValue = lineScanner.next();
	      Feature feature = new Feature(name);
	      FeatureList.add(feature);
	    }
}

  /**
   * Helper function to parse out a single token from the scanners current token input
   * @param fileScanner
   * @return
   */
  private static String parseSingleToken(Scanner fileScanner) {
  String line = findSignificantLine(fileScanner);

  // Once we find a significant line, parse the first token on the
  // line and return it.
  Scanner lineScanner = new Scanner(line);
  return lineScanner.next();
}

  /**
   * Helper function to find significant lines in Input file
   * @param fileScanner
   * @return
   */
  private static String findSignificantLine(Scanner fileScanner) {
  // Keep scanning lines until we find a significant one.
  while(fileScanner.hasNextLine()) {
    String line = fileScanner.nextLine().trim();
    if (isLineSignificant(line)) {
		return line;
    }
  }
  

  // If the file is in proper format, this should never happen.
  System.err.println("Unexpected problem in findSignificantLine.");

  return null;
}

  /**
   * Helper function to test if a line in the input file is significant
   * @param line
   * @return
   */
  private static boolean isLineSignificant(String line) {
  // Blank lines are not significant.
  if(line.length() == 0) {
    return false;
  }

  // Lines which have consecutive forward slashes as their first two
  // characters are comments and are not significant.
  if(line.length() > 2 && line.substring(0,2).equals("//")) {
    return false;
  }

  return true;
}

}

/**
 * Class representing a Perceptron Input node
 * @author RunFranks525
 *
 */
class Input {
	public Feature Feature;
	public double FeatureWeight;
	
	public Input(Feature Feature, double FeatureWeight) {
		this.Feature = Feature;
		this.FeatureWeight = FeatureWeight;
	}
	
	public void setFeatureWeight(double FeatureWeight){
		this.FeatureWeight = FeatureWeight;
	}
	
	public double getFeatureWeight(){
		return this.FeatureWeight;
	}
}

/**
 * Class representing an Example in an Example set
 * @author RunFranks525
 *
 */
class Example {
	 public List<Feature> ListOfFeatures;
	 
     public String Name;
     
     public int OutputCategory;
     
     public Example (List<Feature> ListOfFeatures, String Name, int OutputCategory){
    	 this.ListOfFeatures = ListOfFeatures;
    	 this.Name = Name;
    	 this.OutputCategory = OutputCategory;
     }
     
     public void setListOfFeatures(ArrayList<Feature> ListOfFeatures){
    	 this.ListOfFeatures = ListOfFeatures;
     }
}

/**
 * Class representing a Feature of an Example in an Example set
 * @author RunFranks525
 *
 */
class Feature{
	public String FeatureName;
    public int value;
    
    public Feature(String FeatureName){
    	this.FeatureName = FeatureName;
    }
    
    public Feature(String FeatureName, int value){
    	this.FeatureName = FeatureName;
    	this.value = value;
    }
    
    public void setValue(int value){
    	this.value = value;
    }
    
    public int getValue(){
    	return this.value;
    }
}





  