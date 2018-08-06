package moa.featureselection.test;

import java.io.IOException;

import weka.classifiers.lazy.IBk;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;

import moa.classifiers.lazy.kNN;

import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.featureselection.classifiers.NaiveBayes;
import moa.streams.ArffFileStream;
import moa.streams.generators.RandomRBFGenerator;

public class Experiment {

        public Experiment(){
        }

        public void run(int numInstances, boolean isTesting){
        		//kNN knn = new kNN();
        		//knn.kOption.setValue(3);
        		//Classifier learner = knn;
        		Classifier learner = new NaiveBayes();
                
                //RandomRBFGenerator stream = new RandomRBFGenerator();
        		ArffFileStream stream = new ArffFileStream("/home/athos/airlines.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/real/covtypeNorm.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/real/poker-lsn.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/artificial/sudden_drift_med.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/artificial/incremental_slow_med.arff", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/TEST_FUSINTER/datasets/spambase/spambase-10-1tra-weka.dat", -1);
        		//ArffFileStream stream = new ArffFileStream("/home/sramirez/datasets/drift/artificial/gradual_drift_100k.arff", -1);
        		
        		//stream.numAttsOption.setValue(1000);
                stream.prepareForUse();

                learner.setModelContext(stream.getHeader());
                learner.prepareForUse();

                int numberSamplesCorrect = 0;
                int numberSamples = 0;
                long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                while (stream.hasMoreInstances() && numberSamples < numInstances) {
                        InstanceExample trainInst = stream.nextInstance();
                        if (isTesting) {
                                if (learner.correctlyClassifies(trainInst.getData())){
                                        numberSamplesCorrect++;
                                }
                        }
                        numberSamples++;
                        learner.trainOnInstance(trainInst);
                }
                double accuracy = 100.0 * (double) numberSamplesCorrect/ (double) numberSamples;
                double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
                System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");
        }

        public static void main(String[] args) throws IOException {
        		Experiment exp = new Experiment();
                exp.run(1000000, true);
        }
}