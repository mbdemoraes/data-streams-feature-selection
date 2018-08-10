package moa.featureselection.algorithms;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.AttributesInformation;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceInformation;

import moa.featureselection.common.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.AlgVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import java.util.Collections;


public class ExtremeFeatureSelection extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator{

	/** The number of iterations **/
	protected int m_numIterations = 1;

	/** The promotion coefficient **/
	protected double m_Alpha = 1.5;

	/** The demotion coefficient **/
	protected double m_Beta = 0.5;

	/** Prediction threshold, <0 == numAttributes **/
	protected double m_Threshold = 1.0;

	protected double marginM = 1.0;

	/** Random seed used for shuffling the dataset, -1 == disable **/
	protected int m_Seed = 1;

	/** Accumulated mistake count (for statistics) **/
	protected int m_Mistakes = 0;

	/** Starting weights for the prediction vector(s) **/
	protected double m_defaultWeight = 2.0;

	protected static double maxValue = 0.0;
	protected static double minValue = 0.0;

	protected double score = 0.0;
	protected AlgVector rankedAttributes = null;

	/** The weight vectors for prediction **/
	private AlgVector m_predPosVector = null;
	private AlgVector m_predNegVector = null;

	private double[] uVector = null;
	private double[] vVector = null;


	/** The true threshold used for prediction **/
	private double m_actualThreshold;

	private boolean updated = false;
	
	/**
	   * Returns a string describing this attribute evaluator
	   * 
	   * @return a description of the evaluator suitable for displaying in the
	   *         explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return "Extreme Feature Selection with Modified Balanced Winnow";
	  }
	  
	  /**
	   * Describe the attribute evaluator
	   * 
	   * @return a description of the attribute evaluator as a string
	   */
	  @Override
	  public String toString() {
	    StringBuffer text = new StringBuffer();

	    if (rankedAttributes == null) {
	      text.append("Ranked attributes has not been initialized");
	    } else {
	      text.append("\n Extreme Feature Selection with Modified Balanced Winnow");
	     
	    }

	    text.append("\n");
	    return text.toString();
	  }

	public static void getMaxMinValue(double[] numbers){
		maxValue = numbers[0];
		minValue = numbers[0];
		for(int i=1;i < numbers.length;i++){
			if(numbers[i] > maxValue){
				maxValue = numbers[i];
			} else if(numbers[i] < minValue){
				minValue = numbers[i];
			}

		}

	}



	public void updateModels(Instance inst, double trueClass) throws Exception {

		m_Mistakes++;

		int n1 = inst.numValues(); 
		for(int l = 0 ; l < n1; l++) {
			if(trueClass > 0) { 
				m_predPosVector.setElement(inst.index(l),  (m_predPosVector.getElement(inst.index(l))*m_Alpha*(1+inst.value(l))));		  
				m_predNegVector.setElement(inst.index(l),  (m_predNegVector.getElement(inst.index(l))*m_Beta*(1-inst.value(l))));  
			} else {
				m_predPosVector.setElement(inst.index(l),  (m_predPosVector.getElement(inst.index(l))*m_Beta*(1+inst.value(l))));		  
				m_predNegVector.setElement(inst.index(l),  (m_predNegVector.getElement(inst.index(l))*m_Alpha*(1-inst.value(l))));  
			}
			rankedAttributes.setElement(l, Math.abs(m_predPosVector.getElement(l) - m_predNegVector.getElement(l)));
		}
	}


	public void updateEvaluator(Instance inst) throws Exception {

		if(m_predPosVector == null && m_predNegVector == null) {
			m_predPosVector = new AlgVector(new double[inst.numAttributes()]);
			m_predNegVector = new AlgVector(new double[inst.numAttributes()]);
			rankedAttributes = new AlgVector(new double[inst.numAttributes()]);
			for(int i = 0; i < (inst.numAttributes()); i++) {
				m_predPosVector.setElement(i, m_defaultWeight); 
				m_predNegVector.setElement(i, m_defaultWeight); 
				rankedAttributes.setElement(i, 0);
			}
		}
		
		

		//Augmentation and normalization step
		double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
		double[] normalizedData = new double[rawx.length +1];
		getMaxMinValue(rawx);

		for(int j=0; j < normalizedData.length; j++) {
			if(j == normalizedData.length -1) {
				normalizedData[j] = 1.0;
			} else {
				normalizedData[j] = (rawx[j]-minValue)/(maxValue-minValue);
			}
			
		}

		/*List<Attribute> attList = new ArrayList<Attribute>();

		for(int i=0; i<inst.numAttributes()-1;i++) {
			Attribute atributo = new Attribute(Double.toString(normalizedData[i]));
			attList.add(atributo);
		}
		InstanceInformation instanceInformation  = new InstanceInformation("Test", attList);
		instanceInformation.setClassIndex(inst.numAttributes());
		Attribute attribute = new Attribute("1");
		instanceInformation.insertAttributeAt(attribute, inst.numAttributes()-2);
*/
		/** Set actual prediction threshold **/
		if(m_Threshold<0) {
			m_actualThreshold = (double)inst.numAttributes()-1;
		} else {
			m_actualThreshold = m_Threshold;
		}

		AlgVector x = new AlgVector(normalizedData);
		score = (m_predPosVector.dotMultiply(x)) - (m_predNegVector.dotMultiply(x)) - m_actualThreshold; 
		double classValue = inst.classValue();
		
		if(score * classValue <= marginM){
			updateModels(inst, inst.classValue());
		}
		
		updated = true;


	}

	public void applySelection() {
		updated = false;

	}

	public boolean isUpdated() {
		return updated;
	}

	public double evaluateAttribute(int attribute) throws Exception {
		return rankedAttributes.getElement(attribute);
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub

	}

}
