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
import weka.core.Instances;

public class ExtremeFeatureSelection extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator{
	
	/** The number of iterations **/
	  protected int m_numIterations = 1;

	  /** The promotion coefficient **/
	  protected double m_Alpha = 2.0;

	  /** The demotion coefficient **/
	  protected double m_Beta = 0.5;

	  /** Prediction threshold, <0 == numAttributes **/
	  protected double m_Threshold = -1.0;
	  
	  /** Random seed used for shuffling the dataset, -1 == disable **/
	  protected int m_Seed = 1;

	  /** Accumulated mistake count (for statistics) **/
	  protected int m_Mistakes;

	  /** Starting weights for the prediction vector(s) **/
	  protected double m_defaultWeight = 2.0;
	  
	  /** The weight vectors for prediction **/
	  private double[] m_predPosVector = null;
	  private double[] m_predNegVector = null;

	  /** The true threshold used for prediction **/
	  private double m_actualThreshold;
	  
	  private boolean updated = false;

	public void updateEvaluator(Instance inst) throws Exception {
		
		//Augmentation step
		double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
		List<Attribute> attList = new ArrayList<Attribute>();
		
		for(int i=0; i<inst.numAttributes()-1;i++) {
			Attribute atributo = new Attribute(Double.toString(rawx[i]));
			attList.add(atributo);
		}
		InstanceInformation instanceInformation  = new InstanceInformation("Test", attList);
		instanceInformation.setClassIndex(inst.numAttributes());
		Attribute attribute = new Attribute("1");
		instanceInformation.insertAttributeAt(attribute, inst.numAttributes()-2);
		
		


	}

	public void applySelection() {
		updated = false;
		
	}

	public boolean isUpdated() {
		return updated;
	}

	public double evaluateAttribute(int attribute) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
