package moa.featureselection.algorithms;

import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.concurrent.ThreadLocalRandom;

import moa.featureselection.algorithms.OnlineFeatureSelection.Pair;
import moa.featureselection.common.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.AlgVector;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Capabilities.Capability;

public class OFSpartial extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private static final int R = 10; //according to authors criteria
	private static int numFeatures = 10; 
	private static long selectedFeatures = 0;
	private static int numAttributesInstance = 0;
	private static final double epsilon = 0.2; //according to authors criteria
	private static final double eta = 0.2; //according to authors criteria
	private boolean updated = false;
	

	  /** Treat missing values as a seperate value */
	  private boolean m_missing_merge;

	  /** Just binarize numeric attributes */
	  private boolean m_Binarize;
	
	private static AlgVector weights = null;
	
	/**
	   * Returns a string describing this attribute evaluator
	   * 
	   * @return a description of the evaluator suitable for displaying in the
	   *         explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return "OFSGDAttributeEval :\n\nEvaluates the worth of an attribute "
	      + "by measuring performing an stochastic gradient descent approach with feature truncation\n";
	  }

	  /**
	   * Constructor
	   */
	  public OFSpartial(int numFeatures) {
		this.numFeatures = numFeatures;
	    resetOptions();
	    
	  }	  
	  
	  /**
	   * Returns the tip text for this property
	   * 
	   * @return tip text for this property suitable for displaying in the
	   *         explorer/experimenter gui
	   */
	  public String binarizeNumericAttributesTipText() {
	    return "Just binarize numeric attributes instead of properly discretizing them.";
	  }

	  /**
	   * Binarize numeric attributes.
	   * 
	   * @param b true=binarize numeric attributes
	   */
	  public void setBinarizeNumericAttributes(boolean b) {
	    m_Binarize = b;
	  }

	  /**
	   * get whether numeric attributes are just being binarized.
	   * 
	   * @return true if missing values are being distributed.
	   */
	  public boolean getBinarizeNumericAttributes() {
	    return m_Binarize;
	  }

	  /**
	   * Returns the tip text for this property
	   * 
	   * @return tip text for this property suitable for displaying in the
	   *         explorer/experimenter gui
	   */
	  public String missingMergeTipText() {
	    return "Distribute counts for missing values. Counts are distributed "
	      + "across other values in proportion to their frequency. Otherwise, "
	      + "missing is treated as a separate value.";
	  }

	  /**
	   * distribute the counts for missing values across observed values
	   * 
	   * @param b true=distribute missing values.
	   */
	  public void setMissingMerge(boolean b) {
	    m_missing_merge = b;
	  }

	  /**
	   * get whether missing values are being distributed or not
	   * 
	   * @return true if missing values are being distributed.
	   */
	  public boolean getMissingMerge() {
	    return m_missing_merge;
	  }

	  /**
	   * Returns the capabilities of this evaluator.
	   * 
	   * @return the capabilities of this evaluator
	   * @see Capabilities
	   */
	  @Override
	  public Capabilities getCapabilities() {
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable(Capability.DATE_ATTRIBUTES);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    return result;
	  }

	public void updateEvaluator(Instance inst) throws Exception {
		// TODO Auto-generated method stub
		
		this.numAttributesInstance = inst.numAttributes()-1;
		this.selectedFeatures = Math.round( 0.1 * (inst.numAttributes()-1));
		AlgVector c_t = null;
		
		if(weights == null) {
	  		weights = new AlgVector(new double[inst.numAttributes() - 1]);
	  		c_t = new AlgVector(new double[inst.numAttributes() - 1]);
	  		for(int i = 0; i < weights.numElements(); i++) {
	  			weights.setElement(i, 0); 
	  			c_t.setElement(i,0);
	  		}
	  	}
		
		int z_t = getBernoulli(1,epsilon);
		
		double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
	  	
	  	
	  	if (z_t == 1) {
	  		c_t = getRandomAttributes(inst);
	  	} else {
	  		
	  		for(int i = 0; i < weights.numElements()-1; i++) {
	  			if(weights.getElement(i) != 0) {
	  				c_t.setElement(i, weights.getElement(i)); 
	  			}
	  			
	  		}
	  		
	  	}
	  	
	  	AlgVector x = c_t;
		double pred = weights.dotMultiply(x); 
		AlgVector x_t = getNewXt(x);
		
		if(pred * inst.classValue() <= 1){
			x_t.scalarMultiply(eta * inst.classValue());
	  		//weights = weights.add(x_t);
	  		weights.scalarMultiply(Math.min(1.0, R / weights.norm()));
	  		
			
			int counts = 0;
	  		Pair[] array = new Pair[weights.numElements()];
	  		for(int i = 0; i < weights.numElements(); i++){
	  			array[i] = new Pair(i, weights.getElement(i));
	  			if(weights.getElement(i) != 0) counts++;
	  		}
			
			// Truncate
	  		if(counts > numFeatures) {
	  			Arrays.sort(array);
	  			for(int i = numFeatures + 1; i < array.length; i++)
	  				weights.setElement(array[i].index, 0);
	  		}
	  	
	  	}
		
	  	updated = true;

		
	}
	
	private static AlgVector getRandomAttributes(Instance inst) {
		AlgVector c_t = null;
		double[] newVector = new double[numFeatures-1];
		int maxValue = inst.numAttributes()-1;
		int randomNum = 0;
		
		double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
		
		
		for(int i =0; i < numFeatures-1; i++) {
			//nextInt is normally exclusive of the top value,
			//so add 1 to make it inclusive
			randomNum = ThreadLocalRandom.current().nextInt(0, maxValue-1);
			newVector[i] = rawx[randomNum];
		}
		
		c_t = new AlgVector(newVector);
		
		return c_t;
	}
	
	private static AlgVector getNewXt(AlgVector x) {
		AlgVector xt = null;
		double[] calc = new double[numAttributesInstance-1];		
		 
		for(int i = 0; i < numAttributesInstance-1; i++) {
			if (weights.getElement(i)!=0) {
				calc[i] = x.getElement(i) / (numFeatures/numAttributesInstance * epsilon) +  (weights.getElement(i)* (1-epsilon));
			} else {
				calc[i] = x.getElement(i) / (numFeatures/numAttributesInstance * epsilon);
			}
			
			
		}
		
		xt = new AlgVector(calc);
		
		return xt;
		
	}
	
	private static int getBernoulli(int n, double p) {
		  int x = 0;
		  for(int i = 0; i < n; i++) {
		    if(Math.random() < p)
		      x++;
		  }
		  return x;
	}
	public void applySelection() {
		// TODO Auto-generated method stub
		//System.out.println("Weight values: " + Arrays.toString(weights.getElements()));		
	  	updated = false;
	}

  /**
   * Reset options to their default values
   */
  protected void resetOptions() {
    m_missing_merge = true;
    m_Binarize = false;
  }

  /**
   * evaluates an individual attribute by measuring the amount of information
   * gained about the class given the attribute.
   * 
   * @param attribute the index of the attribute to be evaluated
   * @return the info gain
   * @throws Exception if the attribute could not be evaluated
   */
  
  public double evaluateAttribute(int attribute) throws Exception {
    return weights.getElement(attribute);
  }

  /**
   * Describe the attribute evaluator
   * 
   * @return a description of the attribute evaluator as a string
   */
  @Override
  public String toString() {
    StringBuffer text = new StringBuffer();

    if (weights == null) {
      text.append("First weights has not been built");
    } else {
      text.append("\n OFS Partial Ranking Filter");
      if (!m_missing_merge) {
        text.append("\n\tMissing values treated as seperate");
      }
      if (m_Binarize) {
        text.append("\n\tNumeric attributes are just binarized");
      }
    }

    text.append("\n");
    return text.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10172 $");
  }
  
  class Pair implements Comparable<Pair> {
	    public final int index;
	    public final double value;

	    public Pair(int index, double value) {
	        this.index = index;
	        this.value = value;
	    }

	    public int compareTo(Pair other) {
	        //descending sort order
	        return -1 * Double.valueOf(Math.abs(this.value)).compareTo(Math.abs(other.value));
	    }
	}
  
  @Override
	public void buildEvaluator(Instances arg0) throws Exception {
	// TODO Auto-generated method stub
	
	}

public boolean isUpdated() {
	// TODO Auto-generated method stub
	return updated;
}
	

}
