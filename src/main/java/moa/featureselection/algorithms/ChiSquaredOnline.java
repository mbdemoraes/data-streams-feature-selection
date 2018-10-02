package moa.featureselection.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;


import moa.featureselection.common.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

import com.yahoo.labs.samoa.instances.Instance;

public class ChiSquaredOnline extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator{
	
	/** for serialization */
	  static final long serialVersionUID = -8316857822521717692L;

	  /** Treat missing values as a seperate value */
	  private boolean m_missing_merge;

	  /** Just binarize numeric attributes */
	  private boolean m_Binarize;

	  /** The chi-squared value for each attribute */
	  private double[] m_ChiSquareds;
	  
	  private boolean updated = false;
	  
	  private int classIndex;
	  
	  private HashMap<Key, Float>[] counts = null;
	  
	  /**
	   * Returns a string describing this attribute evaluator
	   * 
	   * @return a description of the evaluator suitable for displaying in the
	   *         explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return "ChiSquaredAttributeEval :\n\nEvaluates the worth of an attribute "
	      + "by computing the value of the chi-squared statistic with respect to the class.\n";
	  }

	  /**
	   * Constructor
	   */
	  public ChiSquaredOnline() {
	    resetOptions();
	  }

	private void resetOptions() {
		 m_ChiSquareds = null;
		    m_missing_merge = true;
		    m_Binarize = false;
		
	}

	public void updateEvaluator(Instance inst) throws Exception {
		
		if(counts == null) {
		    // can evaluator handle data?
			weka.core.Instance winst = new weka.core.DenseInstance(inst.weight(), inst.toDoubleArray());
			ArrayList<Attribute> list = new ArrayList<Attribute>();
		  	//ArrayList<Attribute> list = Collections.list(winst.enumerateAttributes());
		  	//list.add(winst.classAttribute());
			for(int i = 0; i < inst.numAttributes(); i++) 
				list.add(new Attribute(inst.attribute(i).name(), i));
		  	weka.core.Instances data = new weka.core.Instances("single", list, 1);
		  	data.setClassIndex(inst.classIndex());
		  	data.add(winst);
		    //getCapabilities().testWithFail(data);
		    classIndex = inst.classIndex();
		    counts = (HashMap<Key, Float>[]) new HashMap[inst.numAttributes()];
		    for(int i = 0; i < counts.length; i++) counts[i] = new HashMap<Key, Float>();
	  	}
	      for (int i = 0; i < inst.numValues(); i++) {
	        if (inst.index(i) != classIndex) {
	        	Key key = new Key((float) inst.valueSparse(i), (float) inst.classValue());
	        	Float cval = (float) (counts[inst.index(i)].getOrDefault(key, 0.0f) + inst.weight());
	        	counts[inst.index(i)].put(key, cval);
	        }
	      }
	  	

	      
	      updated = true;
	   
	}

	public void applySelection() {
		updated = false;
		
	}

	public boolean isUpdated() {
		// TODO Auto-generated method stub
		return updated;
	}

	public double evaluateAttribute(int attribute) throws Exception {
		
		return m_ChiSquareds[attribute];
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}

	
	private class Key {

	    final float x;
	    final float y;

	    public Key(float x, float y) {
	        this.x = x;
	        this.y = y;
	    }
	    

	    @Override
	    public boolean equals(Object o) {
	        if (this == o) return true;
	        if (!(o instanceof Key)) return false;
	        Key key = (Key) o;
	        return x == key.x && y == key.y;
	    }

	    @Override
	    public int hashCode() {
	        int result = Float.floatToIntBits(x);
	        result = 31 * result + Float.floatToIntBits(y);
	        return result;
	    }

  }


}
