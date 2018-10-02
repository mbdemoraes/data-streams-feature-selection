package moa.featureselection.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;

import java.util.Map.Entry;

import com.yahoo.labs.samoa.instances.Instance;


import moa.featureselection.common.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;


public class FastOSFS extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator{

	/** Treat missing values as a seperate value */
	private boolean m_missing_merge;

	/** Just binarize numeric attributes */
	private boolean m_Binarize;

	private HashMap<Key, Float>[] counts = null;

	private int classIndex;

	private double[] m_InfoValues = null;


	private boolean updated = false;

	private double gSquaredTest(double [][] matrix) {

		int  nrows, ncols, row, col, counter;
		double[] rtotal, ctotal;
		double n = 0;
		
		
		double sum = 0.0;

		nrows = matrix.length;
		ncols = matrix[0].length;
		long[] observed = new long[nrows * ncols];
		double[] expected = new double[nrows * ncols];
		rtotal = new double [nrows];
		ctotal = new double [ncols];
		counter = 0;
		for (row = 0; row < nrows; row++) {
			for (col = 0; col < ncols; col++) {
				rtotal[row] += matrix[row][col];
				ctotal[col] += matrix[row][col];
				observed[counter] += matrix[row][col];
				n += matrix[row][col];
				counter++;
			}
		}
		//df = (nrows - 1)*(ncols - 1);
		
		counter = 0;
		for (row = 0; row < nrows; row++) {
			for (col = 0; col < ncols; col++) {
				expected[counter] = (ctotal[col] * rtotal[row]) / n;
				counter++;
			}
		}

		/*if (expected.length < 2) {
			throw new DimensionMismatchException(expected.length, 2);
		}
		if (expected.length != observed.length) {
			throw new DimensionMismatchException(expected.length, observed.length);
		}*/
		
		if (expected.length > 2) {
			MathArrays.checkPositive(expected);
			MathArrays.checkNonNegative(observed);

			double sumExpected = 0d;
			double sumObserved = 0d;
			for (int i = 0; i < observed.length; i++) {
				sumExpected += expected[i];
				sumObserved += observed[i];
			}
			double ratio = 1d;
			boolean rescale = false;
			if (FastMath.abs(sumExpected - sumObserved) > 10E-6) {
				ratio = sumObserved / sumExpected;
				rescale = true;
			}
			
			for (int i = 0; i < observed.length; i++) {

				final double dev = rescale ?
						FastMath.log((double) observed[i] / (ratio * expected[i])) :
							FastMath.log((double) observed[i] / expected[i]);
						
						if(!Double.isInfinite(dev)) {
							sum += ((double) observed[i]) * dev;
						}
						
			}
		} 
				
		return 2d * sum;

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

	public void applySelection(){
		if(counts != null && updated) {
			m_InfoValues = new double[counts.length];
			for (int i = 0; i < counts.length; i++) {
				if (i != classIndex) {
					Set<Key> keys = counts[i].keySet();
					Set<Entry<Key, Float>> entries = counts[i].entrySet();

					Set<Float> avalues = new HashSet<Float>();
					Set<Float> cvalues = new HashSet<Float>();
					for (Iterator<Key> it = keys.iterator(); it.hasNext(); ) {
						Key key = it.next();
						avalues.add(key.x);
						cvalues.add(key.y);
					}

					Map<Float, Integer> apos = new HashMap<Float, Integer>();
					Map<Float, Integer> cpos = new HashMap<Float, Integer>();

					int aidx = 0;
					for(Iterator<Float> it = avalues.iterator(); it.hasNext();) {
						Float f = it.next();
						apos.put(f, aidx++);
					} 

					int cidx = 0;
					for(Iterator<Float> it = cvalues.iterator(); it.hasNext();) {
						Float f = it.next();
						cpos.put(f, cidx++);
					} 

					double[][] lcounts = new double[avalues.size()][cvalues.size()];		    	
					for (Iterator<Entry<Key, Float>> it = entries.iterator(); it.hasNext(); ) {
						Entry<Key, Float> entry = it.next();
						lcounts[apos.get(entry.getKey().x)][cpos.get(entry.getKey().y)] = entry.getValue();
					}


					//m_InfoValues[i] = gSquaredTest(ContingencyTables.reduceMatrix(lcounts));
					m_InfoValues[i] = ContingencyTables.chiVal(
	            	          ContingencyTables.reduceMatrix(lcounts), false);

				}
			}

			updated = false;
		}
	}

	public boolean isUpdated() {
		// TODO Auto-generated method stub
		return updated;
	}

	public double evaluateAttribute(int attribute) throws Exception {
		 return m_InfoValues[attribute];
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub

	}
	
	/**
	   * Describe the attribute evaluator
	   * 
	   * @return a description of the attribute evaluator as a string
	   */
	  @Override
	  public String toString() {
	    StringBuffer text = new StringBuffer();

	    if (m_InfoValues == null) {
	      text.append("Information Gain attribute evaluator has not been built");
	    } else {
	      
	        text.append("\nTesting");
	      
	    }

	    text.append("\n");
	    return text.toString();
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
