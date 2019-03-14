/*
 *    ExtremeFeatureSelection.java
 *    Copyright (C) 2018 University of Campinas, Brazil
 *    @author Matheus Bernardelli (matheuzmoraes@gmail.com)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

package moa.featureselection.algorithms;

import java.util.Arrays;
import com.yahoo.labs.samoa.instances.Instance;
import moa.featureselection.common.MOAAttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.AlgVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.*;
import java.util.ArrayList;




/**
 * <!-- globalinfo-start --> Extreme Feature Selection :<br/>
 * <br/>
 * Evaluates the worth of an attribute through the computation of weights 
 * based on the Modified Balanced Winnow.<br/>
 * <br/>
 * Carvalho, V. R.; Cohen, W. W. Single-pass online learning. Proceedings of the
 * 12th ACM SIGKDD international conference on Knowledge discovery and data mining -
 * KDD ’06, p. 548, 2006.
 * doi: 10.1145/1150402.1150466 <br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * 
 * @author Matheus Bernardelli (matheuzmoraes@gmail.com)
 */

public class UnsupervisedFeatureSelection extends ASEvaluation implements
AttributeEvaluator, MOAAttributeEvaluator{
	

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/** The promotion coefficient. According to the authors criteria.**/
	protected double m_Alpha = 1.5;

	/** The demotion coefficient. According to the authors criteria.**/
	protected double m_Beta = 0.5;

	/** Prediction threshold. According to the authors criteria.**/
	protected double m_Threshold = 1.0;

	/** Predefined margin. According to the authors criteria. **/
	protected double marginM = 1.0;

	/** Accumulated mistake count (for statistics) **/
	protected int m_Mistakes = 0;

	/** Starting weights for the prediction vector(s) **/
	protected double m_defaultWeight = 2.0;

	protected static double maxValue = 0.0;
	protected static double minValue = 0.0;

	protected double score = 0.0;
	protected AlgVector rankedAttributes = null;
	protected AlgVector rankedAttributes90 = null;
	private static double[][] B = null;

	Matrix C = null;

	/** The weight vectors for prediction **/
	private AlgVector m_predPosVector = null;
	private AlgVector m_predNegVector = null;
	
	private boolean rankAttributes = true;
    protected static int m_numAttribs = 0;
    protected int k_numClasses = 0;
    protected int ell = 0;

	private int numFeatures = 10;
	private boolean updated = false;
	private int instanceConunter = 0;
	protected int n = 0;
	
	/** Will hold the left singular vectors */
	  private Matrix m_u = null;
	 
	  /** Will hold the singular values */
	  private Matrix m_s = null;
	 
	  /** Will hold the right singular values */
	  private Matrix m_v = null;
	
	
	public UnsupervisedFeatureSelection(int numFeatures) {
		this.numFeatures = numFeatures;
	}

	
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

	  public static Matrix concat(double[][] first, double[] second) {
		  double[][] result = new double[m_numAttribs][B[0].length + 1];
		  
		  for(int i=0; i<m_numAttribs; i++) {
			  for(int j=0; j<B[0].length+1;j++) {
				  if(j==B[0].length) {
					  result[i][j] = second[i];
				  } else {
					  result[i][j] = B[i][j];
				  }
			  }
		  }

		  Matrix matrixResult = new Matrix(result);
		  return matrixResult;
		}


	/**
	 * 
	 */

	public void updateEvaluator(Instance inst) throws Exception {
		
		double[] rawx = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);

		if(B == null && C == null) {
			m_numAttribs = inst.numAttributes()-1;
			k_numClasses = inst.numClasses();
			ell = (int) Math.sqrt(m_numAttribs);
			B =  new double[m_numAttribs][ell];
			double[] cutYt = new double[ell];
			
			for(int i=0; i<m_numAttribs; i++) {
				for(int j=0; j<ell; j++) {
					if(j==0) {
						B[i][j] = rawx[i];
					} else {
						B[i][j] = 0.0;
					}
				}
				
			}
			
			System.arraycopy(rawx, 0, cutYt, 0, ell);
			
			C = concat(B, cutYt);
			n = m_numAttribs - ell;
			
		}
		
		SingularValueDecomposition svdC = C.svd();

		m_u = svdC.getU(); // left singular vectors
	    m_s = svdC.getS(); // singular values
	    m_v = svdC.getV(); // right singular vectors
		
		updated = true;


	}

	public void applySelection() {
		updated = false;

	}

	public boolean isUpdated() {
		return updated;
	}

	public double evaluateAttribute(int attribute) throws Exception {

		return rankedAttributes90.getElement(attribute);
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub

	}

}
