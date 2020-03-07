package org.uma.jmetal.problem.multiobjective;

import org.uma.jmetal.problem.impl.AbstractDoubleProblem;
import org.uma.jmetal.solution.DoubleSolution;

import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("serial")
public class QuadraticCubic extends AbstractDoubleProblem {

  /** コンストラクタ */
  public QuadraticCubic() {
    setNumberOfVariables(1);  // 変数は1つ
    setNumberOfObjectives(2); // 目的は2つ
    setName("QuadraticCubic");

    // 決定変数の上下限値
    List<Double> lowerLimit = new ArrayList<>(getNumberOfVariables()) ;
    List<Double> upperLimit = new ArrayList<>(getNumberOfVariables()) ;
    for (int i = 0; i < getNumberOfVariables(); i++) {
      lowerLimit.add(-5.0); // 変数の最小値は-5
      upperLimit.add(5.0); // 変数の最大値は5
    }
    setLowerLimit(lowerLimit);  // jMetal5
    setUpperLimit(upperLimit);  // jMetal5
    // setVariableBounds(lowerLimit, upperLimit);  // jMetal6
  }

  /** 目的関数計算 */
  public void evaluate(DoubleSolution solution) {
    double[] x = new double[getNumberOfVariables()];
    double[] f = new double[getNumberOfObjectives()];

    // 決定変数の取得
    for ( int i=0; i<getNumberOfVariables(); i++ ) x[i]=solution.getVariableValue(i); // jMetal5
    // for ( int i=0; i<getNumberOfVariables(); i++ ) x[i]=solution.getVariable(i);  // jMetal6

    // 目的関数の計算
    f[0] = x[0]*x[0];       // 第1目的関数：二次関数
    f[1] = x[0]*x[0]*x[0];  // 第2目的関数：三次関数

    // 目的関数値の代入
    for ( int i=0; i<getNumberOfObjectives(); i++ ) solution.setObjective(i, f[i]);
  }
}
