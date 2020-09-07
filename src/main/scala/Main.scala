import base.Gaussian
import breeze._
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Success
object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")
    val data:DenseMatrix[Double] = DenseMatrix.tabulate(5,5){(x,y) => (x*5 + y).toDouble}
    println(data)
    val g = Gaussian(0,0,0) fit data
    g match {
      case Success(Gaussian(scope, mean, std,_,_)) => println(mean)
    }
  }
}
/*
import scala.language.postfixOps
import breeze.linalg._

sealed abstract class SPN(ft: Function) {
  def evaluate():Double
  def n_nodes():Int
  def fit(data:DenseMatrix[Double]):SPN
}
sealed trait Function


sealed trait NodeFunction extends Function
case class Sum(w:Double,logw:Double) extends NodeFunction
case class Prod() extends NodeFunction

case class Node(ft: NodeFunction, parent:Node,children: List[SPN], scope:DenseVector[Int]) extends SPN(ft) {
  override def evaluate() = ???
  override def fit(data:DenseMatrix[Double]): SPN = ???
  override def n_nodes(): Int = children map (_.n_nodes()) sum
}


trait LeafFunction extends Function
case class Uniform(value:Double) extends LeafFunction
case class Indicator(value:Double) extends LeafFunction
case class Multinomial(
                        p:Double,
                        logp:Double,
                        logcounts:Double
                      ) extends LeafFunction
case class Gaussian(
                     mean:Double,
                     std:Double,
                     a:Double,
                     b:Double,
                     upper: Double,
                     lower: Double,
                   ) extends LeafFunction

case class Leaf(ft:LeafFunction) extends SPN(ft) {
  override def evaluate() = ???
  override def fit(data:DenseMatrix[Double]): SPN = ???
  override def n_nodes(): Int = 1
}
 */



