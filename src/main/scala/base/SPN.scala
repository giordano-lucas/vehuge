package base
import scala.language.postfixOps
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.util.{Success, Try}
/* ====================================================================================
   ============================ SPN tree Abstract types ===============================
   ==================================================================================== */

sealed abstract class SPN(scope:DenseVector[Int]) {
  def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double]
  def size():Int
}
sealed abstract class Leaf(scope:Int) extends SPN(DenseVector(scope)) { override def size(): Int = 1 }
sealed abstract class Node(scope:DenseVector[Int],parent:Node,children: List[SPN]) extends SPN(scope) {
  override def size(): Int = children map (_.size()) sum
}
/* ====================================================================================
   ========================== Construction abstract types =============================
   ==================================================================================== */

trait Fitable {
  /* Fitable is the returning type of a builder and represents
  the idea of an ongoing construction object than still needs
  data to be able to terminate the actual construction

   /!\ can fail if data is not in the right format
   */
  def fit(data:DenseMatrix[Double]):Try[SPN]
}
trait SPNBuilder[T] {
  /* SPNBuilder must be implemented by companion objects of all
   SPN class. It simply ensures that the object return by calling
   the apply function of the companion object can be fitted with
   some data in the future.
   */
  def apply(arg:T): Fitable
}
/* ====================================================================================
   ==================================== Node Classes ==================================
   ==================================================================================== */
case class Sum(scope:DenseVector[Int],parent:Node,children: List[SPN],w:Double,logw:Double) extends Node(scope,parent,children) {
  override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] = ???
}
case class Prod(scope:DenseVector[Int],parent:Node,children: List[SPN]) extends Node(scope,parent,children) {
  override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] = ???
}

/* ====================================================================================
   ==================================== Leaf Classes ==================================
   ==================================================================================== */

// ---------------------- Uniform -----------------------
case class Uniform(scope:Int,value:Double) extends Leaf(scope) {
  override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] = ???
}

// ---------------------- Indicator -----------------------
case class Indicator(scope:Int,value:Double) extends Leaf(scope) {
  override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] = ???
}
// ---------------------- Multinomial -----------------------
case class Multinomial(
                        scope:Int,
                        p:Double,
                        logp:Double,
                        logcounts:Double
                      ) extends Leaf(scope)
{
override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] = ???
}
object Multinomial extends SPNBuilder[(Int,Int)] {
  override def apply(arg: (Int, Int)): Fitable = {
    val (scope,k) = arg

  }
}
// ---------------------- Gaussian -----------------------
case class Gaussian(
                     scope:Int,
                     mean:Double,
                     std:Double,
                     a:Double,
                     b:Double
                   ) extends Leaf(scope)
{
  override def evaluate(data:DenseMatrix[Double]):DenseMatrix[Double] =
    Utils.logtruncPhi(distributions.Gaussian(mean,std))(data.toDenseVector,a,b).toDenseMatrix
}
object Gaussian extends SPNBuilder[(Int,Double,Double)] {
   override def apply(arg:(Int,Double,Double)):Fitable = new Fitable {
    val (scope,a, b) = arg
    def fit(data: DenseMatrix[Double]): Try[Gaussian] =
      Success(Gaussian(   //to be finished
                scope,
                breeze.stats.mean(data(::, scope)),
                if (data.rows > 0) breeze.stats.stddev(data(::, scope)) else 1, //should change default values
                Double.NegativeInfinity,
                Double.PositiveInfinity
              ))
  }
}



