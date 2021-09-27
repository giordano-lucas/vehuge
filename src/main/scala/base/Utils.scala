package base
import breeze.linalg.{DenseMatrix, DenseVector, convert, sum}
import breeze.stats.distributions.Gaussian
import breeze.numerics._
import breeze.util._
import Conversions._
import base.Stats.IndepTest

import scala.language.implicitConversions

object Utils {
  // implicits definitions for masking operation type
  implicit def bool_to_int(b:Boolean):Int = if (b) 1 else 0
  implicit def bool_to_int(b:DenseVector[Boolean]):DenseVector[Int] = b map bool_to_int

  //PDF for normal distribution at point t
  def phi(g: Gaussian)(t: Double) =
    exp(g.unnormalizedLogPdf(t)) * g.normalizer

  //log of normaj distribution + handling of NaN case
  def logtruncPhi(g: Gaussian)(x: DenseVector[Double], a: Double, b: Double): DenseVector[Double] = {
    val denom = g.probability(a, b)
    x.map(e =>
      if (e < a || e > b) Double.NegativeInfinity
      else if (e.isNaN) 1
      else log(phi(g)(e) / denom)
    )
  }
  //Bin counting with nBuckets
  def bincount(d: DenseVector[Int], nBuckets: Int): DenseVector[Int] = {
    def update(acc:DenseVector[Int],e:Int) = DenseVector.tabulate(nBuckets)(i=>if (i==e) 1 else 0) + acc
    val z: DenseVector[Int] = DenseVector.zeros(nBuckets)
    d.foldLeft(z)(update)
  }
  //deletes nan
  def extract(data:DenseMatrix[Double],i:Int) = data(data(::,i).map(!_.isNaN),i)
  //Cluster the variables in data into independent clusters.
  def getIndepClusters(data:DenseMatrix[Double],scope:DenseVector[Int],ncat:DenseVector[Int],thr:Double) = {
    //helper functions
    def depfunc(i:Int,deplist:Vector[Int]) = if(i==deplist(i)) i else deplist(i)
    def nbUnique(l:DenseVector[Int]):Int = l.toScalaVector.distinct.length
    val n = scope.size // number of variables in the scope
    val optpairs:IndexedSeq[Option[(Int, Int)]] = for { //all pairs (i,j)
      i <- 0 until n-1
      j <- i+1 until n
                                                        } yield {
      val (si,sj) = (scope(i),scope(j)) //scope
      val (nbi,nbj) = (nbUnique(convert(extract(data,si),Int)),nbUnique(convert(extract(data,sj),Int))) //computes the number of unique elements
      val mask:DenseVector[Boolean] = List(si,sj).map(col=> data(::,si).map(!_.isNaN)) reduce (_ &:& _) //only rows where both values are non NaNs
      val m = sum(mask map bool_to_int)

      val indepTest:IndepTest = (nbi > 1 && nbj > 1,m > 4, m > 2*nbi*nbj, ncat(si),ncat(sj)) match {
          case (true,true,_,1,1) => Stats.kendalltau                                    //both continuous
          case (true,true,_,1,_) => Stats.kruskal                                       // i continuous, j discrete
          case (true,true,_,_,1) => Stats.kruskal                                       // i discrite  , j continous
          case (true,false,true,cati,catj) if (cati > 1 && catj > 1) => Stats.chi_test  // both discrete
          case _                 => (x:DenseVector[Double],y:DenseVector[Double])=>(Double.NaN,thr)
      }
      val (vari,varj):(DenseVector[Double],DenseVector[Double]) = (data(mask,si).toDenseVector,data(mask,si).toDenseVector) //vectors without nans
      val res = indepTest(vari,varj)._2                 //run independence test
      if (res.isNaN || res < thr) Some((i,j)) else None //if nan or below threshold => not independent => same cluster
    }
    (optpairs.flatten foldLeft (0 until n).toVector) {
      case (acc,(i,j)) => if (acc(i) < acc(j)) acc.updated(j,acc(i)) else acc.updated(i,acc(j)) //since list is sorted we can simply assign the largest cluster to be equal to the smaller one
    }
  }
  // containing test densevector
  def contains (v: DenseVector[Int], e:Int):Boolean = sum(v :== e) > 0


}



