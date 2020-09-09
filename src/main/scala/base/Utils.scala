package base
import breeze.linalg.{DenseMatrix, DenseVector, convert}
import breeze.stats.distributions.Gaussian
import breeze.numerics._

object Utils {

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
  //Cluster the variables in data into independent clusters.
  def getIndepClusters(data:DenseMatrix[Double],scope:DenseVector[Int],ncat:DenseVector[Int],thr:Double) = {
    //helper functions
    def depfunc(i:Int,deplist:Vector[Int]) = if(i==deplist(i)) i else deplist(i)
    def nbUnique(l:DenseVector[Int]):Int = l.toScalaVector.distinct.length
    val n = scope length // number of variables in the scope
      for {
        i <- 0 until n-1
        j <- i+1 until n
      } yield {
        val (si,sj) = (scope(i),scope(j))
        val (nbi,nbj) = (nbUnique(convert(data(::,si),Int)),nbUnique(convert(data(::,sj),Int)))
      }
  }



}


