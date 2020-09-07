package base
import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import breeze.numerics._

object Utils {
  type Data = Double

  def phi(g:Gaussian)(t:Double) = //PDF for normal distribution at point t
    exp(g.unnormalizedLogPdf(t))*g.normalizer

  def logtruncPhi(g:Gaussian)(x:DenseVector[Double], a:Double, b:Double):DenseVector[Double] = {
    val denom = g.probability(a,b)
    x.map(e =>
      if (e < a || e > b) Double.NegativeInfinity
      else if (e.isNaN) 1
      else log(phi(g)(e)/denom)
    )
  }

}
