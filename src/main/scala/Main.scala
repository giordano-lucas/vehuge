import base.Gaussian
import breeze._
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Success
object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")
    val data:DenseMatrix[Double] = DenseMatrix.tabulate(5,5){(x,y) => (x*5 + y).toDouble}
    println(data(data :== 0.0))
    val g = Gaussian(0,0,0) fit data
    g match {
      case Some(Gaussian(scope, mean, std,_,_)) => println(mean)
    }
  }
}

/*
*
*
* TO DO :
*
* Independance tests in stats.scala (following the file : 'statsutils.py')
*
* */