package base
import breeze.linalg.DenseVector
import scala.Double.NaN

object Stats {
    /*###############################
    ##### Independence Tests ########
    ###############################*/

    //main type
    type IndepTest = (DenseVector[Double],DenseVector[Double]) => (Double,Double)
    //tests
    val kendalltau:IndepTest = (x,y) => if (x.size != y.size || x.map(_.isNaN).reduce(_||_) || y.map(_.isNaN).reduce(_||_)) (NaN,NaN)
    else {
      ???
    }
    val kruskal:IndepTest = (x,y) => ???
    val chi_test:IndepTest = (x,y) => ???

    /*###############################
    ##### Auxiliary Functions #######
    ###############################*/

}
