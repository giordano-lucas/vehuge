/*package base
import breeze.linalg._

class Learner(ncat:DenseVector[Int],
              thr:Double          =0.001,
              nclusters:Int       =2,
              max_height:Int      =1000000,
              classcol:Option[Int]=None) {
  /**
  Learning method based on Gens and Domingos' LearnSPN.

     Attributes
     ----------
     thr: float
     p-value threshold for independence tests in product nodes.
     nclustes: int
     Number of clusters in sum nodes.
     max_height: int
     Maximum height (depth) of the network.
     ncat: numpy array
     Number of categories of each variable in the data. It should be
     one-dimensional and of length equal to the number of variables.
     classcol: int
     The index of the column containing class variable.
   **/

  /**
   *
   * @param data : numpy array
   *             Rows are instances and columns variables.
   * @param last_node: Node object
   *             The parent of the structure to be learned (if any).
   * @return root: SPN object
   *         The SPN learned from data.
   */
  def fit(data:DenseMatrix[Double], last_node:Option[Node] = None):SPN = {

    def addNode(scope:DenseVector[Int], max_height:Int, last_node:Option[Node]):SPN = {
      /* Add a new node to the network. This function is called recursively
         until max_height is reached or the data is fully partioned into leaf nodes. */
      if (scope.length <= 1) addLeaf(scope, last_node) // Single variable in the scope

      val isProdNode = last_node map {_ match {
        case Prod(_,_,_) => true
        case _ => false}
      } getOrElse false

      (if (max_height <= 0 || !isProdNode) addProdNode(scope, max_height, last_node) else None)
        .orElse(addSumNode(scope, max_height, last_node)) /* We were not able to cut vertically, so we run clustering of the
                                                             data (each row is a point), and then we use the result of
                                                             clustering to create the children of a sum node. */
          .getOrElse(Prod(scope,last_node.get,List.empty)) // If no split found, assume fully factorised model.
    }
    def addProdNode(scope:DenseVector[Int], max_height:Int, last_node:Option[Node]):Option[Node] = {
      ???
    }
    def addSumNode(scope:DenseVector[Int], max_height:Int, last_node:Option[Node]):Option[Node] = {
      ???
    }
    def addLeaf(scope:DenseVector[Int], last_node:Option[SPN]):Leaf = {
      assert(scope.length == 1,"Univariate leaf should not have more than one variable in its scope.")
      val categories = ncat(scope(0))
      null
      /*if (categories > 1)
        //Multinomial() fit data
       */
    }

    val scope = DenseVector.tabulate(data.rows) {_.toInt}
    return addNode(scope,max_height,last_node)
  }

}*/
