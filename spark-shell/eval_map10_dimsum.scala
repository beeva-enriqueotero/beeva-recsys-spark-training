import org.apache.spark.mllib.evaluation.{RegressionMetrics, RankingMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vectors._
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry


val PATH = "https://raw.githubusercontent.com/beeva-enriqueotero/beeva-recsys-spark-training/master/data/u1.base"
//val PATH = "/home/enrique/proyectos/movielens/ml-100k/"
//val PATH = "s3://beeva-research-lab/movielens/ml10M/"
val TRAINFILE = "u1.base"
val TESTFILE = "u1.test"

  // Read in the ratings data
  val ratings = sc.textFile(PATH + TRAINFILE).map { line =>
    val fields = line.split("\t")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
  }.cache()


  val ratingstest = sc.textFile(PATH + TESTFILE).map { line =>
    val fields = line.split("\t")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
  }.cache()

  // Map ratings to 1 or 0, 1 indicating a movie that should be recommended
  val binarizedRatings = ratingstest.map(r => Rating(r.user, r.product,
    if (r.rating > 0) 1.0 else 0.0)).cache()

  // Summarize ratings
  val numRatings = ratings.count()
  val numUsers = ratings.map(_.user).distinct().count()
  val numMovies = ratings.map(_.product).distinct().count()
  println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")


// Build the model

val K = 10
val NUM_RECS = 10


val itemsAll = ratings.map(r => (r.product,r.user))
val itemsFiltered = ratings.filter(_.rating>0).map(r => (r.product,r.user))
val numItems = itemsFiltered.keys.distinct().count()
println("num items : "+numItems)

val maxItem = itemsFiltered.keys.max() + 1

val users = itemsAll.map{case (item,user) => (user,item)}


def getUserVectors(rdd : org.apache.spark.rdd.RDD[(Int,Int)],minItemsPerUser : Int,maxItem :Int) : org.apache.spark.rdd.RDD[Vector] =
  {
    rdd.groupByKey().filter(_._2.size >= minItemsPerUser)
     .map{ case (user,items) =>
      Vectors.sparse(maxItem, items.map(item => (item,1.toDouble)).toSeq)
      }
  }

val begin = System.currentTimeMillis
val userVectors = getUserVectors(users, 1, maxItem)

val numUsers = userVectors.count()
println("Number of users : "+numUsers)

val r = new RowMatrix(userVectors);

val THRESHOLD = 0.1
println("Running item similarity with threshold :"+THRESHOLD)
def runDimSum(r :RowMatrix,dimsumThreshold : Double) : org.apache.spark.rdd.RDD[MatrixEntry] =
{
  r.columnSimilarities(dimsumThreshold).entries
}

val simItems = runDimSum(r, THRESHOLD)


/* Get TopK recommendations per user */

def sortAndLimit(similarities : org.apache.spark.rdd.RDD[MatrixEntry],limit : Int) = {
    val v = similarities.map{me => (me.i,(me.j,me.value))}.groupByKey().mapValues(_.toSeq.sortBy{ case (domain, count) => count }(Ordering[Double].reverse).take(limit)).flatMapValues(v => v)
    v
  }

val LIMIT = 100
val mysims = sortAndLimit(simItems,LIMIT)
val end = System.currentTimeMillis
println("Elapsed time for model " + (end-begin)/1000.0 + "s")


/* Generate K recomendations for every user*/

val mysims2 = mysims.map{case (id1,(id2,score)) => (id1.toInt, (id2, score))}
val NUM_INTERACTIONS = 50

val itemsFiltered2 = ratings.filter(_.rating>=0.0).map(r => (r.user,r.product)).groupByKey().mapValues(_.take(NUM_INTERACTIONS)).flatMapValues(v=>v).map{case(u,i)=>(i,u)}
//val itemsFiltered2 = ratings.filter(_.rating>=0.0).map(r => (r.user,(r.product,r.rating))).groupByKey().mapValues(_.toArray.sortBy{ case (domain, count) => count }(Ordering[Double].reverse)).flatMapValues(v=>v).map{case(u,(i,r))=>(i,u)}
//val itemsFiltered2 = ratings.filter(_.rating>=0.0).map(r => (r.user,(r.product,r.rating))).groupByKey().mapValues(_.toArray.sortBy{ case (domain, count) => count }(Ordering[Double])).flatMapValues(v=>v).map{case(u,(i,r))=>(i,u)}
val myjoin=mysims2.join(itemsFiltered2)
val myjoin2 = myjoin.map{case (id1, ((id2, score), user)) => ((user, id2), score)}
val myjoin3 = myjoin2.reduceByKey(_+_)
val myjoin4 = myjoin3.map{case ((user,id2),score) => (user,(id2,score))}
val myrecs = myjoin4.groupByKey().mapValues(_.toArray.sortBy{ case (domain, count) => count }(Ordering[Double].reverse).take(NUM_RECS)).flatMapValues(v => v)

val myrecs2 = myrecs.map{case (user,(item,score)) => (user,Rating(user,item.toInt,score))}.groupByKey()
val userRecommended = myrecs2.map{case (user, iterable) => (user,iterable.toArray)}

// Remove from recommendations items in training set
val itemsFiltered3 = itemsFiltered2.map(x=>(x._2,x._1))
val userRecommended2 = userRecommended.cogroup(itemsFiltered3)
val userRecommended3 = userRecommended2.map(x=>(x._1, (x._2._1.flatMap(r=>r).filter(r=> !x._2._2.toArray.contains(r.product))).toArray))

/* Eval */

// Assume that any movie a user rated 3 or higher (which maps to a 1) is a relevant document
// Compare with top K recommendations
val userTest = binarizedRatings.groupBy(_.user)
val userJoin = userTest.join(userRecommended3)

var relevantDocuments = userJoin.map { case (user, (actual, predictions)) =>
  (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
}

// Skip users with no data in test
relevantDocuments = relevantDocuments.filter(_._2.length>0)




// Instantiate metrics object
val metrics = new RankingMetrics(relevantDocuments)

val begin = System.currentTimeMillis
// Mean average precision
println(s"Mean average precision = ${metrics.meanAveragePrecision}")
val end = System.currentTimeMillis
println("Elapsed time for MAP " + (end-begin)/1000.0 + "s")

lazy val meanAveragePrecisionAtK: Double = {
    relevantDocuments.map { case (pred, lab) =>
      val labSet = lab.toSet

      if (labSet.nonEmpty) {
        var i = 0
        var cnt = 0
        var precSum = 0.0

        var pred2 = pred.slice(0,K)
        val n = pred2.length

        while (i < n) {
          if (labSet.contains(pred2(i))) {
            cnt += 1
            precSum += cnt.toDouble / (i + 1)
          }
          i += 1
        }

        0.0+precSum / math.min(labSet.size,K)
      } else {
        println("Empty ground truth set, check input data")
        0.0
      }
    }.mean()
  }

// Mean average precision
//println(s"Mean average precision = ${metrics.meanAveragePrecision}")
meanAveragePrecisionAtK

userRecommended.flatMapValues(x=>x).count()

/* Example */
val movies = sc.textFile(PATH + "u.item")
val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt,  array(1))).collectAsMap()
titles(123)
users.lookup(310).map(titles)
userRecommended.lookup(310).flatMap(r=>r).map(r=>(titles(r.product), r.rating))



System.exit(0)
