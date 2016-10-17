  import org.apache.spark.mllib.evaluation.{RegressionMetrics, RankingMetrics}
  import org.apache.spark.mllib.recommendation.{ALS, Rating}

  // Read in the ratings data
  val ratings = sc.textFile("s3://beeva-research-lab/movielens/ml10M/u5.train", 1).map { line =>
  //val ratings = sc.textFile("/home/enrique/proyectos/movielens/ml-100k/u5.base").map { line =>
    val fields = line.split("\t")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
  }.cache()


  val ratingstest = sc.textFile("s3://beeva-research-lab/movielens/ml10M/u5.test", 1).map { line =>
  //val ratingstest = sc.textFile("/home/enrique/proyectos/movielens/ml-100k/u5.test").map { line =>
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
val rank = 10
val numIterations = 10
val lambda = 0.015
val numBlocks = -1
val alpha = 1.0
val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, numBlocks, alpha)

val K = 10
val NUM_RECS = 10

// Get sorted top (ten) K predictions for each user and then scale from [0, 1]
val userRecommended = model.recommendProductsForUsers(NUM_RECS)

// Assume that any movie a user rated 3 or higher (which maps to a 1) is a relevant document
// Compare with top K recommendations
val userTest = binarizedRatings.groupBy(_.user)
val userJoin = userTest.join(userRecommended)

var relevantDocuments = userJoin.map { case (user, (actual, predictions)) =>
  (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
}

// Skip users with no data in test
relevantDocuments = relevantDocuments.filter(_._2.length>0)

// Instantiate metrics object
val metrics = new RankingMetrics(relevantDocuments)

// Mean average precision
println(s"Mean average precision = ${metrics.meanAveragePrecision}")

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

System.exit(0)
