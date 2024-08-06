/*
   Sam Chandler
   Final Project code for Spark with Scala
   ALS implementation of collaborative filtering on the MovieLens 25M dataset
*/

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.{DataFrame, SparkSession}

// Define the schema for the ratings data
val schema = StructType(
  Array(
    StructField("userId", IntegerType, true),
    StructField("movieId", IntegerType, true),
    StructField("rating", FloatType, true),
    StructField("timestamp", LongType, true)
  )
)

// Define the schema for the movie data
val movieSchema = StructType(
  Array(
    StructField("movieId", IntegerType, true),
    StructField("title", StringType, true),
    StructField("genres", StringType, true)
  )
)


// Read the ratings CSV file with specified schema and comma separator
val ratings = spark.read.schema(schema).option("header", "false").csv("ml-25m/ratings.csv")

// Rename columns if header exists
val renamedRatings = ratings.toDF("userId", "movieId", "rating", "timestamp")

// Read in the movies CSV file
val movies = spark.read.schema(movieSchema).option("header", "false").csv("ml-25m/movies.csv")
val renamedMovies = movies.toDF("movieId", "title", "genres")


// Print the schema to verify
renamedRatings.printSchema()
renamedMovies.printSchema()

// Make ratings not null
val nonNullRatings = renamedRatings.filter("userId IS NOT NULL")

// Using an 80/20 split for training/testing
val Array(training, test) = nonNullRatings.randomSplit(Array(0.8, 0.2))

/*
  // This code has been commented out
  // This is used for generating a subset (in case fitting the model for the entire dataset is not possible)


  // Define a function to create a subset of data
  def createSubset(data: DataFrame, fraction: Double): DataFrame = {
    data.sample(withReplacement = false, fraction = fraction)
  }

  // This is the fraction of the dataset that you want (i.e., .10 for 10%)
  val fraction = 0.10

  // Create a subset of the training data
  val subsetTraining = createSubset(training, fraction)

  // Fit the ALS model using the subset of training data
  val model = als.fit(subsetTraining)
*/

/*
  ALS model parameters
  maxIter: maximum iterations for ALS, 16 is good
  regParam: regularization parameter, 0.01 is the best for this dataset
*/
val als = new ALS()
  .setMaxIter(16)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")

// Fit the ALS model using the subset of training data
val model = als.fit(training)


// Evaluate the model by computing the RMSE on the test data
// Cold start strategy is set to 'drop' to ensure no NaN evaluation metrics
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")



// Specify the user IDs which you want recommendations for
val usersToRecommendFor = Seq(1, 2, 3)

// Create a DataFrame with the specified user IDs
val usersDF = usersToRecommendFor.toDF("userId")

// Generate recommendations for the specified users
val userRecs = model.recommendForUserSubset(usersDF, 10)

// Show the recommendations
userRecs.show(truncate = false)



// The following method is the primary testing method

/*
 This method generates a specified number of ratings for a specified user
 Input: userId and number of recommendations (Integers)
 Output: DataFrames for the user's existing ratings and the user's new recommendations
*/
def generateRecommendationsFor(user: Int, numRecs: Int): (DataFrame, DataFrame) = {
  val userToRecommendFor = Seq(user)
  val userDF = userToRecommendFor.toDF("userId")
  val userRecs = model.recommendForUserSubset(userDF, numRecs)

  val userRatings = renamedRatings.filter(col("userId") === user)
  // Joining with the movies.csv for better looking output
  val userRatingsWithMovies = userRatings.join(renamedMovies, Seq("movieId"), "left").orderBy(desc("rating"))

  // exploding recommendations to allow for normal output and joining
  val explodedRecs = userRecs.withColumn("recommendation", explode(col("recommendations")))
  val userRecsWithMovies = explodedRecs.select(col("userId"), col("recommendation.movieId"))
  val userRecsWithDetails = userRecsWithMovies.join(renamedMovies, Seq("movieId"), "left")

  println(s"These are the top rated movies for user $user")
  userRatingsWithMovies.show(truncate = false)
  println(s"These are the top recommended movies for user $user")
  userRecsWithDetails.show(truncate = false)

  // Return userRatingsWithMovies and userRecsWithDetails (both sorted)
  (userRatingsWithMovies, userRecsWithDetails)
}

val (previousRatings, newRecommendations) = generateRecommendationsFor(100001, 20)
