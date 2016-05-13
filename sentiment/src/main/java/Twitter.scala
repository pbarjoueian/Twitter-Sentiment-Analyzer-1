import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{ HashingTF, Tokenizer }

object Twitter {

  def main(args: Array[String]) {
    //create spark context 
    
    val conf = new SparkConf().setAppName("Twitter")
    val sc = new SparkContext(conf)

    //val dirLocation = "/user/vineet/recommender/ml_small/"

    //features
    // filter out positive words only

   val data = sc.parallelize(Seq(
  ("negative", "Hi I heard about Spark"),
  ("negative", "I wish Java could use case classes"),
  ("positive", "Logistic regression models are neat")
))

val polarity : Map[String,Int] = Map("good" -> 1, "bad" -> 0, "Hi" -> 5)
val t = data.map( x => x._2.split(" "))
.map(x => x.map(x => polarity.getOrElse(x,0)))
.collect()

data.persist()

 case class Twitter(status: String, location: String, time_stamp: Int) //case keyword makes class Twitter matchable
    def extractFeatures(tweet: String): Any = tweet match {

      case tweet: String => {
        val words = tweet.flatMap(lines => lines.split(" "))
        val taggedwords = words map (word => (word, polarity(word))).reduceByKey(_ + _)

        val positivebag = taggedwords flatMap (tuple => tuple._2) filter (polarity(_) > 0).collect().cache()

        val total_positive = positivebag count
        val total = positivebag.count()

        val tweets = Tweet(tweet,Seq(total_positive,total))
      }

      case _ => null //will need to filter Nulls after

    }

   

    //now that we have featureExtractor function we can load csv file and apply it to tweets

    val data = sc.textFile("input.csv").map(tweet => (tweet, featureExtractor(tweet))).toDF("status", "custom_features")

    //load all data into dataframe to load into model this is spark code uses new MLib library 

    val tokenizer = new Tokenizer()
      .setInputCol("status")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    //create VectorAssembler to combine both feature vectors 

    val assembler = new VectorAssembler()
      .setInputCols(Array("custom_features", "features"))
      .setOutputCol("feature_vector")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, assembler, lr))
    val model = pipeline.fit(data);
  }
}