package edu.metu.ceng790.project

import edu.metu.ceng790.project.TrollTweets.NEWLINE
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}

object ArchivedTweets {
  val TWEETS_FOLDER: String = "data/stream/02/21/*.json.bz2"
  val TWEETS_ROOT_FOLDER: String = "data/stream"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var spark: SparkSession = null
    var sc: SparkContext = null
    var sqlContext: SQLContext = null
    try {
      spark = SparkSession.builder()
        .appName("Examine Archived Tweets Dataset")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir("checkpoint")

      val archivedTweets = spark.read.json(TWEETS_FOLDER)

//      archivedTweets.printSchema()
//      println(s"Archived Tweets Count: ${archivedTweets.count()}")

      val tweetsRDD = archivedTweets.select("text", "lang", "retweeted_status")
        .filter(r => r.get(0) != null && r.get(1) != null && r.get(2) == null)
        .rdd.map(r => (r.getString(0), r.getString(1)))
        .filter(r => r._2 == "tr")
        .map(r => (r._1.trim.replaceAll(NEWLINE, " ")))

//      tweetsRDD.take(25).foreach(println)
//      println(s"Filtered Archived Tweets Count: ${tweetsRDD.count()}")

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}
