package edu.metu.ceng790.project

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import org.apache.log4j.Logger
import org.apache.log4j.Level

object TrollTweets {
  val TWEETS_FOLDER: String = "data/fake/tweets/*.csv"
  val TWEETS_FILE: String = "data/fake/tweets/tweets_2020_01.csv"

  val REG = raw"[^A-Za-z0-9\s]+"
  val NEWLINE = raw"[\n\r\t]+"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Examine Troll Tweets Dataset")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir("checkpoint")

      val tweets = spark.read
        .option("delimiter", ",")
        .option("header", "true")
        .option("escape", "\"")
        .option("multiLine", "true")
        .csv(TWEETS_FILE)
      tweets.printSchema()

      println(s"Troll Tweets Count: ${tweets.count()}")

      val tweetsRDD = tweets.select("tweet_text", "tweet_language", "is_retweet")
        .rdd.map(r => (r.getString(0), r.getString(1), r.getString(2)))
        .filter(r => r._2 == "tr")
        .filter(r => r._3 == "false")
        .map(r => r._1.trim.replaceAll(NEWLINE, " "))

      tweetsRDD.take(10).foreach(println)
      println(s"Filtered Troll Tweets Count: ${tweetsRDD.count()}")

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}
