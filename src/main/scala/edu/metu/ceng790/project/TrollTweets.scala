package edu.metu.ceng790.project

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import org.apache.log4j.Logger
import org.apache.log4j.Level

object TrollTweets {
  val ACCOUNTS_FILE: String = "data/fake/accounts.csv"
  val TWEETS_FOLDER: String = "data/fake/tweets/*.csv"
  val TWEETS_FILE: String = "data/fake/tweets/tweets_2020_01.csv"
  val DATE_TIME_FORMAT = "yyyy-LL-dd"

  val STATE_BACKED_TWEET_COLUMN_NAMES = Seq("tweetid", "user_screen_name", "user_reported_location",
    "user_profile_description", "follower_count", "following_count", "account_creation_date", "tweet_text",
    "tweet_time", "hashtags")

  val REG = raw"[^A-Za-z0-9\s]+"
  val NEWLINE = raw"[\n\r\t]+"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Examine Dataset")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir("checkpoint")

      val tweets = spark.read
        .format("csv")
        .option("delimiter", ",")
        .option("header", "true")
        .option("escape", "\"")
        .option("multiLine", "true")
        .load(TWEETS_FILE)
      tweets.printSchema()

      println(s"Troll Tweets Count: ${tweets.count()}")

      val tweetsRDD = tweets.select("user_screen_name","tweet_text", "tweet_language", "is_retweet")
        .rdd.map(r => (r.getString(0), r.getString(1), r.getString(2), r.getString(3)))
        .filter(r => r._3 == "tr")
        .filter(r => r._4 == "false")
        .map(r => (r._1, r._2.trim))

      tweetsRDD.take(50).foreach(println)
      println(s"Filtered Troll Tweets Count: ${tweetsRDD.count()}")

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}
