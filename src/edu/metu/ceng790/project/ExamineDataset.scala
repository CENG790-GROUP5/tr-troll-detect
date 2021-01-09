package edu.metu.ceng790.project

import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.{col, date_format, to_date, when}
import org.apache.spark.sql.{Row, SparkSession}

import scala.reflect.internal.util.TableDef.Column


object ExamineDataset {
  val ACCOUNTS_FILE: String = "data/fake/accounts.csv"
  val TWEETS_FOLDER: String = "data/fake/tweets/*.csv"
  val TWEETS_FILE: String = "data/fake/tweets/tweets_2020_01.csv"
  val DATE_TIME_FORMAT = "yyyy-LL-dd"

  def mapDate(s: String): String = {
    if (s != null && s.split("-").length == 3) {
      val year = s.split("-")(0)
      val month = s.split("-")(1)
      s"$year-$month"
    } else {
      "unknown"
    }
  }

  val STATE_BACKED_TWEET_COLUMN_NAMES = Seq("tweetid", "user_screen_name", "user_reported_location",
    "user_profile_description", "follower_count", "following_count", "account_creation_date", "tweet_text",
    "tweet_time", "hashtags")

  def main(args: Array[String]): Unit = {
    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Examine Dataset")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir("checkpoint")

      val accounts = spark.read
        .format("csv")
        .option("delimiter", ",")
        .option("header", "true")
        .option("escape", "\"")
        .option("multiLine", "true")
        .load(ACCOUNTS_FILE)
      accounts.printSchema()
      println(accounts.count())

      val tweets = spark.read
        .format("csv")
        .option("delimiter", ",")
        .option("header", "true")
        .option("escape", "\"")
        .option("multiLine", "true")
        .load(TWEETS_FOLDER)
      tweets.printSchema()
//      println(tweets.count()) // 38,248,105 (TOTAL)

      val accountCount = accounts.count()
      println(s"Troll Account Count: $accountCount")

//      // Troll Account Creation Times
//      val accountCreationDates = accounts.select("account_creation_date").rdd.map(r => r.getString(0))
//      accountCreationDates.map(r => mapDate(r))
//        .groupBy(r => r)
//        .map(r => (r._1, r._2.size))
//        .sortBy(r => r._1, numPartitions = 1)
//        .foreach(println)

      val tweetsRDD = tweets.select("userid", "account_creation_date")
        .rdd.map(r => (r.getString(0), r.getString(1)))
        .groupBy(r => (r._1, r._2))
        .map(r => (r._1._1, r._1._2, r._2.size))

      println(s"Troll Account Count From Troll Tweets: ${tweetsRDD.count()}")
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}

