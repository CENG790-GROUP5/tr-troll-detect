package edu.metu.ceng790.project

import edu.metu.ceng790.project.TrollTweets.NEWLINE
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}

object ArchivedTweets {
  val TWEETS_FOLDER_FORMAT: String = "data/stream/%s/%s/*.json.bz2"
  val OUTPUT_FOLDER_PATH: String = "output_archive/%s_%s"

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

      for (day <- List("02", "06", "08", "10", "13", "17", "18", "23", "25", "28");
           hour <- List("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23")) {  // TODO
        println(s"Starting Day: $day, Hour: $hour")
        val archivedTweets = spark.read.json(TWEETS_FOLDER_FORMAT.format(day, hour))

        //archivedTweets.printSchema()
        //println(s"Archived Tweets Count: ${archivedTweets.count()}")

        val tweetsRDD = archivedTweets
          .select("text", "user.followers_count", "user.friends_count", "user.verified", "lang", "extended_tweet.full_text", "retweeted_status")
          .filter(r => r.get(0) != null && r.get(1) != null && r.get(2) != null && r.get(3) != null && r.get(4) != null && r.get(6) == null)
          .rdd.map(r => (if (r.get(5) == null) r.getString(0) else r.getString(5), r.getLong(1), r.getLong(2), r.getBoolean(3), r.getString(4)))
          .filter(r => r._5 == "tr")
          .map(r => (r._1.trim.replaceAll(NEWLINE, " "), r._2, r._3, r._4))

        //tweetsRDD.take(5).foreach(println)
        //println(s"Filtered Archived Tweets Count: ${tweetsRDD.count()}")
        tweetsRDD.map(r => r._1 + "," + r._2 + "," + r._2 + "," + r._4)
          .saveAsTextFile(OUTPUT_FOLDER_PATH.format(day, hour))
        println(s"Finished Day: $day, Hour: $hour")
      }

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}
