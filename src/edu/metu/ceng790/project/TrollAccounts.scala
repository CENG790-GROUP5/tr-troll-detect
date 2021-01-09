package edu.metu.ceng790.project

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object TrollAccounts {
  val ACCOUNTS_FILE: String = "data/fake/accounts.csv"
  val DATE_TIME_FORMAT = "yyyy-LL-dd"

  def mapDate(s: String): String = {
    if (s != null) {
      if (s.split("-").length == 3) {
        val year = s.split("-")(0)
        val month = s.split("-")(1)
        s"$year-$month"
      } else {
        "unkown"
      }
    } else {
      "null"
    }
  }

  def main(args: Array[String]): Unit = {
    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Troll Accounts")
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

      val accountCount = accounts.count()
      println(s"Troll Account Count: $accountCount")
      println(s"Account Creation Date, Account Count: $accountCount")

      // Troll Account Creation Times
      val accountCreationDates = accounts.select("account_creation_date").rdd.map(r => r.getString(0))
      accountCreationDates.groupBy(r => mapDate(r))
        .map(r => (r._1, r._2.size))
        .sortBy(r => r._1, numPartitions = 1)
        .foreach(println)
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}

