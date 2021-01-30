#!/usr/bin/env python3

from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from glob import glob
import os


SparkContext.setSystemProperty('spark.executor.memory', '10g')

def parse_args():
    parser = ArgumentParser(description="Extract data from archived tweets")
    parser.add_argument("input_files", help="Archived tweets in json format")
    parser.add_argument("output_path", help="Output path for processed tweets")
    return parser.parse_args()

def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="extract_verified")
    spark = SparkSession(sc)

    tweets = spark.read.json(args.input_files)

    if "retweeted_status" in tweets.columns:
        filtered_tweets = tweets\
            .filter(F.col("lang") == "tr")\
            .filter(F.col("user.verified"))\
            .filter(F.col("retweeted_status").isNull())
    else:
        filtered_tweets = tweets\
            .filter(F.col("lang") == "tr")\
            .filter(F.col("user.verified"))

    reformatted_filtered_tweets = filtered_tweets\
        .withColumn("user_profile_description", F.col("user.description"))\
        .withColumn("follower_count", F.col("user.followers_count"))\
        .withColumn("following_count", F.col("user.friends_count"))\
        .withColumn("account_creation_date", F.col("user.created_at"))\
        .withColumn("tweet_text", F.when(F.col("extended_tweet.full_text").isNull(), F.col("text")).otherwise(F.col("extended_tweet.full_text")))\
        .withColumn("tweet_time", F.col("created_at"))

    print(f"Shrunk Tweets ({args.input_files}) Count: {tweets.count()}")
    print(f"Filtered Tweets Count: {filtered_tweets.count()}")

    output_columns = ["user_profile_description", "follower_count", "following_count", "account_creation_date",
                      "tweet_text", "tweet_time"]
    reformatted_filtered_tweets.select(output_columns).write\
        .mode('overwrite')\
        .options(delimiter=',', escape='"', header=True, multiLine=True, quote='"', quoteAll=True)\
        .csv(args.output_path)


if __name__ == "__main__":
    exit(main())
