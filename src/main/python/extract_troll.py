#!/usr/bin/env python3

from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SparkContext.setSystemProperty('spark.executor.memory', '10g')

def parse_args():
    parser = ArgumentParser(description="Extract data from troll tweets file")
    parser.add_argument("input_file", help="Troll tweets file")
    parser.add_argument("output_path", help="Output path for processed troll tweets")
    return parser.parse_args()

def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="extract_troll")
    spark = SparkSession(sc)
    troll_tweets = spark.read\
        .options(delimiter=',', escape='"', header=True, multiLine=True, quote='"')\
        .csv(args.input_file)

    filtered_troll_tweets = troll_tweets\
        .filter(F.col("tweet_language") == "tr")\
        .filter(F.col("is_retweet") == "false")\
        .filter(F.col("tweet_time").contains("2020-01"))

    print(f"Troll Tweets ({args.input_file}) Count: {troll_tweets.count()}")
    print(f"Filtered Troll Tweets Count: {filtered_troll_tweets.count()}")

    output_columns = ["user_profile_description", "follower_count", "following_count", "account_creation_date",
                      "tweet_text", "tweet_time"]
    filtered_troll_tweets.select(output_columns).write\
        .mode('overwrite')\
        .options(delimiter=',', escape='"', header=True, multiLine=True, quote='"', quoteAll=True)\
        .csv(args.output_path)


if __name__ == "__main__":
    exit(main())
