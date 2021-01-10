#!/usr/bin/env python3

from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from glob import glob
import os


def parse_args():
    parser = ArgumentParser(
        description="Extract data from troll csv")
    parser.add_argument("archive_path", help="A root folder where the archive of tweets stored")
    parser.add_argument("out_path", help="Output path for model")
    return parser.parse_args()


def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="extract_verified")
    spark = SparkSession(sc)

    for dayPath in glob(os.path.join(args.archive_path, "*")):
        for dayHourPath in glob(os.path.join(dayPath, "*")):
            splittedPath = dayHourPath.split("\\")
            day = splittedPath[-2]
            hour = splittedPath[-1]

            print(f"Starting {day} - {hour}")

            inputPath = os.path.join(dayHourPath, "*.json.bz2")
            outPath = os.path.join(args.out_path, f"{day}_{hour}")
            archived_tweets = spark.read.json(inputPath)
            out_tweets = archived_tweets\
                .filter((archived_tweets.lang == "tr")
                        & archived_tweets.user.verified
                        & archived_tweets.retweeted_status.isNull())

            out_tweets.withColumn("tweet_text", F.when(F.col("extended_tweet.full_text").isNull(), F.col("text"))
                                                 .otherwise(F.col("extended_tweet.full_text")))\
                .select("tweet_text").write.mode('overwrite').csv(outPath, header=True)

            print(f"Finished {day} - {hour}")


if __name__ == "__main__":
    exit(main())
