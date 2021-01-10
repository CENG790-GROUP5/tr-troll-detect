#!/usr/bin/env python3

from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.sql import SparkSession



def parse_args():
    parser = ArgumentParser(
        description="Extract data from troll csv")
    parser.add_argument("troll_file",
                        help="A file where each line is a tweet"
                        " of a troll users")
    parser.add_argument("out_path", help="Output path for model")
    return parser.parse_args()


def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="extract_troll")
    spark = SparkSession(sc)
    troll_tweets = spark.read.csv(args.troll_file, header=True, multiLine=True)
    troll_tweets.printSchema()
    out_tweets = troll_tweets.filter((troll_tweets.tweet_language == "tr") &
                                     (troll_tweets.is_retweet == "false"))
    out_tweets.select("tweet_text").write.csv(args.out_path, header=True)


if __name__ == "__main__":
    exit(main())
