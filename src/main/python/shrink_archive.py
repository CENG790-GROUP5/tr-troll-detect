#!/usr/bin/env python3

from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from glob import glob
import os
from datetime import datetime


SparkContext.setSystemProperty('spark.executor.memory', '10g')

def parse_args():
    parser = ArgumentParser(description="Shrink archived Twitter dataset")
    parser.add_argument("archive_path", help="Root folder for archived tweets")
    parser.add_argument("output_path", help="Output path for shrunk and archived tweets")
    return parser.parse_args()

def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="shrink_archive")
    spark = SparkSession(sc)

    start_time = datetime.now()

    for dayPath in glob(os.path.join(args.archive_path, "*")):
        for dayHourPath in glob(os.path.join(dayPath, "*")):
            hour = dayHourPath.split("\\")[-1]
            day = dayPath.split("\\")[-1]

            input_path = os.path.join(dayHourPath, "*.json.bz2")
            output_path = os.path.join(args.output_path, f"{day}_{hour}")

            if len(glob(input_path)) > 0:
                print(f"### Starting {day} - {hour} ###")

                archived_tweets = spark.read.json(input_path)
                filtered_archived_tweets = archived_tweets \
                    .filter(F.col("lang") == "tr") \
                    .filter(F.col("retweeted_status").isNull())
                filtered_archived_tweets.write.mode('overwrite').json(output_path)

                print(f"### Finished {day} - {hour} ###")

    finish_time = datetime.now()
    print(f"Finish Time ({finish_time.strftime('%H:%M:%S')}) - Start Time ({start_time.strftime('%H:%M:%S')}) = "
          f"{str(finish_time - start_time)}")

if __name__ == "__main__":
    exit(main())
