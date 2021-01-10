#!/usr/bin/env python3

from tr_troll_detect import forward_bert
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.tuning import TrainValidationSplitModel

def parse_args():
    parser = ArgumentParser(
        description="A toy REPL program to measure the "
        "'trollness' of Turkish tweets")
    parser.add_argument("model_path",
                        help="Path to model")
    return parser.parse_args()

def main():
    model_path = parse_args().model_path
    sc = SparkContext(master="local[*]", appName="tr-troll-detect")
    sc.setLogLevel("OFF")
    spark = SparkSession(sc)
    print(f"Loading model {model_path}")
    model = TrainValidationSplitModel.load(model_path)
    bert_name = "dbmdz/distilbert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name)
    while True:
        user_tweet = input("> ")
        embedding = forward_bert(user_tweet, tokenizer, bert_model)
        tweet_in = spark.createDataFrame([(user_tweet, embedding, 0)],
                                         schema=["tweet", "features", "label"])
        trol_prob = model.transform(tweet_in).collect()[0].probability[1]
        print(f"Probability of trolling : {trol_prob}")


if __name__ == "__main__":
    exit(main())
