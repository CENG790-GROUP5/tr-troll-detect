#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer
from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import Vectors
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import logging
import os
import unicodedata

# zemberek-python -> https://github.com/Loodos/zemberek-python
from zemberek import TurkishSentenceNormalizer, TurkishSentenceExtractor, TurkishMorphology

SPIECE_UNDERLINE = u"▁".encode("utf-8")
SparkContext.setSystemProperty('spark.executor.memory', '8g')

class TextNormalization:
    """Text normalization task
    """

    def __init__(self):
        """Constructor method
        """
        self.zemberek_morpholgy = TurkishMorphology.create_with_defaults()
        self.zemberek_normalizer = TurkishSentenceNormalizer(self.zemberek_morpholgy)
        self.zemberek_extractor = TurkishSentenceExtractor()

    def normalize(self,
                  text: str,
                  remove_space: bool = True,
                  do_lower_case: bool = True,
                  normalize_function: str = "NFKC",
                  is_turkish: bool = True,
                  use_zemberek: bool = True):
        """Preprocess text by removing extra space and normalizing via python-unicodedata library.

        :param str text: Text for normalization
        :param bool remove_space: Whether remove empty spaces or not (defaults to True)
        :param bool do_lower_case: Whether do lower case or not (defaults to True)
        :param str normalize_function: Unicodedata normalize function.
            Either "NFC", "NFKC", "NFD" or "NFKD". (defaults to "NFKC")
        :param bool is_turkish: Whether text is in Turkish or not (defaults to True)
        :param bool use_zemberek: Whether to use Zemberek-Python's normalizer. Always do lowercase inside (defaults to True)
        :return: Normalized text
        """
        outputs: str = text

        if remove_space:
            outputs = " ".join(outputs.strip().split())

        outputs = unicodedata.normalize(normalize_function, outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        if use_zemberek:
            sentences = self.zemberek_extractor.from_paragraph(outputs)
            normalized_sentences = []
            for sentence in sentences:
                normalized_sentences.append(self.zemberek_normalizer.normalize(sentence))
            outputs = "".join(normalized_sentences)

        if do_lower_case:
            if is_turkish:
                outputs = outputs.replace('\u0049', '\u0131')  # I -> ı
                outputs = outputs.replace('\u0130', '\u0069')  # İ -> i

            outputs = outputs.casefold()

        return outputs



def forward_bert(tweet, bert_tokenizer=None, bert_model=None):
    """Convert a tweet to a 768 vector space"""
    tokenized = bert_tokenizer(tweet, return_tensors="pt")

    # Return vector of [CLS] token
    return Vectors.dense(
        bert_model(**tokenized).last_hidden_state[0, 0, :].tolist()
    )

def forward_albert(tweet, bert_tokenizer=None, bert_model=None,
                   normalizer=None):
    try:
        norm_tweet = normalizer.normalize(tweet, do_lower_case=True,
                                          is_turkish=True)
    except:
        print(f"Error with {tweet}")
        norm_tweet = tweet

    return forward_bert(norm_tweet, bert_tokenizer, bert_model)


def distbert_setup():
    bert_name = "dbmdz/distilbert-base-turkish-cased"
    tkzr = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name)
    return partial(forward_bert, bert_tokenizer=tkzr, bert_model=model)

def bert_uncased_setup():
    bert_name = "dbmdz/bert-base-turkish-uncased"
    tkzr = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name)
    return partial(forward_bert, bert_tokenizer=tkzr, bert_model=model)

def albert_setup():
    bert_name = "loodos/albert-base-turkish-uncased"
    tkzr = AutoTokenizer.from_pretrained(bert_name, do_lower_case=False,
                                         keep_accents=True)
    model = AutoModel.from_pretrained(bert_name)
    normalizer = TextNormalization()
    return partial(forward_albert, bert_tokenizer=tkzr, bert_model=model,
                   normalizer=normalizer)


def parse_args():
    parser = ArgumentParser(
        description="Train a model to detect Turkish Twitter Troll Users")
    parser.add_argument("troll_file",
                        help="A file where each line is a tweet"
                        " of a troll users")
    parser.add_argument("non_troll_file",
                        help="A file where each line is a tweet"
                        " of a non-troll users")
    parser.add_argument("out_model_path", help="Output path for model")
    parser.add_argument("out_roc_path", help="Output summary path")
    parser.add_argument("-l", "--limit", default=0, type=int,
                        help="Limit size of troll and non-troll."
                        " 0 for unlimited")
    parser.add_argument("-u", "--uncased", action="store_true",
                        help="Use uncased BERT")
    return parser.parse_args()

def load_tweets(tweet_file: str, spark: SparkSession, limit: int) -> DataFrame:
    tweets = spark.read.csv(tweet_file, header=True, multiLine=True)
    tweets = tweets.where(tweets.tweet_text.isNotNull())
    if not limit:
        return tweets

    return tweets.limit(limit)


def trainable_data(tweets, label, is_uncased):
    def mapp(partition):
        fwd_bert = bert_uncased_setup() if is_uncased else distbert_setup()
        for p in partition:
            yield (p[0], fwd_bert(p[0]), p[1])

    return tweets.rdd.map(lambda t: (t.tweet_text, label)).mapPartitions(mapp)


def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="tr-troll-detect")
    spark = SparkSession(sc)
    troll_tweets = load_tweets(args.troll_file, spark, args.limit).cache()
    non_troll_tweets = load_tweets(args.non_troll_file, spark,
                                   args.limit).cache()
    troll_count = troll_tweets.count()
    non_troll_count = non_troll_tweets.count()
    print(f"Troll count: {troll_count}, Non-troll count: {non_troll_count}")
    if troll_count < non_troll_count:
        min_count = troll_count
    else:
        min_count = non_troll_count

    print(f"Truncating to {min_count} samples")
    troll_tweets = troll_tweets.limit(min_count)
    non_troll_tweets = non_troll_tweets.limit(min_count)
    print(f"SPARK VERSION: {spark.version}")
    troll_tweets = trainable_data(troll_tweets, 1, args.uncased)
    non_troll_tweets = trainable_data(non_troll_tweets, 0, args.uncased)
    tweets = spark.createDataFrame(troll_tweets.union(non_troll_tweets),
                                   schema=["tweet", "features", "label"]).cache()

    train_tweets, test_tweets = tweets.randomSplit([0.75, 0.25], seed=42)
    print("Train tweets:")
    train_tweets.show()
    print("Test tweets")
    test_tweets.show()

    lr = LogisticRegression()
    pipeline = Pipeline(stages=[lr])
    param_grid_builder = ParamGridBuilder()
    param_grid_builder.addGrid(lr.regParam, [0.1, 0.01, 0.001])
    param_grid_builder.addGrid(lr.maxIter, [50000, 100000])
    param_grid_builder.addGrid(lr.threshold, [0.5, 0.7, 0.9])
    param_grid = param_grid_builder.build()
    evaluator = BinaryClassificationEvaluator()
    split = TrainValidationSplit(evaluator=evaluator,
        estimator=pipeline, estimatorParamMaps=param_grid, trainRatio=0.8)
    model = split.fit(train_tweets)
    lr_model = model.bestModel.stages[-1]

    print(f"Best params: regParam: {lr_model.getRegParam()},"
          f" maxIter: {lr_model.getMaxIter()},"
          f" threshold: {lr_model.getThreshold()}")
    summary_train = lr_model.evaluate(train_tweets)
    print(f"Train accuracy: {summary_train.accuracy}")
    summary_test = lr_model.evaluate(test_tweets)
    print(f"Test accuracy: {summary_test.accuracy}")
    print(f"Test fMeasure: {summary_test.weightedFMeasure()}")
    model.write().overwrite().save(args.out_model_path)
    roc = summary_test.roc.rdd.collect()
    roc_str_out = "".join([f"({round(r.FPR, 4)},{round(r.TPR, 4)})" for r in roc])
    with open(args.out_roc_path, "w") as rs:
        rs.write(roc_str_out)

    return 0

if __name__ == "__main__":
   exit(main())
