#!/usr/bin/env python3
from transformers import AutoModel, AutoTokenizer
from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import Vectors
from argparse import ArgumentParser
from pathlib import Path

def forward_bert(tweet, bert_tokenizer, bert_model):
    """Convert a tweet to a 768 vector space"""
    tokenized = bert_tokenizer(tweet, return_tensors="pt")

    # Return vector of [CLS] token
    return Vectors.dense(
        bert_model(**tokenized).last_hidden_state[0, 0, :].tolist()
    )

def parse_args():
    parser = ArgumentParser(
        description="Train a model to detect Turkish Twitter Troll Users")
    parser.add_argument("troll_file",
                        help="A file where each line is a tweet"
                        " of a troll users")
    parser.add_argument("non_troll_file",
                        help="A file where each line is a tweet"
                        " of a non-troll users")
    parser.add_argument("out_path", help="Output path for model")
    parser.add_argument("-l", "--limit", default=0, type=int,
                        help="Limit size of troll and non-troll."
                        " 0 for unlimited")
    return parser.parse_args()

def truncate_tweets(tweets: RDD, limit: int) -> RDD:
    if not limit:
        return tweets

    return tweets.zipWithIndex().filter(
        lambda ti: ti[1] < limit
    ).map(
        lambda ti : ti[0]
    )


def main():
    args = parse_args()
    sc = SparkContext(master="local[*]", appName="tr-troll-detect")
    spark = SparkSession(sc)
    troll_tweets = truncate_tweets(sc.textFile(args.troll_file), args.limit)
    nontroll_tweets = truncate_tweets(sc.textFile(args.non_troll_file),
                                      args.limit)
    print(f"SPARK VERSION: {spark.version}")

    bert_name = "dbmdz/distilbert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name)

    troll_tweets = troll_tweets.map(
        lambda t: (t, forward_bert(t, tokenizer, bert_model), 1)
    )

    nontroll_tweets = nontroll_tweets.map(
        lambda t: (t, forward_bert(t, tokenizer, bert_model), 0)
    )

    tweets = spark.createDataFrame(troll_tweets.union(nontroll_tweets),
                                   schema=["tweet", "features", "label"])
    train_tweets, test_tweets = tweets.randomSplit([0.75, 0.25])
    # rf = RandomForestClassifier(seed=42)
    lr = LogisticRegression()
    pipeline = Pipeline(stages=[lr])

    param_grid_builder = ParamGridBuilder()
    param_grid_builder.addGrid(lr.regParam, [0.3, 0.1, 0.01, 0.001])
    param_grid_builder.addGrid(lr.maxIter, [50000, 100000])
    # param_grid_builder.addGrid(rf.impurity, ["gini", "entropy"])
    # param_grid_builder.addGrid(rf.maxBins, [16, 32, 64])
    # param_grid_builder.addGrid(rf.maxDepth, [4, 8, 16])
    # param_grid_builder.addGrid(rf.numTrees, [10, 20, 30]
    param_grid = param_grid_builder.build()

    evaluator = BinaryClassificationEvaluator()
    split = TrainValidationSplit(estimator=pipeline,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator, trainRatio=0.8)
    model = split.fit(train_tweets)
    train_acc = evaluator.evaluate(model.transform(train_tweets))
    print(f"Train accuracy: {train_acc}")

    test_preds = model.transform(test_tweets)
    test_preds.show()
    test_acc = evaluator.evaluate(test_preds)
    print(f"Test accuracy: {test_acc}")
    out_path = Path(args.out_path)

    # Never lose a trained model by blocking and asking to user
    while out_path.exists():
        ostr = str(out_path)
        out_path = Path(input(
            f"'{ostr}' already exists, please input another path: "))

    model.write().save(str(out_path))

if __name__ == "__main__":
   exit(main())
