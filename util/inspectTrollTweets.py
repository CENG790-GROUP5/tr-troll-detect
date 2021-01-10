from glob import glob
import os
import io

VERIFIED_TWEETS = "trollTweets.txt"
OUTPUT_PATH = "../output_troll"
tweetsCount = 0

with io.open(VERIFIED_TWEETS, mode='w+', encoding='utf-8') as o:
	successFilePath = os.path.join(OUTPUT_PATH, "_SUCCESS")
	if os.path.exists(successFilePath):
		filePath = os.path.join(OUTPUT_PATH, "part-*")
		for file in glob(filePath):
			with io.open(file, mode='r', encoding='utf-8') as f:
				for line in f.readlines():
					o.write(line.rstrip("\n\r"))
					o.write("\n")
					tweetsCount += 1

print(f"Troll Tweets Count: {tweetsCount}")