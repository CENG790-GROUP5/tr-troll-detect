from glob import glob
import os
import io

VERIFIED_TWEETS = "verifiedTweets.txt"
OUTPUT_PATH = "../output_archive/*"
hourCount = 0
tweetsCount = 0
verifiedTweetsCounts = 0

with io.open(VERIFIED_TWEETS, mode='w+', encoding='utf-8') as o:
	for folderPath in glob(OUTPUT_PATH):
		splittedPath = folderPath.split("\\")
		dayHour = splittedPath[-1].split("_")
		day = dayHour[0]
		hour = dayHour[1]
		
		successFilePath = os.path.join(folderPath, "_SUCCESS")
		if os.path.exists(successFilePath):
			filePath = os.path.join(folderPath, "part-*")
			for file in glob(filePath):
				with io.open(file, mode='r', encoding='utf-8') as f:
					for line in f.readlines():
						verified = line.split(",")[-1]
						if "true" in verified:
							verifiedTweetsCounts += 1
							o.write(line.split(",")[0])
							o.write("\n")
						tweetsCount += 1
			hourCount += 1

print(f"Verified Tweets Count: {verifiedTweetsCounts}, Tweets Count: {tweetsCount}, Total Hours: {hourCount}, Total Days: {int(hourCount/24)}")