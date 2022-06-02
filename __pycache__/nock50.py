import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('newsCorpora.csv', sep = "\t", names=["ID","TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"])
df = df[(df["PUBLISHER"] == "Reuters") | (df["PUBLISHER"] == "Huffington Post") | (df["PUBLISHER"] == "Businessweek") | (df["PUBLISHER"] == "Contactmusic.com") | (df["PUBLISHER"] == "Daily Mail")]
df1 = df[["TITLE", "CATEGORY"]]

train, valid = train_test_split(df, test_size=0.2)
valid, test = train_test_split(valid, test_size=0.5)

print(len(train))
print(len(valid))
print(len(test))