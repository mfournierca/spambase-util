from urllib import urlretrieve
from os.path import join
from zipfile import ZipFile
from pandas import read_csv
from sklearn.cross_validation import train_test_split


class Spambase(object):

    def __init__(self, root):
        self.source_url = ("https://archive.ics.uci.edu/ml/"
                           "machine-learning-databases/"
                           "spambase/spambase.zip")
        self.root = root
        self.archive = join(self.root, "spambase.zip")
        self.data = join(self.root, "spambase/")

    def download(self):
        print("downloading data set ...")
        urlretrieve(self.source_url, self.archive)
        print("download complete, unzipping data ...")
        with ZipFile(self.archive) as z:
            z.extractall(self.data)

    def load(self, data_file="spambase.data", names_file="spambase.names"):
        """Return a data frame containing all the spambase data"""
        names = []
        with open(join(self.data, names_file), "r") as f:
            for i, e in enumerate(f):
                if 32 < i < 90:
                    names.append(e[:e.index(":")])
        names.append("spam")
        df = read_csv(join(self.data, data_file), header=None, names=names)
        return df

    def split(self, df, test_ratio=0.33, random_seed=1):
        """Return xtrain, xtest, ytrain, ytest"""
        x = df[df.columns.difference(["spam"])]
        y = df["spam"]
        xtrain, xtest, ytrain, ytest = train_test_split(
            x, y, test_size=test_ratio, random_state=random_seed)
        return xtrain, xtest, ytrain, ytest

    def get(self):
        return self.split(self.load())
