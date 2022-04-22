from collections import Counter

import pandas as pd


class Classifier:

    #
    #
    #  -------- __init__ -----------
    def __init__(self, stop_words_file: str = None):

        self.stop_words: list = []

        # load data from stopwords csv if path is given by user
        if stop_words_file:
            self.stop_words = list(pd.read_csv(stop_words_file, sep=','))

            # filter and strip stopwords
            self.stop_words = list(word.replace('\"', '').strip() for word in self.stop_words)

        # create empty polarity words holder
        self.polarity_words: dict = {}

    #
    #
    #
    #  -------- fit -----------
    def fit(self, data: pd.DataFrame):

        # iterate over each polarity
        for label, group in data.groupby('label'):

            # count most common words
            self.polarity_words[label]: Counter = Counter(list(group['text'].explode()))

            # filter stopwords
            for word in self.stop_words:
                if word in self.polarity_words[label]:
                    del self.polarity_words[label][word]

    #
    #
    #
    # -------- predict -----------
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:

        # create empty predictions dataframe
        predictions: pd.DataFrame = pd.DataFrame()

        # calculate a score for each polarity
        for label, counter in self.polarity_words.items():
            predictions[label] = data["text"].apply(lambda x: sum([counter.get(w, 0) for w in x]))

        # get label of polarity with the highest prediction value, add gold labels
        predictions['prediction'] = predictions[[label for label, _ in self.polarity_words.items()]].idxmax(axis=1)
        predictions['gold'] = data['label']

        return predictions






