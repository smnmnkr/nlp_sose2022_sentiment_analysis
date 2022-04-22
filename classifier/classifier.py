from collections import Counter

import pandas as pd


class Classifier:

    #
    #
    #  -------- __init__ -----------
    def __init__(
            self,
            stop_words_path: str = None,
            use_most_common: int = None,
            remove_shared: bool = False
    ) -> None:
        self.use_most_common = use_most_common
        self.remove_shared = remove_shared

        # init empty stop words list
        self.stop_words: list = []

        # load data from stop words csv if path is given by user
        if stop_words_path:
            self.stop_words = list(pd.read_csv(stop_words_path, sep=','))

            # split/strip stop words and convert into list
            self.stop_words = list(
                word.replace('\"', '').strip()
                for word in self.stop_words
            )

        # init empty polarity words dictionary
        self.polarity_words: dict = {}

    #
    #
    #
    #  -------- fit -----------
    def fit(self, data: pd.DataFrame) -> None:

        # iterate over each polarity
        for label, group in data.groupby('label'):

            # count most common words (absolute frequencies)
            self.polarity_words[label]: Counter = Counter(list(group['text'].explode()))

            # filter stop words
            for word in self.stop_words:
                if word in self.polarity_words[label]:
                    del self.polarity_words[label][word]

            # (optional) use only n most common words
            if self.use_most_common:
                self.polarity_words[label] = dict(self.polarity_words[label].most_common(self.use_most_common))

            # calculate relative word frequencies
            polarity_words_sum: int = sum(val for _, val in self.polarity_words[label].items())
            for lbl, val in self.polarity_words[label].items():
                self.polarity_words[label][lbl]: float = val / polarity_words_sum

        # (optional) removed shared polarity words
        if self.remove_shared:

            # collect sets of all words for each polarity
            all_words: list = [
                set(lbl for lbl, _ in dct.items())
                for _, dct in self.polarity_words.items()
            ]

            # remove each word which is in the intersection of all polarity words
            for word in set.intersection(*map(set, all_words)):

                # loop through each polarity count
                for _, dictionary in self.polarity_words.items():
                    if word in dictionary:
                        del dictionary[word]

    #
    #
    #
    # -------- predict -----------
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:

        # create empty predictions dataframe
        predictions: pd.DataFrame = pd.DataFrame()

        # calculate a score for each polarity
        for label, counter in self.polarity_words.items():
            predictions[label] = data["text"].apply(
                lambda x: sum([counter.get(w, 0) for w in x])
            )

        # get label of polarity with the highest prediction value, add gold labels
        predictions['prediction'] = predictions[[label for label, _ in self.polarity_words.items()]].idxmax(axis=1)
        predictions['gold'] = data['label']

        return predictions
