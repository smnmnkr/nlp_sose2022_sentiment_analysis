import pandas as pd

from classifier import Classifier


class Main:

    #
    #
    #  -------- __call__ -----------
    def __call__(self):

        # load data from csv files
        self.data: dict = {
            'train': pd.read_csv("../data/train.csv", sep=','),
            'eval': pd.read_csv("../data/eval.csv", sep=','),
        }

        # prepare train, eval data frame
        for _, data in self.data.items():
            Main._prepare_df(data)

        # load different classifier types
        self.classifier: dict = {
            'base': Classifier(),
            'generic_stop_words': Classifier("../data/stop_words.csv"),
            'customized_stop_words': Classifier("../data/stop_words_mod.csv"),
        }

        # iterate over each classifier
        for lbl_clsf, clsf in self.classifier.items():

            # fit to train data
            clsf.fit(self.data['train'])

            # predict train and eval set
            prediction: dict = {
                'train': clsf.predict(self.data['train']),
                'eval': clsf.predict(self.data['eval']),
            }

            # print results to console
            for lbl_data, data in prediction.items():
                valid: int = sum(data['prediction'] == data['gold'])
                print(f'CLASSIFIER: {lbl_clsf}, ACC({lbl_data})={valid / len(data)}')

    #
    #
    #  -------- _prepare_dfs -----------
    # (all inplace)
    @staticmethod
    def _prepare_df(data) -> None:
        # drop unnecessary columns
        data.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True, errors='ignore')

        # rename x and y labels
        data.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)

        # remove duplicates
        data.drop_duplicates(inplace=True)

        # tokenizer text convert to lowercase
        data['text'] = data['text'].str.lower().str.split()


if __name__ == "__main__":
    Main()()
