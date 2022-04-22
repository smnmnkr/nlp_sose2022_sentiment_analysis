import pandas as pd

from classifier import Classifier


class Main:

    #
    #
    #  -------- __call__ -----------
    def __call__(self, config: dict) -> None:

        # load data from csv files
        self.data: dict = {
            'train': pd.read_csv(config['data']['train_path'], sep=','),
            'eval': pd.read_csv(config['data']['eval_path'], sep=','),
        }

        # prepare train, eval data frame
        for _, data in self.data.items():
            Main._prepare_df(data)

        # load different classifier types
        self.classifier: dict = {
            label: Classifier(**cfg)
            for label, cfg in config['classifier'].items()
        }

        # iterate over each classifier
        for classifier_label, classifier in self.classifier.items():

            # fit to train data
            classifier.fit(self.data['train'])

            # predict train and eval set
            prediction: dict = {
                'train': classifier.predict(self.data['train']),
                'eval': classifier.predict(self.data['eval']),
            }

            # print results to console
            for data_label, data in prediction.items():
                # count valid predictions, print to console
                valid: int = sum(pd.Series(data['prediction'] == data['gold']))
                print(f'CLASSIFIER: {classifier_label:24}\t ACC({data_label})\t= {valid / len(data):.4f}')

            print()

    #
    #
    #  -------- _prepare_dfs -----------
    # (all inplace)
    @staticmethod
    def _prepare_df(data: pd.DataFrame) -> None:

        # drop unnecessary columns
        data.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True, errors='ignore')

        # rename x and y labels
        data.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)

        # remove duplicates
        data.drop_duplicates(inplace=True)

        # tokenizer text convert to lowercase
        data['text'] = data['text'].str.lower().str.split()


if __name__ == "__main__":
    CONFIG: dict = {
        'data': {
            'train_path': './data/train.csv',
            'eval_path': './data/eval.csv',
        },
        'classifier': {
            'base': {},
            'stopwords': {
                'stop_words_path': './data/stop_words.csv',
            },
            'stopwords+onlyCommon1000': {
                'stop_words_path': './data/stop_words.csv',
                'use_most_common': 1000,
            },
            'stopwords+sharedRemoved': {
                'stop_words_path': './data/stop_words.csv',
                'remove_shared': True,
            }
        }
    }

    Main()(CONFIG)
