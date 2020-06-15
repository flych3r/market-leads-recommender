import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class Recommender:
    """
    Class implementing a recommender system

    Attributes
    ----------
    vectorizer: TfidfVectorizer
        A Tf-Idf vectorizer used for creating a vocabulary

    item_ids: list
        Ids of the items to build a profile for the recommendation

    content: list
        List of strings containing the columns and values of the data
        from the ids as a string

    tfidf_matrix: sparse matrix
        Tf-Idf matrix with the result of the fitted vectorizer

    matching: tuple
        For prediction, saves the number of ids in the prediction found
        in the item_ids list and used in the prediction

    """

    def __init__(
        self,
        analyzer='word',
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        norm='l2'
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            norm=norm,
            dtype=np.float32
        )
        self.item_ids = None
        self.content = None
        self.tfidf_matrix = None
        self.matching = None

    def fit(self, data):
        """
        Preprocessess the data and fits the model

        Parameters
        ----------
        data: dataframe
            Dataframe used to fit the model
            This data is preprocessed and the ids and columns-values are saved
            in the item_ids and content attributes.

        """

        print('Preprocessing data')
        market = Preprocessor().preprocess_data(data)
        self.item_ids = market['id'].to_list()
        self.content = market['content'].to_list()
        print('Fitting model')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.content)

    def predict(self, data, topn=1000):
        """
        Uses the tfidx_matrix and cosine similarity to recommends the
        topn most similar values from the data

        Parameters
        ----------
        data: dataframe
            The data to build the profile and recommend items

        topn: int
            The number of recommendations to be made

        """

        profile = self._get_item_profiles(data['id'])
        similarity = self._cosine_similarity_batch(profile, chunk=64)
        similarity = similarity.mean(axis=0)
        recommendation = pd.DataFrame(data={
            'id': self.item_ids, 'similarity': similarity
        })
        recommendation = recommendation[~recommendation['id'].isin(data)]
        recommendation = recommendation.sort_values(by='similarity', ascending=False)
        return recommendation.head(topn)

    def _cosine_similarity_batch(self, profile, chunk=64):
        batches = (profile.shape[0] // chunk) + 1
        similarity = [
            cosine_similarity(
                profile[(chunk * batch): (chunk * (batch + 1))],
                self.tfidf_matrix
            )
            for batch in range(batches)
        ]
        return np.concatenate(similarity)

    def _get_item_profile(self, item_id):
        """
        Searches for the item_id in the item_ids attribute

        Parameters
        ----------
        item_id: string
            Id of the item

        Returns
        -------
        sparse matrix or None
            An element from the tfidf_matrix correspondint to the item
            or None if the item is not found

        """

        try:
            idx = self.item_ids.index(item_id)
            item_profile = self.tfidf_matrix[idx:idx + 1]
            return item_profile
        except ValueError:
            return None

    def _get_item_profiles(self, ids):
        """
        Gets the tfidf_matrix values from the ids.

        Parameters
        ----------
        ids: list
            Ids of the items to be built a profile

        Returns
        -------
        sparse matrix

        """

        item_profiles_list = [self._get_item_profile(x) for x in ids]
        item_profiles = [*filter(lambda x: x is not None, item_profiles_list)]
        self.matching = (len(item_profiles), len(item_profiles_list))
        item_profiles = vstack(item_profiles)
        return item_profiles

    @staticmethod
    def _hits(rec, lst):
        """
        Number of items recommended in the test set

        Parameters
        ----------
        rec: dataframe
            Recommendations
        lst: list
            Test set ids

        Returns
        -------
        float, int, int
            Proportion of items predicted in the test set
            Number of correct items predicted
            Size of the test set

        """

        hit = rec['id'].isin(lst).sum()
        size = len(lst)
        return hit / size, hit, size

    def evaluate_model(self, portfolios, test_size=0.3, random_state=42, topn=None):
        """
        Evaluates the model, computing the number of recommendations
        present in a test set

        Parameters
        ----------
        portfolio: list
            list of dataframes to test on

        test_size: float, optional, default 0.3
            Percentage of each portifolio used for testing

        random_state: float, optional, default 42
            Seed for the train-test separation

        topn: int, optional, default None
            Number of pretictions to be made. If None, the model will
            be evaluated predicting 10x the number of the test set

        """

        for i, port in enumerate(portfolios):
            x, y = train_test_split(port, test_size=0.3, random_state=42)
            y = y['id']
            if not topn:
                topn = len(y) * 10
            rec = self.predict(x, topn=topn)
            p_hits, n_hits, s_lst = self._hits(rec, y)
            print('Portifólio {}'.format(i + 1))
            print('({:.2f} %) {} hits from {} in {} predictions'.format(
                p_hits * 100, n_hits, s_lst, len(rec)
            ))
