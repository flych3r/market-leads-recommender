from numpy import inf
from pandas import cut


class Preprocessor:
    """
    Class responsible for preprocessing data for the Recommender model

    Attributes
    ----------
    data: dataframe
        The dataframe to be processed

    categoric_cols: list
        The categorial columns on the dataframe

    numeric_cols: list
        The numeric columns on the dataframe

    """

    def __init__(self):
        self.data = None
        self.categoric_cols = None
        self.numeric_cols = None

    def preprocess_data(self, data, thresh=0.5):
        """
        Processess the dataframe

        Parameters
        ----------
        data: dataframe
            The dataframe to be processed

        thresh: float
            Threshold for dropping columns with nan values

        Returns
        -------
        dataframe
            A dataframe with the ids and content of the data

        Examples
        --------
        >>> print(df)
        id C1 C2
        1  a  33
        2  b  66
        3  c  99
        >>> print(preprocess_data(df))
        id  content
        1   C1_a C2_33
        2   C1_b C2_66
        3   C1_c C2_99

        """
        self.data = data.copy()
        self._remove_nan(thresh)
        self._remove_related()
        self._process_categoric()
        self._process_numeric()
        self._normalize_columns()
        self._create_content()
        return self.data[['id', 'content']]

    def _remove_nan(self, thresh=0.5):
        """
        Removes the columns with more than thresh nan values

        Parameters
        ----------
        thresh: float
            Threshold for dropping columns with nan values

        """

        missing = (self.data.isna().mean() > thresh)
        self.data = self.data.loc[:, ~missing]

    def _remove_related(self):
        """
        Removes columns that contain unnecessary information or information
        closely related to other columns

        """

        to_remove = [
            'fl_telefone', 'fl_email', 'de_saude_rescencia',
            'nu_meses_rescencia', 'idade_empresa_anos', 'dt_situacao',
            'vl_total_veiculos_pesados_grupo', 'vl_total_veiculos_leves_grupo',
            'idade_maxima_socios', 'idade_minima_socios',
            'vl_faturamento_estimado_aux', 'vl_faturamento_estimado_grupo_aux'
        ]
        self.data.drop(columns=to_remove, inplace=True)

    def _process_categoric(self):
        """
        Processess the categoric columns of the dataframe
        Merges the columns related with legal names and fills
        missing values with 'SEM INFORMACAO'

        """

        self.categoric_cols = self.data.drop(columns='id')
        self.categoric_cols = self.categoric_cols.select_dtypes(include=[object, bool])
        self.categoric_cols = self.categoric_cols.columns.to_list()
        self.data['fl_rm'].replace({'SIM': True, 'NAO': False}, inplace=True)
        nm_legal = ['fl_me', 'fl_sa', 'fl_epp', 'fl_mei', 'fl_ltda']
        self.data['nm_legal'] = self.data[nm_legal].idxmax(1)
        self.data.drop(columns=nm_legal, inplace=True)
        for c in nm_legal:
            self.categoric_cols.remove(c)
        self.categoric_cols.append('nm_legal')
        self.data[self.categoric_cols] = self.data[self.categoric_cols].fillna(
            'SEM INFORMACAO'
        ).astype(str)

    def _process_numeric(self):
        """
        Processess the numeric columns of the dataframe
        Fills the missing values with -1 and buckets the values

        """

        self.numeric_cols = self.data.drop(columns='id')
        self.numeric_cols = self.numeric_cols.select_dtypes(exclude=[object, bool])
        self.numeric_cols = self.numeric_cols.columns.to_list()

        self.data[self.numeric_cols] = self.data[self.numeric_cols].fillna(-1)
        self.data[self.numeric_cols] = self.data[self.numeric_cols].astype(int)
        self.data['empsetorcensitariofaixarendapopulacao'] = cut(
            self.data['empsetorcensitariofaixarendapopulacao'],
            [-2, 1, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 10000, inf]
        )

        self.data['qt_socios'] = cut(
            self.data['qt_socios'],
            [-2, 0, 1, 5, 15, 50, 100, inf]
        )
        self.data['qt_socios_pf'] = cut(
            self.data['qt_socios_pf'],
            [-2, 0, 1, 5, 15, 50, 100, inf]
        )
        self.data['qt_socios_pj'] = cut(
            self.data['qt_socios_pj'],
            [-2, 0, 1, inf]
        )
        self.data['qt_socios_st_regular'] = cut(
            self.data['qt_socios_st_regular'],
            [-2, 0, 1, 5, 15, 50, 100, inf]
        )

        self.data['idade_media_socios'] = cut(
            self.data['idade_media_socios'],
            [-2, 0, 22, 45, 65, inf]
        )

        self.data['qt_filiais'] = cut(
            self.data['qt_filiais'],
            [-2, 0, 5, 10, 50, 100, 1000, inf]
        )

        self.data[self.numeric_cols] = self.data[self.numeric_cols].astype(str)

    def _normalize_columns(self):
        """
        Replaces all num alphanumeric characters with '_'

        """

        for col in self.categoric_cols + self.numeric_cols:
            self.data[col] = self.data[col].str.replace(r'\W', '_')

    @staticmethod
    def _columns_to_string(x):
        """
        Transforms a row's columns and its values in a string

        """

        return ' '.join(['%s_%s' % a for a in list(zip(x.index, x.values))])

    def _create_content(self):
        """
        Creates the content column, transforming a row into a string
        containing the columns and its values.

        """

        self.data['content'] = self.data.drop(columns='id').apply(
            lambda x: self._columns_to_string(x), axis=1
        )
