import joblib
import matplotlib.pyplot as plt
import pandas as pd
import SessionState
import streamlit as st

# Persists information until the page is reloaded
# This is done so that the datasets and the model
# are not loaded at every page interaction
ss = SessionState.get(
    market=None, geo=None, leads=None, recommender=None
)

if ss.market is None:
    print('loading market')
    ss.market = pd.read_csv('../data/estaticos_market.zip', index_col='Unnamed: 0')

if ss.geo is None:
    print('loading geo')
    ss.geo = pd.read_csv('../data/geo.zip')

if ss.recommender is None:
    print('loading recommender')
    ss.recommender = joblib.load('../data/recommender.pkl')


def main():
    st.title('Leads Recommender')
    st.sidebar.markdown("""
    # About

    This app was made for the codenation Aceleradev Challenge 2020

    # How does it work

    A Tf-Idf matrix was created using the dataset features

    With this matrix, the cosine similarity is calculated between the
    given ids and the dataset.

    Then, a score is attributed to each example on the dataset by
    the mean of the similarity scores

    The source code can be found on [gitlab]()
    """)

    portfolio = None
    uploaded_file = st.file_uploader(
        'Upload a csv containing a column with the id of the businesses', type='csv'
    )
    if uploaded_file:
        portfolio = pd.read_csv(uploaded_file, usecols=['id'])

    topn = st.number_input(
        'Number of leads',
        min_value=1,
        value=100,
        format='%d'
    )
    find_leads = st.button('Find leads!')
    if portfolio is not None and find_leads:
        with st.spinner('Finding leads on the market dataset'):
            ss.leads = ss.recommender.predict(portfolio, topn)
            ss.leads = ss.leads.merge(ss.market, on='id')
            st.success(
                'Done! There were {0[0]} ids from {0[1]} in the database.'.format(
                    ss.recommender.matching
                )
            )

    if ss.leads is not None:
        leads = ss.leads.merge(ss.geo, on='id')

        columns = leads.columns.to_list()
        show_cols = st.multiselect(
            'Select the information to show',
            columns,
            default=['id']
        )
        st.write(leads.loc[:, show_cols])

        st.map(leads)

        plt.hist(leads['empsetorcensitariofaixarendapopulacao'], bins=50, color='#f63366')
        plt.title('Average Income from a sample of Residents')
        st.pyplot()

        st.write('Number of businesses by type.')
        st.write(leads['nm_divisao'].value_counts())


if __name__ == '__main__':
    main()
