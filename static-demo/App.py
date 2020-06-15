import matplotlib.pyplot as plt
import pandas as pd
import SessionState
import streamlit as st

# Persists information until the page is reloaded
# This is done so that the datasets and the model
# are not loaded at every page interaction
ss = SessionState.get(
    loaded=False, portfolios=dict(), selected=None
)

if not ss.loaded:
    ss.portfolios['Portfolio 1'] = (pd.read_csv('data/leads_1.csv'), 555)
    ss.portfolios['Portfolio 2'] = (pd.read_csv('data/leads_2.csv'), 566)
    ss.portfolios['Portfolio 3'] = (pd.read_csv('data/leads_3.csv'), 265)


def main():
    st.title('Leads Recommender')
    st.sidebar.markdown("""
    # About

    This app was made using streamlit for the codenation Aceleradev Challenge 2020

    ## How does it work

    A Tf-Idf matrix was created using the dataset features

    With this matrix, the cosine similarity is calculated between the
    given ids and the dataset.

    Then, a score is attributed to each example on the dataset by
    the mean of the similarity scores

    This is a static demo, meaning that no computations are made.

    You can find the complete version of the app on [gitlab](%s)
    """ % 'https://gitlab.com/flycher/codenation-aceleradev-ds-2020')

    option = st.selectbox(
        'Select a portfolio to show the leads.',
        ['', 'Portfolio 1', 'Portfolio 2', 'Portfolio 3'],
    )

    topn = st.number_input(
        'Number of leads',
        min_value=1,
        value=100,
        format='%d'
    )
    find_leads = st.button('Find leads!')
    if option and find_leads:
        ss.selected = ss.portfolios[option][0]

    if ss.selected is not None:
        with st.spinner('Finding leads on the market dataset'):
            leads = ss.selected.head(topn)
            size = ss.portfolios[option][1]

        st.success(
            'Done! There were {0[0]} ids from {0[1]} in the database.'.format(
                (size, size)
            )
        )
        columns = leads.columns.to_list()
        show_cols = st.multiselect(
            'Select the information to show',
            columns,
            default=['id']
        )
        st.write(leads.loc[:, show_cols])

        st.map(leads)

        leads['empsetorcensitariofaixarendapopulacao'].hist(bins=50, color='#f63366')
        plt.title('Average Income from a sample of Residents')
        st.pyplot()

        st.write('Number of businesses by type.')
        st.write(leads['nm_divisao'].value_counts())


if __name__ == '__main__':
    main()
