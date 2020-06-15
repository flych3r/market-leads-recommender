# About

This app was made using streamlit for the codenation Aceleradev Challenge 2020

It uses a dataset of businesses information to recommend similar businesses based on a portfolio of existing clients

## How does it work

A Tf-Idf matrix was created using the dataset features

With this matrix, the cosine similarity is calculated between the
given ids and the dataset.

Then, a score is attributed to each example on the dataset by
the mean of the similarity scores

## Running the app

Create a virtual environment and install the necessary libraries from `requirements.txt`

Download the data using `misc/download.sh`

Go to the `src` directoty

Train the model executing `src/train_model.py`

Create the geolocations executins `src/geolocations.py`

To start the app run `streamlit run app.py`

## Exploration

In the notebook folder the is a jupyter notebook exploring how te model mas build.

## Data

- `estaticos_market.zip` &rarr; Market data
- `estaticos_portifolio{1, 2, 3}.csv` &rarr; Test sets
- `geo.zip` &rarr; Geolocation of each address from `estaticos_market.csv`
- `recommender.pkl` &rarr; Trained Recommender model
- `features_dictionary.pdf` &rarr; Description of the features
- `download.sh` &rarr; Bash script to download the csv data
- `links.txt` &rarr; File containing the urls of the files to download
- `README.md` &rarr; Description of the challenge

## Scripts

- `app.py` &rarr; Streamlit app file
- `geolocations.py` &rarr; Script created to extract the geolocation from the location on the `estaticos_market.csv` file
- `preprocessor.py` &rarr; Class responsible for preprocessing the `estaticos_market.csv` data to be used in the `Recommender` class
- `recommender.py` &rarr; Class implementing a recommendation system based on text simiarity using tf-idf and cosine distance
- `SessionState.py` &rarr; Class used for persisting user session data on streamlit
- `train_model.py` &rarr; Script user to train the recommender modelr model

### Miscelaneous files

- `download.sh` &rarr;  Downloads the data
- `setup.sh` &rarr; Configures streamlit for deployr deploy
