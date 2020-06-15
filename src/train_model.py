import joblib
import pandas as pd
from recommender import Recommender

market = pd.read_csv('../data/estaticos_market.zip', index_col='Unnamed: 0')

rec = Recommender()
rec.fit(market)

print('Saving model')
joblib.dump(rec, '../data/recommender.pkl', compress=9)
