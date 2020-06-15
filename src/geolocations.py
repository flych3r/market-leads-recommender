import pandas as pd
from geopy.geocoders import Nominatim
from tqdm import tqdm

states = """
Acre: AC
Alagoas: AL
Amapá: AP
Amazonas: AM
Bahia: BA
Ceará: CE
Distrito Federal: DF
Espírito Santo: ES
Goiás: GO
Maranhão: MA
Mato Grosso: MT
Mato Grosso do Sul: MS
Minas Gerais: MG
Pará: PA
Paraíba : PB
Paraná: PR
Pernambuco: PE
Piauí: PI
Rio de Janeiro: RJ
Rio Grande do Norte: RN
Rio Grande do Sul : RS
Rondônia: RO
Roraima: RR
Santa Catarina : SC
São Paulo : SP
Sergipe: SE
Tocantins: TO
"""

states = states.split('\n')
states = [st.split(':') for st in states]

states_dict = dict()
for v, k in states[1:-1]:
    states_dict[k.strip()] = v.strip()

geo = pd.read_csv(
    '../data/estaticos_market.zip', index_col='Unnamed: 0',
    usecols=['Unnamed: 0', 'id', 'nm_micro_regiao', 'nm_meso_regiao', 'sg_uf']
)
geo['sg_uf'].replace(states_dict, inplace=True)


def get_address(x):
    """
    Transforms the columns refering to locations into a single address str columns
    """
    return '{}, {}, {}'.format(
        x['sg_uf'].title(),
        x['nm_meso_regiao'].title(),
        x['nm_micro_regiao'].title()
    )


geo.fillna('', inplace=True)
geo['address'] = geo.apply(get_address, axis=1)


def get_geolocation(address):
    """
    Tries to find the geolocation, reducing the
    address precision if doesn't finds one
    """
    location = geolocator.geocode(address)
    while not location and address:
        address = ','.join(address.split(',')[:-1])
        location = geolocator.geocode(address)
    if not address:
        return None
    return (location.latitude, location.longitude)


geolocator = Nominatim(user_agent='lead_recommender')
locations = dict()
for address in tqdm(geo['address'].unique(), desc='Finding geolocations'):
    locations[address] = get_geolocation(address)

geo['lat'], geo['lon'] = zip(*geo['address'].map(locations))
compression_opts = {
    'method': 'zip',
    'archive_name': 'geo.csv'
}
geo[['id', 'lat', 'lon']].to_csv(
    '../data/geo.zip',
    index=False,
    compression=compression_opts
)
