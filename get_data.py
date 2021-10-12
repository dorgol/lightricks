import pandas as pd

source_dir = 'https://raw.githubusercontent.com/dorgol/lightricks/main/'

df = pd.read_csv(f'{source_dir}/Subscripiton_Prediction.csv')
df[['install_date', 'device_timestamp', 'subscription_date']] = \
    df[['install_date', 'device_timestamp', 'subscription_date']].apply(pd.to_datetime)

df = df.drop(df.columns[0], axis=1)

df['device'] = df['device'].str.replace('UIDeviceKind', '', regex=True)
df['feature_name'] = df['feature_name'].str.replace(' ', '', regex=True)
gdp = pd.read_csv(f'{source_dir}/gdp.csv')
gdp = gdp.loc[gdp['Year'] == 2019]


df_cluster = pd.read_csv(f'{source_dir}/Clustering_Data.csv')
df_cluster = df_cluster.fillna(0)
