from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import get_data

df = get_data.df
gdp = get_data.gdp


def split(data, train_or_cv):
    """
    split the data to x and y for cross validation or to x_train,y_train,x_test,y_test for training
    :param data: dataframe of features and target
    :param train_or_cv: either split to x, y or train, test
    :return: see description
    """
    y = data.subscriber
    x = data.drop('subscriber', axis=1)
    if train_or_cv == 'train':
        return train_test_split(x, y, test_size=0.25, random_state=27, shuffle=True,
                                stratify=y)
    elif train_or_cv == 'cv':
        return x, y
    else:
        raise ValueError('value has to be either "train" or "cv"')


def adding_data(data: pd.DataFrame, added_data, left_col, right_col) -> pd.DataFrame:
    """
    adding data from other sources to enrich our data
    :param data: data to enrich
    :param added_data: external data
    :param left_col: column name from our data to merge by
    :param right_col: column name from new data to merge by
    :return: dataframe of the merged dataframes
    """
    df_added = pd.merge(data, added_data, left_on=left_col, right_on=right_col)
    return df_added


def scaling(data):
    """
    scaling the data before training, scales only columns that aren't 0's and 1's.
    :param data: data to scale
    :return: scaled data
    """
    to_scale = [col for col in data.columns if data[col].max() > 1]
    mms = MinMaxScaler()
    scaled = mms.fit_transform(data[to_scale])
    scaled = pd.DataFrame(scaled, columns=to_scale, index=data.index)

    for col in scaled:
        data[col] = scaled[col]
    return data


def binarize_regex(data, col, regex):
    """
    replace regex with 0 and 1
    :param data: dataframe
    :param col: column name to transform
    :param regex: regex to binarize by
    :return: dataframe with replaced binarized column
    """
    column = data[col]
    return column.str.contains(regex)


def ohe(data, added_prefix, col_to_ohe):
    """
    replace categorical columns with one hot encoded dataframe
    :param data: dataframe with categorical columns
    :param added_prefix: what prefix to add to ohe columns
    :param col_to_ohe: what column to transform
    :return: dataframe with ohe instead of categorical data
    """
    return pd.get_dummies(data, prefix=added_prefix, columns=col_to_ohe, drop_first=True)


def create_features(data, method, cols=None, regex=None):
    """
    create features by averaging, summing and counting columns. The data is grouped by user
    :param data: dataframe
    :param method: mean, sum or count (every method of grouped dataframe is possible)
    :param cols: column to transform
    :param regex: allows to filter data by regex
    :return: dataframe with transformed columns
    """
    data = getattr(data.groupby('id_for_vendor'), method)()
    if cols is not None:
        data = data[cols]
    if regex is not None:
        data = data.filter(regex=regex)
    return data


def multiply_features(data, col_to_multiply):
    """
    multiply features by other features. Helper function for app feature features.
    :param data: dataframe
    :param col_to_multiply: which column to multiply with the app feature columns
    :return: dataframe with transformed columns
    """
    data = data.set_index('id_for_vendor')
    return data.filter(regex="feature.*").multiply(data[col_to_multiply], axis="index")


def basic_editing():
    """
    function to pipe some of our function in order to transform the data
    :return: dataframe
    """
    df_gdp = adding_data(df, gdp, 'country', 'Entity')
    df_gdp['device_type'] = binarize_regex(df_gdp, 'device', 'Ipad')
    df_transformed = ohe(df_gdp, ['feature'], ['feature_name'])
    return df_transformed


def piping(merge_type):
    """
    function to pipe some of our function in order to transform the data
    :param merge_type: allows to choose how to create "app features" features. Can be either 'additive', 'acceptance'
    or 'duration'.
    :return: transformed dataframe
    """
    df_transformed = basic_editing()
    accepted_duration = create_features(df_transformed, 'sum', ['accepted', 'usage_duration'])
    num_of_sessions = create_features(df_transformed, 'count', 'app_session_id')
    gdp_col = create_features(df_transformed, 'mean', 'gdp')
    device_type = create_features(df_transformed, 'mean', 'device_type')
    subscriber = create_features(df_transformed, 'mean', 'subscriber')

    if merge_type == 'additive':
        times_per_feature = create_features(df_transformed, 'sum', regex="feature.*")
        selected = times_per_feature
    elif merge_type == 'acceptance':
        app_features_accept = multiply_features(df_transformed, 'accepted')
        selected = create_features(app_features_accept, 'sum')
    elif merge_type == 'duration':
        app_features_duration = multiply_features(df_transformed, 'usage_duration')
        selected = create_features(app_features_duration, 'sum')
    else:
        raise ValueError('merge type has to be "additive", "acceptance" or "duration"')

    dfs = [accepted_duration, num_of_sessions, gdp_col, device_type, selected, subscriber]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['id_for_vendor'],
                                                    how='outer'), dfs)
    return df_merged


def train_mlm(data, cols, regex=None):
    cols = [col for col in data.columns if (col in cols) | (regex in col)]
    data = data[cols]

    def formula_from_cols():
        return 'subscriber' + ' ~ ' + ' + '.join([col for col in data.columns if not col == 'subscriber'])
    formula = formula_from_cols()
    md = smf.mixedlm(formula, data, groups=data["id_for_vendor"])
    mdf = md.fit(method=["lbfgs"])
    print(mdf.summary())


def plot_pivot(indices, cols, values='subscriber', agg_func='sum', cluster_unit='id_for_vendor'):
    """
    plot pivot tables. Can aggregate at multiple level with different aggregation functions (sum, mean, count)
    :param indices: column(s) to use as index for the pivot table
    :param cols: column(s) to use as columns for the pivot table
    :param values: what should be the values in each cell
    :param agg_func: how to aggregate the values in the cell; (sum, mean, count)
    :param cluster_unit: whether to group by users, sessions or none
    :return: plotly plot and pivot table dataframe
    """
    if agg_func == 'sum':
        table = pd.pivot_table(df, values=values, index=indices,
                               columns=cols, aggfunc=agg_func, fill_value=-0.1)
    elif agg_func == 'mean':
        table = pd.pivot_table(df, index=indices, values=values,
                               columns=cols, aggfunc=np.mean, fill_value=-0.1)
    else:
        table = pd.pivot_table(df, index=indices, values=values,
                               columns=cols, aggfunc=lambda x: len(x.unique()), fill_value=-0.1)
    if cluster_unit is not None:
        table = table[cluster_unit]
    fig = px.imshow(table, color_continuous_scale='viridis')
    return fig, table


def get_unique_values(cols):
    col = get_data.df[cols].unique()
    return col


def plot_hist(data, col):
    fig = px.histogram(data, x=col)
    return fig
