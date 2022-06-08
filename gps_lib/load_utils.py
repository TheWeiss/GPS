import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV


def get_filtered_data(
    data = 'tot_filtered_data.csv', 
    features = 'final_features',
    ASR_data = 'filtered_ASR_data.csv', 
    species_sep = True, 
    antibiotic_sep=True,
    species_filter_index=0, 
    antibiotic_index=0,
    naive=True, 
    task='regression', 
    strip_range_train=False,
    distance_range_train=False,
    range_moved=5,
    reg_stratified=True,
):
    data = pd.read_csv('../resources/'+data)
    with open("../resources/"+ features, "rb") as fp:
        features = pickle.load(fp)
    ASR_data = pd.read_csv('../resources/'+ASR_data)
    
    species2merge = data[['biosample_id', 'species_fam']]
    filtered_ASR = ASR_data.drop('species_fam', axis=1).merge(species2merge, on='biosample_id')
    filtered_ASR.set_index('biosample_id', inplace=True)
    filtered_ASR = filtered_ASR[filtered_ASR['units']=='mg/L']
    filtered_ASR = filtered_ASR[filtered_ASR['ast_standard']=='CLSI']
    filtered_ASR = filtered_ASR[filtered_ASR['species_fam']!='senterica']
    filtered_ASR = filtered_ASR[filtered_ASR['species_fam']!='spneumoniae']
    data.set_index('biosample_id', inplace=True)
    data.drop(['Unnamed: 0', 'species_fam', 'run_id'], axis=1, inplace=True)
            
    if species_sep:
        species = filtered_ASR['species_fam'].value_counts().reset_index()['index'].iloc[species_filter_index]
        filtered_ASR = filtered_ASR[filtered_ASR['species_fam'] == species]
    else:
        species = None
        
    if antibiotic_sep:
        anti_list = filtered_ASR['antibiotic_name'].value_counts().index.values
        label = anti_list[antibiotic_index]
        filtered_ASR = filtered_ASR[filtered_ASR['antibiotic_name'] == label]
    else:
        label = 'measurement'
    
    range_ASR = filtered_ASR[filtered_ASR['measurement_sign']!='=']        
    exact_ASR = filtered_ASR[filtered_ASR['measurement_sign']=='=']

    
    range_data_values = data.loc[range_ASR.index][label]
    range_labels = pd.DataFrame({
        'values':[],
        'direction': [],
    })
    range_labels['values'] = range_data_values.apply(lambda x: float(x.split(' ')[1]))
    range_labels['direction'] = range_data_values.apply(lambda x: x.split(' ')[0].replace('=', ''))
    
    exact_y = data.loc[exact_ASR.index][label]
    if task == 'regression':
        exact_y = exact_y.apply(lambda x: float(x.split(' ')[1]))
    elif task == 'classification':
        exact_y = exact_y.apply(lambda x: str(x.split(' ')[1]))
        
    exact_X = data.loc[exact_ASR.index][features]
    exact_X.dropna(axis=1, how='all', inplace=True)
    exact_X.fillna(0, inplace=True) 
    exact = exact_X.merge(exact_y, left_index=True, right_index=True)
    
    if not naive:
        range_y = range_data_values
        if task == 'regression':
            if strip_range_train:
                range_y = range_labels['values']
            elif distance_range_train:
                signs = range_labels['direction']
                range_y = range_labels['values']
                range_y = range_y.mask(signs == '>', range_y + range_moved)
                range_y = range_y.mask(signs == '<', range_y - range_moved)
            else:
                print('regression not in the naive approach is not implemented yet.') 
        range_y.name = label
    else:
        range_y = pd.DataFrame({})
        
    train_features = exact_X.columns.values
    range_X = data.loc[range_ASR.index][train_features]
    range_X.fillna(0, inplace=True)
    range_X = range_X.reset_index().set_index('biosample_id')
    range_data = range_X.merge(range_y, left_index=True, right_index=True)
    
    range_train_ids = []
    range_test_ids = []
    if reg_stratified:
        exact_train_ids = []
        exact_test_ids = []
        for y_val in set(exact_y.values):    
            sub_value_id = list(exact_y[exact_y == y_val].index)
            if len(sub_value_id) > 1:
                exact_train_id, exact_test_id = train_test_split(sub_value_id, test_size=0.2, random_state=42)
                exact_train_ids = exact_train_ids + exact_train_id
                exact_test_ids = exact_test_ids + exact_test_id
            else:
                exact_train_ids = exact_train_ids + sub_value_id
        if not naive:
            for y_val in set(range_y.values):    
                sub_value_id = list(range_y[range_y == y_val].index)
                if len(sub_value_id) > 1:
                    range_train_id, range_test_id = train_test_split(sub_value_id, test_size=0.2, random_state=42)
                    range_train_ids = range_train_ids + range_train_id
                    range_test_ids = range_test_ids + range_test_id
                else:
                    range_train_id = range_train_id + sub_value_id
    else:
        exact_train_ids, exact_test_ids = train_test_split(list(exact_y.index), test_size=0.2, random_state=42)
        if not naive:
            range_train_ids, range_test_ids = train_test_split(list(range_y.index), test_size=0.2, random_state=42)

    train = pd.concat([exact.loc[exact_train_ids,], range_data.loc[range_train_ids,]])
    test = pd.concat([exact.loc[exact_test_ids,], range_data.loc[range_test_ids,]])
    
    return train, test, range_X, range_labels, list(train_features), label, species


def get_filtered_data(
        data='tot_filtered_data.csv',
        features='final_features',
        ASR_data='filtered_ASR_data.csv',
        species_sep=True,
        antibiotic_sep=True,
        species_filter_index=0,
        antibiotic_index=0,
        naive=True,
        task='regression',
        strip_range_train=False,
        distance_range_train=False,
        range_moved=5,
        reg_stratified=True,
        cv_num=3,
):
    data = pd.read_csv('../resources/' + data)
    with open("../resources/" + features, "rb") as fp:
        features = pickle.load(fp)
    ASR_data = pd.read_csv('../resources/' + ASR_data)

    species2merge = data[['biosample_id', 'species_fam']]
    filtered_ASR = ASR_data.drop('species_fam', axis=1).merge(species2merge, on='biosample_id')
    filtered_ASR.set_index('biosample_id', inplace=True)
    filtered_ASR = filtered_ASR[filtered_ASR['units'] == 'mg/L']
    filtered_ASR = filtered_ASR[filtered_ASR['ast_standard'] == 'CLSI']
    filtered_ASR = filtered_ASR[filtered_ASR['species_fam'] != 'senterica']
    filtered_ASR = filtered_ASR[filtered_ASR['species_fam'] != 'spneumoniae']
    data.set_index('biosample_id', inplace=True)
    data.drop(['Unnamed: 0', 'species_fam', 'run_id'], axis=1, inplace=True)

    if species_sep:
        species = filtered_ASR['species_fam'].value_counts().reset_index()['index'].iloc[species_filter_index]
        filtered_ASR = filtered_ASR[filtered_ASR['species_fam'] == species]
    else:
        species = None

    if antibiotic_sep:
        anti_list = filtered_ASR['antibiotic_name'].value_counts().index.values
        label = anti_list[antibiotic_index]
        filtered_ASR = filtered_ASR[filtered_ASR['antibiotic_name'] == label]
    else:
        label = 'measurement'

    range_ASR = filtered_ASR[filtered_ASR['measurement_sign'] != '=']
    exact_ASR = filtered_ASR[filtered_ASR['measurement_sign'] == '=']

    range_data_values = data.loc[range_ASR.index][label]
    range_labels = pd.DataFrame({
        'values': [],
        'direction': [],
    })
    range_labels['values'] = range_data_values.apply(lambda x: float(x.split(' ')[1]))
    range_labels['direction'] = range_data_values.apply(lambda x: x.split(' ')[0].replace('=', ''))

    exact_y = data.loc[exact_ASR.index][label]
    if task == 'regression':
        exact_y = exact_y.apply(lambda x: float(x.split(' ')[1]))
    elif task == 'classification':
        exact_y = exact_y.apply(lambda x: str(x.split(' ')[1]))

    exact_X = data.loc[exact_ASR.index][features]
    exact_X.dropna(axis=1, how='all', inplace=True)
    exact_X.fillna(0, inplace=True)
    exact = exact_X.merge(exact_y, left_index=True, right_index=True)

    if not naive:
        range_y = range_data_values
        if task == 'regression':
            if strip_range_train:
                range_y = range_labels['values']
            elif distance_range_train:
                signs = range_labels['direction']
                range_y = range_labels['values']
                range_y = range_y.mask(signs == '>', range_y + range_moved)
                range_y = range_y.mask(signs == '<', range_y - range_moved)
            else:
                print('regression not in the naive approach is not implemented yet.')
        range_y.name = label
    else:
        range_y = pd.DataFrame({})

    train_features = exact_X.columns.values
    range_X = data.loc[range_ASR.index][train_features]
    range_X.fillna(0, inplace=True)
    range_X = range_X.reset_index().set_index('biosample_id')
    range_data = range_X.merge(range_y, left_index=True, right_index=True)

    range_train_ids = []
    range_test_ids = []
    if reg_stratified:
        exact_train_ids, exact_test_ids = strat_id(exact_y)
        if not naive:
            range_train_ids, range_test_ids = strat_id(range_y)
    else:
        exact_train_ids, exact_test_ids = train_test_split(list(exact_y.index), test_size=0.2, random_state=42)
        if not naive:
            range_train_ids, range_test_ids = train_test_split(list(range_y.index), test_size=0.2, random_state=42)
    exact_train = exact.loc[exact_train_ids,]
    range_train = range_data.loc[range_train_ids,]
    train = pd.concat([exact_train, range_train])
    test = pd.concat([exact.loc[exact_test_ids,], range_data.loc[range_test_ids,]])

    # generate cv from train that is stratified
    cv = []
    train_ref = train.reset_index()
    for i in np.arange(cv_num):
        exact_train_ids, exact_test_ids = strat_id(exact_train[label], seed_add=i)
        range_train_ids, range_test_ids = [], []
        if not naive:
            range_train_ids, range_test_ids = strat_id(range_train[label], seed_add=i)
        cv_train_ids = exact_train_ids + range_train_ids
        cv_test_ids = exact_test_ids + range_test_ids
        cv.append((list(train_ref[train_ref['index'].isin(cv_train_ids)].index),
                   list(train_ref[train_ref['index'].isin(cv_test_ids)].index)))

    return train, test, range_X, range_labels, list(train_features), label, species, cv

def strat_id(y, seed_add=0):
    train_ids = []
    test_ids = []
    for y_val in set(y.values):
        sub_value_id = list(y[y == y_val].index)
        if len(sub_value_id) > 1:
            train_id, test_id = train_test_split(sub_value_id, test_size=0.2, random_state=42+seed_add)
            train_ids = train_ids + train_id
            test_ids = test_ids + test_id
        else:
            train_ids = train_ids + sub_value_id
    return train_ids, test_ids