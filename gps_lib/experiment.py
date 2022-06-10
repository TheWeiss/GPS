import os
from tqdm import tqdm
import h2o
import pandas as pd
import re
import glob
import numpy as np
from autoxgb.cli.predict import PredictAutoXGBCommand
from parse_raw_utils import get_isolate_features
import data_sets as ds


def run_h2o(model_param, exp_name):
    print(exp_name)
    # Import a sample binary outcome train/test set into H2O
    trainH2o = h2o.import_file('../experiments/{}/train.csv'.format(exp_name))
    print(trainH2o.as_data_frame().columns[-1])
    testH2o = h2o.import_file('../experiments/{}/test.csv'.format(exp_name))
    rangeH2o = h2o.import_file('../experiments/{}/X_range.csv'.format(exp_name))
    model_name = '_'.join(['_'.join([k, str(v)]) for k, v in model_param.items()])

    # Identify predictors and response
    x = list(pd.read_csv('../experiments/{}/features.csv'.format(exp_name))['features'].values)
    y = pd.read_csv('../experiments/{}/label.csv'.format(exp_name)).loc[0, 'label']

    # Run AutoML for 20 base models
    aml = H2OAutoML(max_models=model_param['max_models'], seed=42, max_runtime_secs=model_param['train_time'])
    aml.train(x=x, y=y, training_frame=trainH2o)

    # View the AutoML Leaderboard
    lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
    lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

    model = aml.leader
    model_path = h2o.save_model(model=model, path='../experiments/{}/{}/model'.format(exp_name, model_name), force=True)
    lb.as_data_frame().to_csv('../experiments/{}/{}/leader_board.csv'.format(exp_name, model_name))
    test_preds = model.predict(testH2o).as_data_frame()
    range_preds = model.predict(rangeH2o).as_data_frame()
    train_preds = model.predict(trainH2o).as_data_frame()
    test_preds.to_csv('../experiments/{}/{}/test_preds.csv'.format(exp_name, model_name))
    range_preds.to_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
    train_preds.to_csv('../experiments/{}/{}/train_preds.csv'.format(exp_name, model_name))
    return aml


def fill_all_results():
    h2o.init()
    exp_dir_path = '../experiments/'
    results = pd.DataFrame({})
    for path in os.listdir(exp_dir_path):
        data_param_path = exp_dir_path + path + '/data_param.csv'
        if not os.path.exists(data_param_path):
            continue
        data_param = pd.read_csv(data_param_path)
        x = next(walk_level(exp_dir_path + path))
        for model_dir in x[1]:
            if '.ipynb' in model_dir:
                continue
            if os.path.exists(exp_dir_path + path + '/' + model_dir + '/model_param.csv'):
                pass
            else:
                model_name = model_dir.split('_')[1]
                if model_name == 'h2o':
                    pass
                else:
                    continue
                trainH2o = h2o.import_file(exp_dir_path + path + '/train.csv')
                model_path = os.listdir(('{}{}/{}/model'.format(exp_dir_path, path, model_dir)))[0]
                model = h2o.load_model('{}{}/{}/model/{}'.format(exp_dir_path, path, model_dir, model_path))
                train_preds = model.predict(trainH2o).as_data_frame()
                train_preds.to_csv('{}{}/{}/train_preds.csv'.format(exp_dir_path, path, model_dir))


def walk_level(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def run_exp(dataset: ds.MICDataSet, ds_param, antibiotic=None, species=None):
    if ds_param is None:
        pass
        # default_values = {
        #     'species_sep': True,
        #     'antibiotic_sep': True,
        #     'handle_range': 'remove',  # remove/strip/move
        #     'handle_multi_mic': 'remove',  # remove/max/min/rand
        #     'ignore_small_dilu': True,  # remove/max/min/rand
        #     'task': 'regression',  # regression/classification/SIR
        #     'log2': True,
        #     'move_range_by': 5,
        #     'reg_stratified': True,
        #     'stratified_cv_num': 3,
        # }
    train, test, range_data, col_names, ds_param_files_path, antibiotic_name, species_name = dataset.generate_dataset(ds_param, antibiotic, species)
    exp_name = '_'.join(ds_param_files_path.split('/')[-3::-1])

    os.makedirs('../experiments/{}'.format(exp_name), exist_ok=True)
    with open('../experiments/data_path.txt', "w") as data_path:
        data_path.write(ds_param_files_path)



def main():
    species_filter_index_list = [0,1]
    antibiotic_index_list = [0,1]  # np.arange(10, 20)

    data_param = {
        'naive': True,
        'strip_range_train': False,
        'distance_range_train': False,
        'reg_stratified': True,
        'species_sep': True,
        'antibiotic_sep': True,
        'exp_describtion': 'naive_strat',
    }

    model_param = {
        'model': 'h2o',
        'train_time': 3600,
        'max_models': 100,
    }

    exp_names = []
    for species_filter_index in tqdm(species_filter_index_list):
        for antibiotic_index in tqdm(antibiotic_index_list):
            train, test, X_range, y_range, features, label, species = get_filtered_data(
                data='tot_filtered_data.csv',
                features='final_features',
                ASR_data='filtered_ASR_data.csv',
                species_filter_index=species_filter_index,
                antibiotic_index=antibiotic_index,
                species_sep=data_param['species_sep'],
                antibiotic_sep=data_param['antibiotic_sep'],
                naive=data_param['naive'],
                strip_range_train=data_param['strip_range_train'],
                reg_stratified=data_param['reg_stratified'],
                distance_range_train=data_param['distance_range_train'],
                task='regression',
            )
            data_param['species'] = species
            data_param['antibiotic'] = label
            exp_name = '{}_{}_{}'.format(species, label, data_param['exp_describtion'])
            exp_names.append(exp_name)
            os.makedirs('../experiments/{}'.format(exp_name), exist_ok=True)
            train.to_csv('../experiments/{}/train.csv'.format(exp_name))
            test.to_csv('../experiments/{}/test.csv'.format(exp_name))
            X_range.to_csv('../experiments/{}/X_range.csv'.format(exp_name))
            y_range.to_csv('../experiments/{}/y_range.csv'.format(exp_name))
            pd.DataFrame({'features': features}).to_csv('../experiments/{}/features.csv'.format(exp_name))
            pd.DataFrame({'label': [label]}).to_csv('../experiments/{}/label.csv'.format(exp_name))
            pd.DataFrame(data_param, index=[0]).to_csv('../experiments/{}/data_param.csv'.format(exp_name))

    h2o.init()
    for exp_name in tqdm(exp_names):
        a = run_h2o(model_param, exp_name)

if __name__ == "__main__":
    main()