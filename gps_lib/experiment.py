import os
from tqdm import tqdm
import h2o
import pandas as pd
import re
import glob
import numpy as np
from parse_raw_utils import get_isolate_features
import data_sets as ds
from data_sets import SpecAntiNotExistError
from autoxgb import AutoXGB
from autoxgb.cli.predict import PredictAutoXGBCommand
import sys
import traceback
import logging
import getopt



def run_h2o(exp_name, model_param, ds_param_files_path, col_names):
    pass
    # print(exp_name)
    # # Import a sample binary outcome train/test set into H2O
    # trainH2o = h2o.import_file('../experiments/{}/train.csv'.format(exp_name))
    # print(trainH2o.as_data_frame().columns[-1])
    # testH2o = h2o.import_file('../experiments/{}/test.csv'.format(exp_name))
    # rangeH2o = h2o.import_file('../experiments/{}/X_range.csv'.format(exp_name))
    # model_name = '_'.join(['_'.join([k, str(v)]) for k, v in model_param.items()])
    #
    # # Identify predictors and response
    # x = list(pd.read_csv('../experiments/{}/features.csv'.format(exp_name))['features'].values)
    # y = pd.read_csv('../experiments/{}/label.csv'.format(exp_name)).loc[0, 'label']
    #
    # # Run AutoML for 20 base models
    # aml = H2OAutoML(max_models=model_param['max_models'], seed=42, max_runtime_secs=model_param['train_time'])
    # aml.train(x=x, y=y, training_frame=trainH2o)
    #
    # # View the AutoML Leaderboard
    # lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
    # lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
    #
    # model = aml.leader
    # model_path = h2o.save_model(model=model, path='../experiments/{}/{}/model'.format(exp_name, model_name), force=True)
    # lb.as_data_frame().to_csv('../experiments/{}/{}/leader_board.csv'.format(exp_name, model_name))
    # test_preds = model.predict(testH2o).as_data_frame()
    # range_preds = model.predict(rangeH2o).as_data_frame()
    # train_preds = model.predict(trainH2o).as_data_frame()
    # test_preds.to_csv('../experiments/{}/{}/test_preds.csv'.format(exp_name, model_name))
    # range_preds.to_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
    # train_preds.to_csv('../experiments/{}/{}/train_preds.csv'.format(exp_name, model_name))
    # return aml

def run_autoxgb(exp_name, model_param, ds_param_files_path, col_names):
    model_name = '_'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
    # required parameters:
    train_filename = '{}/train.csv'.format(ds_param_files_path)
    output = '../experiments/{}/{}/model'.format(exp_name, model_name)
    if os.path.exists('/sise/home/amitdanw/GPS/experiments/{}/{}/model'.format(exp_name, model_name)):
        os.system("rm -R " + "'/sise/home/amitdanw/GPS/experiments/{}/{}/model'".format(exp_name, model_name))
        os.system("rm -R " + "'/sise/home/amitdanw/GPS/experiments/{}/{}'".format(exp_name, model_name))

    # optional parameters
    test_filename = '{}/test.csv'.format(ds_param_files_path)
    task = 'regression'
    idx = 'run_id'
    targets = [col_names['label']]
    features = col_names['features']
    categorical_features = None
    use_gpu = True
    num_folds = 5
    seed = 42
    num_trials = model_param['max_models']
    time_limit = model_param['train_time']
    fast = True

    # os.system("autoxgb train \
    #             --train_filename {} \
    #             --output {} \
    #             --test_filename {} \
    #             --use_gpu")
    # Now its time to train the model!
    axgb = AutoXGB(
        train_filename=train_filename,
        output=output,
        test_filename=test_filename,
        task=task,
        idx=idx,
        targets=targets,
        features=features,
        categorical_features=categorical_features,
        use_gpu=use_gpu,
        num_folds=num_folds,
        seed=seed,
        num_trials=num_trials,
        time_limit=time_limit,
        fast=fast,
    )
    axgb.train()
    try:
        PredictAutoXGBCommand('../experiments/{}/{}/model'.format(exp_name, model_name),
                          '{}/range_X.csv'.format(ds_param_files_path),
                          '../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name)).execute()
    except ValueError:
        pd.DataFrame({}).to_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
    os.rename("../experiments/{}/{}/model/oof_predictions.csv".format(exp_name, model_name),
              "../experiments/{}/{}/train_preds.csv".format(exp_name, model_name))
    os.rename("../experiments/{}/{}/model/test_predictions.csv".format(exp_name, model_name),
              "../experiments/{}/{}/test_preds.csv".format(exp_name, model_name))

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

def run_exp(dataset: ds.MICDataSet, model_param, ds_param=None, species=None, antibiotic=None, exp_desc = ''):
    if type(species) == list:
        for species_j in species:
            if type(antibiotic)==list:
                for antibiotic_i in antibiotic:
                            run_exp(dataset, model_param, ds_param, species_j, antibiotic_i, exp_desc)
            else:
                run_exp(dataset, model_param, ds_param, species, antibiotic_i, exp_desc)
    else:
        if type(antibiotic) == list:
            for antibiotic_i in antibiotic:
                run_exp(dataset, model_param, ds_param, species, antibiotic_i, )
        else:
            try:
                train, test, range_X, range_y, col_names, ds_param_files_path, species_name, antibiotic_name, cv = dataset.generate_dataset(
                    ds_param, species, antibiotic)
            except Exception as e:
                print(e)
                return -1
            exp_name = '_'+'_'.join([ds_param_files_path.split('/')[-3::][i] for i in [1, 2, 0]])+'_'+exp_desc

            os.makedirs('../experiments/{}'.format(exp_name), exist_ok=True)
            with open('../experiments/{}/data_path.txt'.format(exp_name), "w") as data_path:
                data_path.write(ds_param_files_path)
            model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
            os.makedirs('../experiments/{}/{}'.format(exp_name, model_name), exist_ok=True)
            pd.DataFrame(model_param, index=[0]).to_csv(
                '../experiments/{}/{}/model_param.csv'.format(exp_name, model_name))
            try:
                if model_param['model'] == 'autoxgb':
                    run_autoxgb(exp_name, model_param, ds_param_files_path, col_names)
                elif model_param['model'] == 'h2o':
                    run_h2o(exp_name, model_param, ds_param_files_path, col_names)
            except Exception as e:
                with open('../experiments/{}/{}/tb.txt'.format(exp_name, model_name), 'w+') as f:
                    traceback.print_exc(file=f)
                print("Unexpected error:", e)
                return -1


def main_h2o():
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


def main(argv):
    pre_params=None
    data = ds.CollectionDataSet(dbs_name_list=[
        'PATAKI',
        'VAMP',
        'PA',
        'PATRIC',
    ], pre_params=pre_params)

    model_param = {
        'model': 'autoxgb',
        'train_time': 3600,
        'max_models': 100,
    }
    ds_param = None

    species_list = [0, 1]  # ['Pseudomonas aeruginosa'] #+list(np.arange(5))
    anti_list = [20] #list(np.arange(0, 20))
    run_exp(data, model_param, ds_param, species=species_list, antibiotic=anti_list)



if __name__ == "__main__":
    # # Store argument variable omitting the script name
    # argv = sys.argv[1:]
    #
    # try:
    #     # Define getopt short and long options
    #     options, args = getopt.getopt(sys.argv[1:], 's:a', ['spec=', 'anti='])
    #
    #     # Read each option using for loop
    #     for opt, arg in options:
    #         # Calculate the sum if the option is -a or --add
    #         if opt in ('-s', '--spec'):
    #             spec = int(argv[1]) + int(argv[2])
    #
    #         # Calculate the suntraction if the option is -s or --sub
    #         elif opt in ('-s', '--sub'):
    #             result = int(argv[1]) - int(argv[2])
    #
    #     print('Result = ', result)
    #
    # except getopt.GetoptError:
    #
    #     # Print the error message if the wrong option is provided
    #     print('The wrong option is provided')
    #
    #     # Terminate the script
    #     sys.exit(2)
    # main(sys.argv[1:])