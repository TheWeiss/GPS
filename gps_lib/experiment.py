import os
from tqdm import tqdm
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
from h2o.automl import H2OAutoML
import argparse
import h2o
from abc import ABC, abstractmethod
import json


class Model(ABC):
    def __init__(self, exp_name, model_name, ds_param_files_path, exp_dir_path='../experiments/'):
        super().__init__()
        self.exp_name = exp_name
        self.model_name = model_name
        self.ds_param_files_path = ds_param_files_path
        with open(self.ds_param_files_path + '/col_names.json') as json_file:
            self.col_names = json.load(json_file)
        self.exp_dir_path = exp_dir_path

    def get_train(self):
        return pd.read_csv('{}/train.csv'.format(self.ds_param_files_path))[self.col_names['features']]

    def get_test(self):
        return pd.read_csv('{}/test.csv'.format(self.ds_param_files_path))[self.col_names['features']]

    def get_range(self):
        return pd.read_csv('{}/range_X.csv'.format(self.ds_param_files_path))

    @abstractmethod
    def predict(self, X_test):
        pass

class Model_h2o(Model):
    def __init__(self, exp_name, model_name, ds_param_files_path, exp_dir_path='../experiments/'):
        super().__init__(exp_name, model_name, ds_param_files_path, exp_dir_path)
        h2o.init()
        model_path = os.listdir(('{}{}/{}/model'.format(self.exp_dir_path, self.exp_name, self.model_name)))[0]
        self.model = h2o.load_model('{}{}/{}/model/{}'.format(self.exp_dir_path, self.exp_name, self.model_name, model_path))

    def convert_X_test(self, X_test):
        X_test = pd.DataFrame(data = X_test, columns = self.col_names['features'])
        X_test[self.col_names['id']] = np.arange(len(X_test))
        X_test[self.col_names['label']] = 0
        test_h2o = h2o.H2OFrame(X_test)
        return test_h2o

    def get_model(self):
        return self.model

    def predict(self, X_test):
        test_h2o = self.convert_X_test(X_test)
        test_preds = self.model.predict(test_h2o).as_data_frame()
        h2o.remove(test_h2o)
        return test_preds


class Model_axgb(Model):
    def __init__(self, exp_name, model_name, ds_param_files_path, exp_dir_path='../experiments/'):
        super().__init__(exp_name, model_name, ds_param_files_path, exp_dir_path)

    def predict(self, X_test):
        X_test.to_csv('{}{}/{}/tmp_test_X.csv'.format(self.exp_dir_path, self.exp_name, self.model_name))
        PredictAutoXGBCommand('{}{}/{}/model'.format(self.exp_dir_path, self.exp_name, self.model_name),
                              '{}{}/{}/tmp_test_X.csv'.format(self.exp_dir_path, self.exp_name, self.model_name),
                              '{}{}/{}/tmp_test_preds.csv'.format(
                                  self.exp_dir_path, self.exp_name, self.model_name)).execute()
        test_preds = pd.read_csv('{}{}/{}/tmp_test_preds.csv'.format(self.exp_dir_path, self.exp_name, self.model_name))
        os.system("rm -R " + "'/sise/home/amitdanw/GPS/experiments/{}/{}/tmp_test_X.csv'".format(
            self.exp_name, self.model_name))
        os.system("rm -R " + "'/sise/home/amitdanw/GPS/experiments/{}/{}/tmp_test_preds.csv'".format(
            self.exp_name, self.model_name))
        return test_preds

def run_h2o(exp_name, model_param, ds_param_files_path, col_names):
    h2o.init()
    # Import a sample binary outcome train/test set into H2O
    trainH2o = h2o.import_file('{}/train.csv'.format(ds_param_files_path))
    testH2o = h2o.import_file('{}/test.csv'.format(ds_param_files_path))
    rangeH2o = h2o.import_file('{}/range_X.csv'.format(ds_param_files_path))
    model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])

    # Identify predictors and response
    x = col_names['features']
    y = col_names['label']

    # Run AutoML for 20 base models
    aml = H2OAutoML(
        max_models=model_param['max_models'],
        seed=42,
        max_runtime_secs=model_param['train_time'],
        stopping_metric='RMSE',
    )
    aml.train(x=x, y=y, training_frame=trainH2o)
    #
    # View the AutoML Leaderboard
    lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
    lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
    #
    model = aml.leader
    model_path = h2o.save_model(model=model, path='../experiments/{}/{}/model'.format(exp_name, model_name), force=True)
    lb.as_data_frame().to_csv('../experiments/{}/{}/leader_board.csv'.format(exp_name, model_name))
    test_preds = model.predict(testH2o).as_data_frame()
    range_preds = model.predict(rangeH2o).as_data_frame()
    train_preds = model.predict(trainH2o).as_data_frame()
    test_preds.to_csv('../experiments/{}/{}/test_preds.csv'.format(exp_name, model_name))
    range_preds.to_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
    train_preds.to_csv('../experiments/{}/{}/train_preds.csv'.format(exp_name, model_name))
    # h2o.shutdown()
    return aml


def run_autoxgb(exp_name, model_param, ds_param_files_path, col_names):
    model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
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


def run_exp(dataset: ds.MICDataSet, model_param, ds_param=None, species=None, antibiotic=None, exp_desc='',
            run_over=False):
    if type(species) == list:
        for species_j in species:
            if type(antibiotic) == list:
                for antibiotic_i in antibiotic:
                    run_exp(dataset, model_param, ds_param, species_j, antibiotic_i, exp_desc, run_over=False)
            else:
                run_exp(dataset, model_param, ds_param, species_j, antibiotic, exp_desc, run_over=False)
    else:
        if type(antibiotic) == list:
            for antibiotic_i in antibiotic:
                run_exp(dataset, model_param, ds_param, species, antibiotic_i, exp_desc, run_over=False)
        else:
            try:
                train, test, range_X, range_y, col_names, ds_param_files_path, species_name, antibiotic_name, cv = dataset.generate_dataset(
                    ds_param, species, antibiotic)

            except Exception as e:
                print(type(e))
                print(e)
                return -1
            exp_name = '|' + '|'.join(
                [ds_param_files_path.split('/')[-3::][i] for i in [1, 2, 0]]) + '|' + dataset.name + '|' + exp_desc

            os.makedirs('../experiments/{}'.format(exp_name), exist_ok=True)
            with open('../experiments/{}/data_path.txt'.format(exp_name), "w") as data_path:
                data_path.write(ds_param_files_path)
            if len(train) < 40:
                with open('../experiments/{}/tb.txt'.format(exp_name), 'w+') as f:
                    f.write('Training set doesnt have at-least 40 samples reqiered for training')
                    print('{} is too small, train size - {}'.format(exp_name, len(train)))
                    return -1
            model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
            os.makedirs('../experiments/{}/{}'.format(exp_name, model_name), exist_ok=True)
            pd.DataFrame(model_param, index=[0]).to_csv(
                '../experiments/{}/{}/model_param.csv'.format(exp_name, model_name))

            if not run_over:
                if os.path.exists('../experiments/{}/{}/tb.txt'.format(exp_name, model_name)) or \
                        os.path.exists('../experiments/{}/{}/test_preds.csv'.format(exp_name, model_name)):
                    print('{}|{} was already run'.format(exp_name, model_name))
                    return 0
            try:
                if model_param['model'] == 'autoxgb':
                    run_autoxgb(exp_name, model_param, ds_param_files_path, col_names)
                elif model_param['model'] == 'h2o':
                    run_h2o(exp_name, model_param, ds_param_files_path, col_names)
                print('{}|{} done running exp'.format(exp_name, model_name))
                return 0
            except Exception as e:
                with open('../experiments/{}/{}/tb.txt'.format(exp_name, model_name), 'w+') as f:
                    traceback.print_exc(file=f)
                print('{}|{} ERROR: {}'.format(exp_name, model_name, e))
                return -1


def main(args):
    pre_params = None

    data = ds.CollectionDataSet(dbs_name_list=args.data_sets, pre_params=pre_params)

    ds_param = {}
    if args.handle_range:
        ds_param['handle_range'] = args.handle_range
    if args.move_range_by:
        ds_param['move_range_by'] = args.move_range_by
    if ds_param == {}:
        ds_param = None

    model_param = {}
    model_param['model'] = args.model
    model_param['train_time'] = args.train_time
    model_param['max_models'] = args.max_models

    if type(args.anti_list) == list:
        anti_list = [int(anti) if anti.isnumeric() else anti for anti in args.anti_list]
    else:
        if args.anti_list.isnumeric():
            anti_list = int(args.anti_list)

    if type(args.species_list) == list:
        species_list = [int(species) if species.isnumeric() else ' '.join(species.split('_')) for species in args.species_list]
    else:
        if args.species_list.isnumeric():
            species_list = int(args.species_list)
    run_exp(data, model_param, ds_param, species=species_list, antibiotic=anti_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-over', dest='run_over', action="store_true")

    # dataset to load
    parser.add_argument('--data-sets', dest='data_sets', default=['PATAKI', 'VAMP', 'PA', 'PATRIC'], nargs='+')

    # pairs to generate X-y data for
    parser.add_argument('--species-list', dest='species_list', default=0, nargs='+')
    parser.add_argument('--anti-list', dest='anti_list', default=0, nargs='+')

    # ds parameters like range handling
    parser.add_argument('--handle-range', dest='handle_range',
                        choices=['remove', 'strip', 'move'])
    parser.add_argument('--move-range-by', dest='move_range_by', type=int, nargs='?')

    # pre parameters like geno thresholds

    # models to run
    parser.add_argument('--model', dest='model', default='autoxgb', type=str, nargs='?')
    parser.add_argument('--train-time', dest='train_time', default=1, type=int, nargs='?')
    parser.add_argument('--max-models', dest='max_models', default=1, type=int, nargs='?')

    args = parser.parse_args()
    main(args)
