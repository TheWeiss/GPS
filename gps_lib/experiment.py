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
        with open('{}/col_names.json'.format(self.ds_param_files_path)) as json_file:
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
        test_preds_h2o = self.model.predict(test_h2o)
        test_preds = test_preds_h2o.as_data_frame().copy()
        h2o.remove(test_h2o)
        h2o.remove(test_preds_h2o)
        return test_preds


class Model_axgb(Model):
    def __init__(self, exp_name, model_name, ds_param_files_path, exp_dir_path='../experiments/'):
        super().__init__(exp_name, model_name, ds_param_files_path, exp_dir_path)

    def convert_X_test(self, X_test):
        X_test = pd.DataFrame(data=X_test, columns=self.col_names['features'])
        X_test[self.col_names['id']] = np.arange(len(X_test))
        X_test[self.col_names['label']] = 0
        X_test.to_csv('{}{}/{}/tmp_test_X.csv'.format(self.exp_dir_path, self.exp_name, self.model_name))

    def predict(self, X_test):
        self.convert_X_test(X_test)
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

def run_h2o(exp_path, model_param, ds_param_files_path, col_names):
    h2o.init()
    # Import a sample binary outcome train/test set into H2O
    train = pd.read_csv('{}/train.csv'.format(ds_param_files_path))
    test = pd.read_csv('{}/test.csv'.format(ds_param_files_path))
    trainH2o = h2o.H2OFrame(train)
    testH2o = h2o.H2OFrame(test)
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
    model_path = h2o.save_model(model=model, path='{}/{}/model'.format(exp_path, model_name), force=True)
    lb.as_data_frame().to_csv('{}/{}/leader_board.csv'.format(exp_path, model_name))
    test_preds = model.predict(testH2o).as_data_frame()
    range_preds = model.predict(rangeH2o).as_data_frame()
    train_preds = model.predict(trainH2o).as_data_frame()
    test_preds.to_csv('{}/{}/test_preds.csv'.format(exp_path, model_name))
    range_preds.to_csv('{}/{}/range_preds.csv'.format(exp_path, model_name))
    train_preds.to_csv('{}/{}/train_preds.csv'.format(exp_path, model_name))
    # h2o.shutdown()
    return aml


def run_autoxgb(exp_path, model_param, ds_param_files_path, col_names):
    model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
    # required parameters:
    train_filename = '{}/train.csv'.format(ds_param_files_path)
    output = '{}/{}/model'.format(exp_path, model_name)
    full_exp_path = '/sise/home/amitdanw/GPS/{}'.format(exp_path[3:])
    if os.path.exists('{}/{}/model'.format(full_exp_path, model_name)):
        os.system("rm -R " + "'{}/{}/model'".format(full_exp_path, model_name))
        os.system("rm -R " + "'{}/{}'".format(full_exp_path, model_name))

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
        PredictAutoXGBCommand('{}/{}/model'.format(exp_path, model_name),
                              '{}/range_X.csv'.format(ds_param_files_path),
                              '{}/{}/range_preds.csv'.format(exp_path, model_name)).execute()
    except ValueError:
        pd.DataFrame({}).to_csv('{}/{}/range_preds.csv'.format(exp_path, model_name))
    os.rename("{}/{}/model/oof_predictions.csv".format(exp_path, model_name),
              "{}/{}/train_preds.csv".format(exp_path, model_name))
    os.rename("{}/{}/model/test_predictions.csv".format(exp_path, model_name),
              "{}/{}/test_preds.csv".format(exp_path, model_name))


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


def generate_stacked_dataset(results, data_index):
    data_path = results.loc[results.iloc[data_index].name, 'data_path']
    ds_param = pd.read_csv('{}/ds_param.csv'.format(data_path))
    with open(data_path + '/col_names.json') as json_file:
        col_names = json.load(json_file)
    with open(data_path + '/cv.json') as json_file:
        cv = json.load(json_file)
    train = pd.read_csv('{}/train.csv'.format(data_path))
    test = pd.read_csv('{}/test.csv'.format(data_path))
    range_X = pd.read_csv('{}/range_X.csv'.format(data_path))
    range_y = pd.read_csv('{}/range_y.csv'.format(data_path))
    new_train = train[[col_names['id'], col_names['label']]]
    new_test = test[[col_names['id'], col_names['label']]]
    new_range = range_X[[col_names['id']]]

    for model_index in np.arange(len(results)):
        exp_name = results.loc[results.iloc[model_index].name, 'exp_path']
        model_type = results.loc[results.iloc[model_index].name, 'model']
        model_name = results.loc[results.iloc[model_index].name, 'model_path']

        if model_type == 'h2o':
            model = Model_h2o(exp_name, model_name, data_path)
            results_col = 0
        elif model_type == 'autoxgb':
            model = Model_axgb(exp_name, model_name, data_path)
            results_col = 1
        else:
            print('model_type not supported: {}'.format(model_type))

        train_preds = model.predict(train).iloc[:, results_col]
        test_preds = model.predict(test).iloc[:, results_col]
        range_preds = model.predict(range_X).iloc[:, results_col]
        train_preds.name = '{}_preds'.format(results.loc[results.iloc[model_index].name, 'antibiotic'])
        test_preds.name = '{}_preds'.format(results.loc[results.iloc[model_index].name, 'antibiotic'])
        range_preds.name = '{}_preds'.format(results.loc[results.iloc[model_index].name, 'antibiotic'])
        new_train = pd.concat([new_train, train_preds], axis=1)
        new_test = pd.concat([new_test, test_preds], axis=1)
        new_range = pd.concat([new_range, range_preds], axis=1)
        col_names['features'] = list(set(new_train.columns) - set([col_names['id'], col_names['label']]))
    return new_train, new_test, new_range, range_y, col_names, ds_param, cv


def run_exp_stack(stacked_param, model_param, species=None, antibiotic=None, exp_desc='',
            run_over=False, exp_dir_path='../experiments', data_base_path='../pre_proccesing/base_line/PATAKI_VAMP_PA_PATRIC'):
    if type(species) == list:
        for species_j in species:
            if type(antibiotic) == list:
                for antibiotic_i in antibiotic:
                    run_exp(stacked_param, model_param, species_j, antibiotic_i, exp_desc, run_over=run_over)
            else:
                run_exp(stacked_param, model_param, species_j, antibiotic, exp_desc, run_over=run_over)
    else:
        if type(antibiotic) == list:
            for antibiotic_i in antibiotic:
                run_exp(stacked_param, model_param, species, antibiotic_i, exp_desc, run_over=run_over)
        else:
            stacked_name = '|'.join([':'.join([k, str(v)]) for k, v in stacked_param.items()])
            if not os.path.exists('{}/{}'.format(data_base_path, stacked_name)):
                os.makedirs('{}/{}'.format(data_base_path, stacked_name))
            pd.DataFrame(stacked_param, index=[0]).to_csv('{}/{}/stacked_param.csv'.format(data_base_path, stacked_name))
            res = pd.read_csv('{}/results_summery.csv'.format(exp_dir_path)).drop('Unnamed: 0', axis=1)
            res = res[res['stacked'] == False]
            res = res[res['train_time'] > 100]

            results = res.sort_values(ascending=False, by='{}_test'.format(stacked_param['metric'])).drop_duplicates(
                subset=['species', 'antibiotic'], keep='first')
            results = results.sort_values(ascending=False, by='{}_test'.format(stacked_param['metric'])).reset_index(
                drop=True)

            if stacked_param.get('filter_small'):
                results = results[results['size'] > 100][results['exact_size'] > 50]
            if stacked_param.get('filter_learned'):
                results = results[results['learned_essential_agreement_test'] > 1.05][results['learned_RMSE_test'] <0.95]
            if stacked_param.get('species_sep'):
                results = results[results['species']==species]
            results.sort_values(by='{}_test'.format(stacked_param['metric']), inplace=True)
            if not os.path.exists('{}/{}/{}'.format(data_base_path, stacked_name, species)):
                os.makedirs('{}/{}/{}'.format(data_base_path, stacked_name, species))
            results.to_csv('{}/{}/{}/models_to_stack.csv'.format(data_base_path, stacked_name, species))


            for data_index in np.arange(len(results)):
                try:
                    train, test, range_X, range_y, col_names, ds_param, cv = generate_stacked_dataset(results, data_index)
                    data_antibiotic = results.loc[results.iloc[data_index].name, 'antibiotic']
                    ds_param_files_path = '{}/{}/{}/{}'.format(data_base_path, stacked_name, species, data_antibiotic)
                    if not os.path.exists(ds_param_files_path):
                        os.makedirs(ds_param_files_path)

                    train.set_index(col_names['id']).to_csv('{}/train.csv'.format(ds_param_files_path))
                    test.set_index(col_names['id']).to_csv('{}/test.csv'.format(ds_param_files_path))
                    range_X.set_index(col_names['id']).to_csv('{}/range_X.csv'.format(ds_param_files_path))
                    range_y.set_index(col_names['id']).to_csv('{}/range_y.csv'.format(ds_param_files_path))
                    if 'Unnamed: 0' in ds_param.columns:
                        ds_param.drop('Unnamed: 0', inplace=True, axis=1)
                    pd.DataFrame(ds_param, index=[0]).to_csv('{}/ds_param.csv'.format(ds_param_files_path), index=False)
                    with open('{}/col_names.json'.format(ds_param_files_path), "w") as fp:
                        json.dump(col_names, fp)
                    with open('{}/cv.json'.format(ds_param_files_path), "w") as fp:
                        json.dump(cv, fp)

                except Exception as e:
                    print(type(e))
                    print(e)
                    continue
                exp_name = '|' + '|'.join(
                    [ds_param_files_path.split('/')[-3::][i] for i in [1, 2, 0]]) + '|' + 'PATAKI_VAMP_PA_PATRIC' + '|' + exp_desc

                os.makedirs('{}{}'.format(exp_dir_path, exp_name), exist_ok=True)
                with open('{}{}/data_path.txt'.format(exp_dir_path, exp_name), "w") as data_path:
                    data_path.write(ds_param_files_path)
                if len(train) < 40:
                    with open('{}{}/tb.txt'.format(exp_dir_path, exp_name), 'w+') as f:
                        f.write('Training set doesnt have at-least 40 samples reqiered for training')
                        print('{} is too small, train size - {}'.format(exp_name, len(train)))
                        continue
                model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
                os.makedirs('{}{}/{}'.format(exp_dir_path, exp_name, model_name), exist_ok=True)
                pd.DataFrame(model_param, index=[0]).to_csv(
                    '{}{}/{}/model_param.csv'.format(exp_dir_path, exp_name, model_name))

                if not run_over:
                    if os.path.exists('{}{}/{}/tb.txt'.format(exp_dir_path, exp_name, model_name)) or \
                            os.path.exists('{}{}/{}/test_preds.csv'.format(exp_dir_path, exp_name, model_name)):
                        print('{}|{} was already run'.format(exp_name, model_name))
                        continue
                try:
                    if model_param['model'] == 'autoxgb':
                        run_autoxgb(exp_name, model_param, ds_param_files_path, col_names)
                    elif model_param['model'] == 'h2o':
                        run_h2o(exp_name, model_param, ds_param_files_path, col_names)
                    print('{}|{} done running exp'.format(exp_name, model_name))
                    continue
                except Exception as e:
                    with open('{}{}/{}/tb.txt'.format(exp_dir_path, exp_name, model_name), 'w+') as f:
                        traceback.print_exc(file=f)
                    print('{}|{} ERROR: {}'.format(exp_name, model_name, e))
                    continue


def run_exp(dataset: ds.MICDataSet, model_param, ds_param=None, species=None, antibiotic=None,
            run_over=False):
    if type(species) == list:
        for species_j in species:
            if type(antibiotic) == list:
                for antibiotic_i in antibiotic:
                    run_exp(dataset, model_param, ds_param, species_j, antibiotic_i, run_over=run_over)
            else:
                run_exp(dataset, model_param, ds_param, species_j, antibiotic, run_over=run_over)
    else:
        if type(antibiotic) == list:
            for antibiotic_i in antibiotic:
                run_exp(dataset, model_param, ds_param, species, antibiotic_i, run_over=run_over)
        else:
            try:
                train, test, range_X, range_y, col_names, ds_param_files_path, species_name, antibiotic_name, cv = dataset.generate_dataset(
                    ds_param, species, antibiotic)

            except Exception as e:
                print(type(e))
                print(e)
                return -1

            exp_path = '../experiments/{}/{}/{}'.format(dataset.pre_params_name, dataset.name, '/'.join(ds_param_files_path.split('/')[-3::]))
            os.makedirs(exp_path, exist_ok=True)
            with open('{}/data_path.txt'.format(exp_path), "w") as data_path:
                data_path.write(ds_param_files_path)
            if len(train) < 40:
                with open('{}/tb.txt'.format(exp_path), 'w+') as f:
                    f.write('Training set doesnt have at-least 40 samples reqiered for training')
                    print('{} is too small, train size - {}'.format(exp_path, len(train)))
                    return -1
            model_name = '|'.join([':'.join([k, str(v)]) for k, v in model_param.items()])
            os.makedirs('{}/{}'.format(exp_path, model_name), exist_ok=True)
            pd.DataFrame(model_param, index=[0]).to_csv(
                '{}/{}/model_param.csv'.format(exp_path, model_name))

            if not run_over:
                if os.path.exists('{}/{}/tb.txt'.format(exp_path, model_name)) or \
                        os.path.exists('{}/{}/test_preds.csv'.format(exp_path, model_name)):
                    print('{}|{} was already run'.format(exp_path, model_name))
                    return 0
            try:
                if model_param['model'] == 'autoxgb':
                    run_autoxgb(exp_path, model_param, ds_param_files_path, col_names)
                elif model_param['model'] == 'h2o':
                    run_h2o(exp_path, model_param, ds_param_files_path, col_names)
                print('{}|{} done running exp'.format(exp_path, model_name))
                return 0
            except Exception as e:
                with open('{}/{}/tb.txt'.format(exp_path, model_name), 'w+') as f:
                    traceback.print_exc(file=f)
                print('{}|{} ERROR: {}'.format(exp_path, model_name, e))
                return -1


def main(args):
    pre_params = {}
    if args.filter_genome_size:
        pre_params['filter_genome_size'] = args.filter_genome_size
    if args.filter_contig_num:
        pre_params['filter_contig_num'] = args.filter_contig_num
    if args.cov_thresh:
        pre_params['cov_thresh'] = args.cov_thresh
    if args.id_thresh:
        pre_params['id_thresh'] = args.id_thresh
    if pre_params == {}:
        pre_params = None

    data = ds.CollectionDataSet(dbs_name_list=args.data_sets, pre_params=pre_params)

    ds_params = {}
    if args.handle_range:
        ds_params['handle_range'] = args.handle_range
    if args.move_range_by:
        ds_params['move_range_by'] = args.move_range_by
    if args.per_gene_features:
        ds_params['per_gene_features'] = args.per_gene_features
    if args.pca:
        ds_params['pca'] = args.pca
    if args.scalar:
        ds_params['scalar'] = args.scalar
    if args.not_equal_meaning:
        ds_params['not_equal_meaning'] = args.not_equal_meaning
    if ds_params == {}:
        ds_params = None

    model_params = {}
    model_params['model'] = args.model
    model_params['train_time'] = args.train_time
    model_params['max_models'] = args.max_models

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
    run_exp(data, model_params, ds_params, species=species_list, antibiotic=anti_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-over', dest='run_over', action="store_true")

    # dataset to load
    parser.add_argument('--data-sets', dest='data_sets', default=['PATAKI', 'VAMP', 'PA', 'PATRIC'], nargs='+')
    parser.add_argument('--id-thresh', dest='id_thresh', type=int, nargs='?')
    parser.add_argument('--cov-thresh', dest='cov_thresh', type=int, nargs='?')
    parser.add_argument('--filter-genome-size', dest='filter_genome_size', type=int, nargs='?')
    parser.add_argument('--filter-contig-num', dest='filter_contig_num', action="store_true")

    # pairs to generate X-y data for
    parser.add_argument('--species-list', dest='species_list', default=0, nargs='+')
    parser.add_argument('--anti-list', dest='anti_list', default=0, nargs='+')

    # ds parameters like range handling
    parser.add_argument('--handle-range', dest='handle_range',
                        choices=['remove', 'strip', 'move'])
    parser.add_argument('--per-gene-features', dest='per_gene_features', nargs='?')
    parser.add_argument('--move-range-by', dest='move_range_by', type=int, nargs='?')
    parser.add_argument('--not-equal-meaning', dest='not_equal_meaning', action="store_true")
    parser.add_argument('--pca', dest='pca', choices=['per_gene', 'all'])
    parser.add_argument('--scalar', dest='scalar', action="store_true")

    # pre parameters like geno thresholds

    # models to run
    parser.add_argument('--model', dest='model', default='autoxgb', type=str, nargs='?')
    parser.add_argument('--train-time', dest='train_time', default=1, type=int, nargs='?')
    parser.add_argument('--max-models', dest='max_models', default=1, type=int, nargs='?')

    args = parser.parse_args()
    main(args)
