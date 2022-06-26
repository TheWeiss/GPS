import os
import pandas as pd
import numpy as np
import json

def parse_results(exp_dir_path):
    exp_list =  []
    for path in os.listdir(exp_dir_path):
        if len(path.split('|')[0]) > 0:
            continue
        exp_list.append(path)
    results = pd.DataFrame({
        'exp_path': exp_list,
        'species': [path.split('|')[1] for path in exp_list],
        'antibiotic': [path.split('|')[2] for path in exp_list],
    })
    return results

def align_model_params_files(exp_path):
    full_path = '../experiments/{}'.format(exp_path)
    data_path = full_path + '/data_path.txt'
    if not os.path.exists(data_path):
        return 'missing data_path from the new format'
    for model_path in os.listdir(full_path):
        if os.path.isdir(full_path + '/' + model_path):
            if model_path == '.ipynb_checkpoints':
                continue
            if not os.path.exists('{}/{}/model_param.csv'.format(full_path, model_path)):
                model_param = {
                    'model': model_path.split('|')[0].split(':')[1],
                    'train_time': model_path.split('|')[1].split(':')[1],
                    'max_models': model_path.split('|')[2].split(':')[1],
                }
                pd.DataFrame(model_param, index=[0]).to_csv(
                    '{}/{}/model_param.csv'.format(full_path, model_path))


def align_error_files_files(exp_path):
    if os.path.exists('../experiments/{}/tb.txt'.format(exp_path)):
        if os.path.isdir('../experiments/{}/model:autoxgb|train_time:3600|max_models:100'.format(exp_path)):
            os.system("mv ../experiments/{}/tb.txt ../experiments/{}/model:autoxgb|train_time:3600|max_models:100/tb.txt".format(exp_path, exp_path))
            return 'moved'
        else:
            return 'no long exp found'
    return 'no error file found'


def fill_data_path(exp_path):
    data_file_path = '../experiments/{}/data_path.txt'.format(exp_path)
    with open(data_file_path) as f:
        return f.readlines()[0]


def fill_data_set(data_path):
    return data_path.split('/')[-4]


def fill_data_param(row):
    i = row.to_frame().T.index.values[0]
    data_param = pd.read_csv('{}/ds_param.csv'.format(row['data_path'])).drop('Unnamed: 0',axis=1)
    data_param.index = [i]
    tmp = pd.concat([row.to_frame().T, data_param], axis=1)
    tmp = tmp.iloc[0]
    return tmp


def fix_range_values(res):
    res.loc[res['handle_range']=='remove', 'move_range_by'] = None
    res.loc[res['handle_range'] == 'strip', 'move_range_by'] = 0
    return res


def compare_to_naive(res):
    res['learned_accuracy_train'] = res['accuracy_train'].div(res['accuracy_naive'].where(res['accuracy_naive'] != 0, np.nan))
    res['learned_accuracy_test'] = res['accuracy_test'].div(res['accuracy_naive'].where(res['accuracy_naive'] != 0, np.nan))
    res['learned_essential_agreement_train'] = res['essential_agreement_train'].div(res['essential_agreement_naive'].where(res['essential_agreement_naive'] != 0, np.nan))
    res['learned_essential_agreement_test'] = res['essential_agreement_test'].div(res['essential_agreement_naive'].where(res['essential_agreement_naive'] != 0, np.nan))
    res['learned_RMSE_train'] = res['exact_RMSE_train'].div(res['exact_RMSE_naive'].where(res['exact_RMSE_naive'] != 0, np.nan))
    res['learned_RMSE_test'] = res['exact_RMSE_test'].div(res['exact_RMSE_naive'].where(res['exact_RMSE_naive'] != 0, np.nan))
    return res


def fill_model_param(row):
    i = row.to_frame().T.index.values[0]
    exp_path = row['exp_path']
    model_params = pd.DataFrame({})
    if os.path.exists('../experiments/{}/tb.txt'.format(exp_path)):
        model_params = pd.DataFrame({
            'error': [True],
            'tb': [''],
        }, index=[i])
        with open('../experiments/{}/tb.txt'.format(exp_path)) as f:
            model_params.loc[i, 'tb'] = f.readlines()[0]
    for model_path in os.listdir('../experiments/{}'.format(exp_path)):
        if os.path.isdir('../experiments/{}/{}'.format(exp_path, model_path)):
            if os.path.exists('../experiments/{}/{}/model_param.csv'.format(exp_path, model_path)):
                model_param = pd.read_csv('../experiments/{}/{}/model_param.csv'.format(exp_path, model_path)).drop('Unnamed: 0',axis=1)
                model_param['model_path'] = model_path
                model_param.index = [i]
                if os.path.exists('../experiments/{}/{}/tb.txt'.format(exp_path, model_path)):
                    model_param['error'] = True
                    with open('../experiments/{}/{}/tb.txt'.format(exp_path, model_path)) as f:
                        model_param['tb'] = f.readlines()[0]
                else:
                    model_param['error'] = False
                    model_param['tb'] = None
                if len(model_params) == 0:
                    model_params = model_param
                else:
                    model_params = pd.concat([model_params, model_param], axis=0)
    tmp = row.to_frame().T.merge(right=model_params, left_index=True, right_index=True, how='outer')
    # print(tmp)
    return tmp


def read_exp_dirs(exp_dir_path):
    results = parse_results(exp_dir_path)
    results['exp_path'].apply(align_model_params_files)
    results['data_path'] = results['exp_path'].apply(fill_data_path)
    results['dataset'] = results['data_path'].apply(fill_data_set)
    results = results.apply(fill_data_param, axis=1)
    results = pd.concat(results.apply(fill_model_param, axis=1).values).reset_index(drop=True)
    results = add_metrices(results, equal_meaning=True)
    results = compare_to_naive(results)
    results = fix_range_values(results)
    return results


def add_metrices(res, equal_meaning=True, range_conf=False):
    results = res.copy()
    for i in np.arange(len(results)):
        try:
            exp_name = results['exp_path'].iloc[i]
            model_name = results['model_path'].iloc[i]
            data_path = results['data_path'].iloc[i]
            if results['error'].iloc[i]:
                continue
            with open(data_path + '/col_names.json') as json_file:
                col_names = json.load(json_file)
            range_y = pd.read_csv('{}/range_y.csv'.format(data_path)).set_index(col_names['id'])
            range_y.columns = ['y_true', 'sign']
            train_y = pd.read_csv('{}/{}.csv'.format(data_path, 'train')).rename(columns={"Unnamed: 0": col_names['id']})[
                [
                    col_names['id'],
                    col_names['label'],
                ]].set_index(col_names['id'])
            train_y.columns = ['y_true']
            train_indexs = train_y.index
            train_y = train_y.loc[set(train_indexs) - set(range_y.index)]
            test_y = pd.read_csv('{}/{}.csv'.format(data_path, 'test')).rename(columns={"Unnamed: 0": col_names['id']})[
                [
                    col_names['id'],
                    col_names['label'],
                ]].set_index(col_names['id'])
            test_y.columns = ['y_true']
            test_indexs = test_y.index
            test_y = test_y.loc[set(test_indexs) - set(range_y.index)]
            y = pd.concat([range_y, train_y, test_y], axis=0)
            mode = y['y_true'].mode().values[0]
            mean = y['y_true'].mean()
            y = pd.concat([train_y, test_y], axis=0)

            y['naive_residual_mode'] = y['y_true'] - mode
            y['naive_residual_mean'] = y['y_true'] - mean
            y['naive_error'] = y['naive_residual_mode'].abs() < 1
            y['naive_error2'] = y['naive_residual_mode'].abs() < 2

            split_res = {}
            for split in ['train', 'test']:
                split_y = pd.read_csv('{}/{}.csv'.format(data_path, split)).rename(columns={"Unnamed: 0": col_names['id']})[
                    [
                        col_names['id'],
                        col_names['label'],
                    ]].set_index(col_names['id'])
                split_y.columns = ['y_true']
                if results['model'].iloc[i] == 'autoxgb':
                    split_preds = pd.read_csv(
                        '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_name, split)).set_index(col_names['id'])
                    split_preds.columns = ['y_pred']
                    split_res_i = split_preds.merge(split_y, left_index=True, right_index=True, how='inner')
                elif results['model'].iloc[i] == 'h2o':
                    split_preds = pd.read_csv(
                        '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_name, split)).drop('Unnamed: 0', axis=1)
                    split_preds.columns = ['y_pred']
                    split_res_i = split_preds.merge(split_y.reset_index(), left_index=True, right_index=True,
                                                    how='inner').set_index(col_names['id'])
                split_res_i = split_res_i.loc[set(split_res_i.index) - set(range_y.index)]

                split_res_i['y_true'] = np.round(split_res_i['y_true'])
                min_true = split_res_i['y_true'].min()
                max_true = split_res_i['y_true'].max(axis=0)
                split_res_i['y_pred'] = split_res_i['y_pred'].clip(lower=min_true, upper=max_true)
                split_res_i['residual'] = split_res_i['y_true'] - split_res_i['y_pred']
                split_res_i['y_pred'] = np.round(split_res_i['y_pred'])
                split_res_i['round_residual'] = split_res_i['y_true'] - split_res_i['y_pred']
                split_res_i['error'] = split_res_i['round_residual'].abs() < 1
                split_res_i['error2'] = split_res_i['round_residual'].abs() < 2
                split_res[split] = split_res_i

            regression_res = pd.DataFrame({
                'exact_RMSE': [np.sqrt(split_data['residual'].pow(2).mean()) for split_data in split_res.values()],
                'exact_rounded_RMSE': [np.sqrt(split_data['round_residual'].pow(2).mean()) for split_data in
                                       split_res.values()],
                'exact_accuracy': [split_data['error'].mean() for split_data in split_res.values()],
                'exact_accuracy2': [split_data['error2'].mean() for split_data in split_res.values()],
            }, index=['train', 'test'])

            if results['model'].iloc[i] == 'autoxgb':
                print('xgb')
                range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
                if len(range_preds) == 0:
                    range_preds = pd.DataFrame({col_names['id']: [], 'measurment': []}, index=[])
                range_preds = range_preds.set_index(col_names['id'])
                range_preds.columns = ['y_pred']
                print(range_y)
                print(range_preds)
                range_res = range_preds.merge(range_y, left_index=True, right_index=True, how='inner')
                print(range_res)
            elif results['model'].iloc[i] == 'h2o':
                print('h2o')
                range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name)).drop('Unnamed: 0', axis=1)
                if len(range_preds) == 0:
                    range_preds = pd.DataFrame({'measurment': []}, index=[])
                range_preds.columns = ['y_pred']
                print(range_y)
                print(range_preds)
                range_res = range_preds.merge(range_y.reset_index(), left_index=True, right_index=True, how='inner')
                print(range_res)

            range_res['y_true'] = np.round(range_res['y_true'])
            range_res['updated_y_true'] = np.nan
            range_res['updated_sign'] = np.nan
            if not equal_meaning:
                range_res.loc[range_res['sign'] == '>=', 'updated_y_true'] = range_res['y_true'] - 1
                range_res.loc[range_res['sign'] == '<=', 'updated_y_true'] = range_res['y_true'] + 1
            range_res.loc[range_res['sign'] == '>=', 'updated_sign'] = '>'
            range_res.loc[range_res['sign'] == '<=', 'updated_sign'] = '<'
            range_res.loc[:, 'updated_y_true'].fillna(range_res['y_true'], inplace=True)
            range_res.loc[:, 'updated_sign'].fillna(range_res['sign'], inplace=True)

            range_res.loc[range_res['updated_sign'] == '>', 'error'] = (
                    range_res['y_pred'] > range_res['updated_y_true'])
            range_res.loc[range_res['updated_sign'] == '<', 'error'] = (
                    range_res['y_pred'] < range_res['updated_y_true'])
            range_res.loc[range_res['updated_sign'] == '>', 'error2'] = (
                    range_res['y_pred'] > range_res['updated_y_true'] - 1)
            range_res.loc[range_res['updated_sign'] == '<', 'error2'] = (
                    range_res['y_pred'] < range_res['updated_y_true'] + 1)


            train_range_res = range_res.loc[set(range_res.index).intersection(set(train_indexs))]
            test_range_res = range_res.loc[set(range_res.index) - set(train_indexs)]

            if range_conf:
                for key, res in {'train': train_range_res, 'test': test_range_res}.items():
                    range_confusion = res.groupby(by=['sign', 'y_true'])['error'].agg(['count', 'sum']).replace(True, 1)
                    range_confusion['perc'] = range_confusion['sum'] / range_confusion['count']
                    range_confusion.columns = ['range_total', 'range_true', 'range_accuracy']
                    range_confusion = pd.DataFrame(range_confusion.stack()).T.swaplevel(i=2, j=0, axis=1)
                    range_confusion.index = [key]
                    regression_res = pd.concat([regression_res, range_confusion], axis=1)

            regression_res_cleaned = pd.DataFrame({})
            for col in regression_res.columns:
                if len(regression_res[[col]].columns) > 1:
                    regression_res_cleaned[col] = regression_res[[col]].iloc[:, 0].fillna(
                        regression_res[[col]].iloc[:, 1])
                else:
                    regression_res_cleaned[col] = regression_res[[col]]
            regression_res = regression_res_cleaned
            regression_res['range_accuracy'] = [
                train_range_res['error'].mean(),
                test_range_res['error'].mean(),
            ]
            regression_res['range_accuracy'].fillna(0, inplace=True)

            regression_res['range_accuracy2'] = [
                train_range_res['error2'].mean(),
                test_range_res['error2'].mean(),
            ]
            regression_res['range_accuracy2'].fillna(0, inplace=True)


            regression_res['range_size'] = [
                len(train_range_res),
                len(test_range_res),
            ]
            regression_res['range_size'].fillna(0, inplace=True)
            regression_res['exact_size'] = [len(split_data) for split_data in split_res.values()]
            regression_res['exact_size'].fillna(0, inplace=True)
            regression_res['accuracy'] = (regression_res['exact_accuracy'].fillna(0) * regression_res[
                'exact_size'].fillna(0) \
                                          + regression_res['range_accuracy'] * regression_res['range_size']) \
                                         / (regression_res['range_size'] + regression_res['exact_size'].fillna(0))


            regression_res['essential_agreement'] = (regression_res['exact_accuracy2'].fillna(0) * regression_res[
                'exact_size'].fillna(0) \
                                                     + regression_res['range_accuracy2'] * regression_res['range_size']) \
                                                    / (regression_res['range_size'] + regression_res[
                'exact_size'].fillna(0))


            regression_res = pd.DataFrame(regression_res.unstack()).T

            regression_res.columns = ['{}_{}'.format(col[0], col[1])
                                      for col in regression_res.columns]
            regression_res.index = [i]

            regression_res['exact_RMSE_naive'] = np.sqrt(y['naive_residual_mean'].pow(2).mean())
            regression_res['exact_accuracy_naive'] = y['naive_error'].mean()
            regression_res['exact_accuracy2_naive'] = y['naive_error2'].mean()

            range_res.loc[range_res['updated_sign'] == '>', 'naive_error'] = (
                    mode >= range_res['updated_y_true'])
            range_res.loc[range_res['updated_sign'] == '<', 'naive_error'] = (
                    mode <= range_res['updated_y_true'])
            regression_res['range_accuracy_naive'] = range_res['naive_error'].mean()
            range_res.loc[range_res['updated_sign'] == '>', 'naive_error2'] = (
                    mode >= range_res['updated_y_true'] - 1)
            range_res.loc[range_res['updated_sign'] == '<', 'naive_error2'] = (
                    mode <= range_res['updated_y_true'] + 1)
            regression_res['range_accuracy2_naive'] = range_res['naive_error2'].mean()
            regression_res['exact_size'] = regression_res['exact_size_train'] + regression_res['exact_size_test']
            regression_res['range_size'] = regression_res['range_size_train'] + regression_res['range_size_test']

            regression_res['accuracy_naive'] = (regression_res['exact_accuracy_naive'].fillna(0) * regression_res[
                'exact_size'].fillna(0) \
                                                + regression_res['range_accuracy_naive'].fillna(0) * regression_res['range_size'].fillna(0)) \
                                               / (regression_res['range_size'].fillna(0) + regression_res['exact_size'].fillna(0))

            regression_res['essential_agreement_naive'] = (regression_res['exact_accuracy2_naive'].fillna(0) *
                                                           regression_res[
                                                               'exact_size'].fillna(0) \
                                                           + regression_res['range_accuracy2_naive'].fillna(0) * regression_res[
                                                               'range_size'].fillna(0)) \
                                                          / (regression_res['range_size'].fillna(0) + regression_res[
                'exact_size'].fillna(0))

            regression_res['size'] = regression_res['exact_size'] + regression_res['range_size']
            regression_res['exp_done'] = True
        except (FileNotFoundError, OSError):
            regression_res = pd.DataFrame({}, index=[0])
            regression_res['exp_done'] = False
        regression_res.index = [i]
        results = pd.concat([results, pd.DataFrame(columns=regression_res.columns)])
        results.update(regression_res)
    return results