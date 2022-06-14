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
    data_path = full_path +'/data_path.txt'
    if not os.path.exists(data_path):
        return 'missing data_path from the new format'
    for model_path in os.listdir(full_path):
        if os.path.isdir(full_path + '/' + model_path):
            if not os.path.exists('{}/{}/model_param.csv'.format(full_path, model_path)):
                model_param = {
                    'model': model_path.split('|')[0].split(':')[1],
                    'train_time': model_path.split('|')[1].split(':')[1],
                    'max_models': model_path.split('|')[2].split(':')[1],
                }

                pd.DataFrame(model_param, index=[0]).to_csv(
                            '{}/{}/model_param.csv'.format(full_path, model_path))
            else:
                return 'alreadt exist'
    return 'filled model_param_file'


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


def fill_model_param(row):
    i = row.to_frame().T.index.values[0]
    exp_path = row['exp_path']
    model_params = pd.DataFrame({})
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
    results = add_exact_param_metrices(results, equal_meaning=True)
    results['exact_size'] = results['exact_size_train'] + results['exact_size_test']
    results['range_size'] = results['range_size_train'] + results['range_size_test']
    results['size'] = results['exact_size'] + results['range_size']
    return results


def add_exact_metrices(results, equal_meaning=True):
    for i in np.arange(len(results)):
        try:
            exp_name = results['exp_path'].iloc[i]
            if results['dup_drop'].iloc[i]:
                id_col = 'biosample_id'
            else:
                id_col = 'unique_id'
            label = pd.read_csv('../resources/label_{}.csv'.format(exp_name)).loc[0, 'label']
            y_range = pd.read_csv('../resources/y_range_{}.csv'.format(exp_name)).set_index(id_col)
            y = pd.read_csv('../resources/train_{}.csv'.format(exp_name)).set_index(id_col)[label]

            train_res = pd.read_csv('../experiments/{}/oof_predictions.csv'.format(exp_name)).set_index(id_col).merge(y,
                                                                                                                      left_index=True,
                                                                                                                      right_index=True,
                                                                                                                      how='inner')
            train_res = train_res.loc[set(train_res.index) - set(y_range.index)]
            train_res.columns = ['y_pred', 'y_true']
            train_res['y_true'] = np.round(train_res['y_true'])
            min_true = train_res['y_true'].min()
            max_true = train_res['y_true'].max(axis=0)
            train_res['y_pred'] = train_res['y_pred'].clip(lower=min_true, upper=max_true)
            train_res['residual'] = train_res['y_true'] - train_res['y_pred']
            train_res['y_pred'] = np.round(train_res['y_pred'])
            train_res['round_residual'] = train_res['y_true'] - train_res['y_pred']
            train_res['error'] = train_res['round_residual'].abs() < 1
            train_res['error2'] = train_res['round_residual'].abs() < 2

            y = pd.read_csv('../resources/test_{}.csv'.format(exp_name)).set_index(id_col)[label]
            test_res = pd.read_csv('../experiments/{}/test_predictions.csv'.format(exp_name)).set_index(id_col).merge(y,
                                                                                                                      left_index=True,
                                                                                                                      right_index=True,
                                                                                                                      how='inner')
            test_res = test_res.loc[set(test_res.index) - set(y_range.index)]
            test_res.columns = ['y_pred', 'y_true']
            test_res['y_true'] = np.round(test_res['y_true'])
            min_true = test_res['y_true'].min()
            max_true = test_res['y_true'].max(axis=0)
            test_res['y_pred'] = test_res['y_pred'].clip(lower=min_true, upper=max_true)
            test_res['residual'] = test_res['y_true'] - test_res['y_pred']
            test_res['y_pred'] = np.round(test_res['y_pred'])
            test_res['round_residual'] = test_res['y_true'] - test_res['y_pred']
            test_res['error'] = test_res['round_residual'].abs() < 1
            test_res['error2'] = test_res['round_residual'].abs() < 2

            regression_res = pd.DataFrame({
                'exact RMSE': [np.sqrt(train_res['residual'].pow(2).mean()),
                               np.sqrt(test_res['residual'].pow(2).mean())],
                'exact_rounded RMSE': [np.sqrt(train_res['round_residual'].pow(2).mean()),
                                       np.sqrt(test_res['round_residual'].pow(2).mean())],
                'exact_accuracy': [train_res['error'].mean(), test_res['error'].mean()],
                'exact_accuracy2': [train_res['error2'].mean(), test_res['error2'].mean()],
            }, index=['train', 'test'])

            range_res = pd.read_csv('../experiments/{}/range_preds.csv'.format(exp_name)).set_index(id_col).merge(
                y_range, left_index=True, right_index=True, how='inner')
            range_res.columns = ['y_pred'] + list(range_res.columns.values)[1:]
            range_res['values'] = np.round(range_res['values'])
            range_res['updated_values'] = np.nan
            range_res['updated_direction'] = np.nan
            if equal_meaning:
                range_res.loc[range_res['direction'] == '>=', 'updated_values'] = range_res['values'] - 1
                range_res.loc[range_res['direction'] == '<=', 'updated_values'] = range_res['values'] + 1
            range_res.loc[range_res['direction'] == '>=', 'updated_direction'] = '>'
            range_res.loc[range_res['direction'] == '<=', 'updated_direction'] = '<'

            range_res.loc[:, 'updated_values'].fillna(range_res['values'], inplace=True)
            range_res.loc[:, 'updated_direction'].fillna(range_res['direction'], inplace=True)

            range_res.loc[range_res['updated_direction'] == '>', 'error'] = (
                        range_res['y_pred'] > range_res['updated_values'])
            range_res.loc[range_res['updated_direction'] == '<', 'error'] = (
                        range_res['y_pred'] < range_res['updated_values'])
            range_res.loc[range_res['updated_direction'] == '>', 'error2'] = (
                        range_res['y_pred'] > range_res['updated_values'] - 1)
            range_res.loc[range_res['updated_direction'] == '<', 'error2'] = (
                        range_res['y_pred'] < range_res['updated_values'] + 1)
            y = pd.read_csv('../resources/train_{}.csv'.format(exp_name)).set_index(id_col)[label]
            train_res_index = pd.read_csv('../experiments/{}/oof_predictions.csv'.format(exp_name)).set_index(
                id_col).merge(y, left_index=True, right_index=True, how='inner').index
            train_range_res = range_res.loc[set(range_res.index).intersection(set(train_res_index))]
            test_range_res = range_res.loc[set(range_res.index) - set(train_res_index)]
            for key, res in {'train': train_range_res, 'test': test_range_res}.items():
                range_confusion = res.groupby(by=['direction', 'values'])['error'].agg(['count', 'sum']).replace(True,
                                                                                                                 1)
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
            regression_res['exact_size'] = [
                len(train_res),
                len(test_res),
            ]
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
            regression_res['exp_done'] = True
        except:
            regression_res = pd.DataFrame({}, index=[0])
            regression_res['exp_done'] = False
        regression_res.index = [i]
        results = pd.concat([results, pd.DataFrame(columns=regression_res.columns)])
        results.update(regression_res)
    return results


def add_exact_param_metrices(res, equal_meaning=True):
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
                    print('h2o not implemented yet')

                split_res_i.loc[set(split_res_i.index) - set(range_y.index)]

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
                'exact RMSE': [np.sqrt(split_data['residual'].pow(2).mean()) for split_data in split_res.values()],
                'exact_rounded RMSE': [np.sqrt(split_data['round_residual'].pow(2).mean()) for split_data in
                                       split_res.values()],
                'exact_accuracy': [split_data['error'].mean() for split_data in split_res.values()],
                'exact_accuracy2': [split_data['error2'].mean() for split_data in split_res.values()],
            }, index=['train', 'test'])

            if results['model'].iloc[i] == 'autoxgb':
                range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
                if len(range_preds) == 0:
                    range_preds = pd.DataFrame({col_names['id']: [], 'measurment': []}, index=[])
                range_preds = range_preds.set_index(col_names['id'])
                range_preds.columns = ['y_pred']
                range_res = range_preds.merge(range_y, left_index=True, right_index=True, how='inner')
            elif results['model'].iloc[i] == 'h2o':
                print('h2o not implemented yet')

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

            train_range_res = range_res.loc[set(range_res.index).intersection(set(split_res['train'].index))]
            test_range_res = range_res.loc[set(range_res.index) - set(split_res['train'].index)]

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
            regression_res['exp_done'] = True
        except (FileNotFoundError, Exception):
            regression_res = pd.DataFrame({}, index=[0])
            regression_res['exp_done'] = False
        regression_res.index = [i]
        results = pd.concat([results, pd.DataFrame(columns=regression_res.columns)])
        results.update(regression_res)
    return results