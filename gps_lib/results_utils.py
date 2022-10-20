import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from experiment import Model_h2o, Model_axgb
import argparse

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


def results_by(results, metric, ascending=False):
    filtered_res = results.sort_values(ascending=ascending, by='{}_test'.format(metric)).drop_duplicates(subset=['species', 'antibiotic'], keep='first')
    filtered_res = filtered_res.sort_values(ascending=ascending, by='{}_test'.format(metric)).reset_index(drop=True)
    return filtered_res


def print_results_by(results, sort_metric, metrices, infos, ascending=False):
    if sort_metric in metrices:
        metrices.remove(sort_metric)
    fig, axis = plt.subplots(len(metrices) + len(infos) + 1 + 1, sharex=True,
                             figsize=(15, 4 * (len(metrices) + len(infos) + 1)))
    fig.suptitle('performance sorted by {}'.format(sort_metric))
    results_by(results, sort_metric, ascending)['{}_train'.format(sort_metric)].plot(legend='train', ax=axis[0],
                                                                                     grid=True)
    results_by(results, sort_metric, ascending)['{}_test'.format(sort_metric)].plot(legend='test', ax=axis[0],
                                                                                    grid=True)
    try:
        results_by(results, sort_metric, ascending)['{}_naive'.format(sort_metric)].plot(legend='naive', ax=axis[0],
                                                                                         grid=True)
    except:
        pass
    size_infos = ['exact_size', 'range_size']
    results_by(results, sort_metric, ascending)[size_infos].plot.area(legend=size_infos, ax=axis[1], grid=True)

    for i, info in enumerate(infos):
        results_by(results, sort_metric, ascending)[info].plot(legend=info, ax=axis[i + 2], grid=True)

    for i, met in enumerate(metrices):
        results_by(results, sort_metric, ascending)['{}_train'.format(met)].plot(legend='test',
                                                                                 ax=axis[i + 2 + len(infos)], grid=True)
        results_by(results, sort_metric, ascending)['{}_test'.format(met)].plot(legend='test',
                                                                                ax=axis[i + 2 + len(infos)], grid=True)
        try:
            results_by(results, sort_metric, ascending)['{}_naive'.format(met)].plot(legend='naive',
                                                                                     ax=axis[i + 2 + len(infos)],
                                                                                     grid=True)
        except:
            pass
    plt.show()


def get_exp_id_by_criterion(results, sort_metric, ascending=False, get_next=0):
    criterion = sort_metric #'exact_RMSE'

    criterion = criterion+'_test'
    if ascending:
        accuracy_score = results.groupby(['species', 'antibiotic'])[criterion].min()
        species, antibiotic = accuracy_score.sort_values(ascending=True).index[get_next]
    else:
        accuracy_score = results.groupby(['species', 'antibiotic'])[criterion].max()
        species, antibiotic = accuracy_score.sort_values(ascending=False).index[get_next]
    i = results[np.logical_and(results['species'] == species, results['antibiotic'] == antibiotic)].sort_values(
        by=criterion, ascending=ascending).iloc[0].dropna().name
    return i


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
    results.to_csv('{}/results_summery.csv'.format(exp_dir_path), index=True)
    return results


def exact_plots(i):
    res = pd.read_csv('../experiments/results_summery.csv')
    exp_name = res.loc[i, 'exp_path']
    model_path = res.loc[i, 'model_path']
    model = res.loc[i, 'model']
    data_path = res.loc[i, 'data_path']
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
        if model == 'autoxgb':
            split_preds = pd.read_csv(
                '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_path, split)).set_index(col_names['id'])
            split_preds.columns = ['y_pred']
            split_res_i = split_preds.merge(split_y, left_index=True, right_index=True, how='inner')
        elif model == 'h2o':
            split_preds = pd.read_csv(
                '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_path, split)).drop('Unnamed: 0', axis=1)
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

    tics = np.sort(list(set(list(np.round(split_res['train']['y_true']).unique())).union(
        set(list(np.round(split_res['train']['y_pred']).unique())))))

    N = len(tics)

    for key, fold in split_res.items():

        title = 'Exact confusion matrix of the pair ({},{})- {}'.format(res.loc[i, 'species'], res.loc[i, 'antibiotic'], key)
        # for title, normalize in titles_options:
        plt.figure(figsize=(13, 13))

        # Generate the confusion matrix
        cf_matrix = confusion_matrix(np.round(fold['y_true']), np.round(fold['y_pred']), labels=tics)
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

        cf_matrix = confusion_matrix(np.round(fold['y_true']), np.round(fold['y_pred']), normalize='true', labels=tics)
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()]

        labels = [f"{v1}\n({v2})" for v1, v2 in
                  zip(group_percentages, group_counts)]

        labels = np.asarray(labels).reshape(N, N)

        ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        # ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

        ax.set_title(title);
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(tics)
        ax.yaxis.set_ticklabels(tics)
        plt.savefig('../experiments/{}/{}/exact_conf_mat_{}'.format(exp_name, model_path, key))
        ## Display the visualization of the Confusion Matrix.
        plt.show()

    plt.show()


def PA_plot(i, threshold):
    res = pd.read_csv('../experiments/results_summery.csv')
    exp_name = res.loc[i, 'exp_path']
    model_path = res.loc[i, 'model_path']
    model = res.loc[i, 'model']
    data_path = res.loc[i, 'data_path']
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
        if model == 'autoxgb':
            split_preds = pd.read_csv(
                '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_path, split)).set_index(col_names['id'])
            split_preds.columns = ['y_pred']
            split_res_i = split_preds.merge(split_y, left_index=True, right_index=True, how='inner')
        elif model == 'h2o':
            split_preds = pd.read_csv(
                '../experiments/{}/{}/{}_preds.csv'.format(exp_name, model_path, split)).drop('Unnamed: 0', axis=1)
            split_preds.columns = ['y_pred']
            split_res_i = split_preds.merge(split_y.reset_index(), left_index=True, right_index=True,
                                            how='inner').set_index(col_names['id'])
        split_res_i = split_res_i.loc[set(split_res_i.index) - set(range_y.index)]

        split_res_i['y_true'] = np.round(split_res_i['y_true'])
        split_res_i['y_true_side'] = np.sign(split_res_i['y_true'] - threshold)

        min_true = split_res_i['y_true'].min()
        max_true = split_res_i['y_true'].max(axis=0)
        split_res_i['y_pred'] = split_res_i['y_pred'].clip(lower=min_true, upper=max_true)
        split_res_i['y_pred_side'] = np.sign(split_res_i['y_pred'] - threshold)

        split_res_i['error'] = split_res_i['y_true_side'] == split_res_i['y_pred_side']
        split_res_i['error'].replace(True, 'correctly_classified', inplace=True)
        split_res_i['error'].replace(False, 'misclassified', inplace=True)
        split_res_i.loc[split_res_i['y_true_side'] == 0, 'error'] = 'intermidiate_(not_classified)'

        if split == 'train':
            low = np.round(split_res_i['y_true'].min())
            high = np.round(split_res_i['y_true'].max())

        hist_range = np.arange(low - 0.5, high + 1, 1)
        bins_count = pd.DataFrame(
            split_res_i.groupby(by='error')['y_true'].apply(lambda x: np.histogram(x, bins=hist_range)[0]))

        tmp = pd.DataFrame(
            {'y_true': [np.zeros(len(hist_range) - 1)]},
            index=['correctly_classified', 'misclassified', 'intermidiate_(not_classified)']
        )
        bins_count = pd.concat([bins_count, tmp], axis=0)
        bins_count = bins_count[~bins_count.index.duplicated(keep='first')]
        bins_count = bins_count.loc[['correctly_classified', 'misclassified', 'intermidiate_(not_classified)']]

        # bins_count = bins_count.join(pd.DataFrame({'fill': [np.zeros(len(hist_range)-1)]},
        #                                            index=['correctly_classified', 'misclassified', 'intermidiate (not classified)']),
        #                               left_index=True, right_index=True, how='right')
        # print(bins_count)

        # bins_count['y_true'].fillna(bins_count['fill'], inplace=True)
        pd.DataFrame(bins_count['y_true'].tolist(), index=bins_count.index, columns=hist_range[:-1] + 0.5).T.plot.bar(
            stacked=True, figsize=(10, 6), color=['g', 'r', 'purple'])
        plt.title('SIR based on MIC predictionof the pair ({},{}) - {}'.format(res.loc[i, 'species'], res.loc[i, 'antibiotic'], split))
        plt.xlabel('log2(mg//L)')
        plt.ylabel('#')
        plt.savefig('../experiments/{}/{}/SIR_inference_{}'.format(exp_name, model_path, split))
        plt.show()


def range_plots(i):
    res = pd.read_csv('../experiments/results_summery.csv')
    exp_name = res.loc[i, 'exp_path']
    model = res.loc[i, 'model']
    model_path = res.loc[i, 'model_path']
    data_path = res.loc[i, 'data_path']

    equal_meaning = True
    with open(data_path + '/col_names.json') as json_file:
        col_names = json.load(json_file)
    range_y = pd.read_csv('{}/range_y.csv'.format(data_path)).set_index(col_names['id'])
    range_y.columns = ['y_true', 'sign']

    split_idx = {}
    for split in ['train', 'test']:
        split_y = pd.read_csv('{}/{}.csv'.format(data_path, split)).rename(columns={"Unnamed: 0": col_names['id']})[
            [
                col_names['id'],
                col_names['label'],
            ]].set_index(col_names['id'])
        split_idx[split] = split_y.index

    if model == 'autoxgb':
        range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_path))
        if len(range_preds) == 0:
            range_preds = pd.DataFrame({col_names['id']: [], 'measurment': []}, index=[])
        range_preds = range_preds.set_index(col_names['id'])
        range_preds.columns = ['y_pred']
        range_res = range_preds.merge(range_y, left_index=True, right_index=True, how='inner')
    elif model == 'h2o':
        range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_path)).drop(
            'Unnamed: 0', axis=1)
        if len(range_preds) == 0:
            range_preds = pd.DataFrame({'measurment': []}, index=[])
        range_preds.columns = ['y_pred']
        range_res = range_preds.merge(range_y.reset_index(), left_index=True, right_index=True, how='inner').set_index(
            col_names['id'])

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

    train_range_res = range_res.loc[set(range_res.index).intersection(set(split_idx['train']))]
    test_range_res = range_res.loc[set(range_res.index) - set(split_idx['train'])]

    for key, split_res in {'train': train_range_res, 'test': test_range_res}.items():
        range_confusion = split_res.groupby(by=['y_true', 'sign'])['error'].agg(['count', 'sum']).replace(True, 1)
        range_confusion['perc'] = 100 * range_confusion['sum'] / range_confusion['count']
        range_confusion.columns = ['range_total', 'correctly_classified', 'range_accuracy']
        range_confusion['correctly_classified'] = range_confusion['correctly_classified'].astype(int)
        range_confusion['misclassified'] = range_confusion['range_total'] - range_confusion['correctly_classified']

        ax = range_confusion[['correctly_classified', 'misclassified']].plot.bar(stacked=True, figsize=(10, 6),
                                                                                 color=['g', 'r'])
        ax.set_ylabel("#Isolates")
        ax2 = range_confusion[['range_accuracy']].plot(ax=ax.twinx(), color='blue', marker="o")
        ax2.set_title('Range results of the pair ({},{}) - {}'.format(res.loc[i, 'species'], res.loc[i, 'antibiotic'], key))
        ax2.set_ylabel("Accuracy [%]", color="blue", fontsize=14)
        ax2.set_ylim((0, 101))
        plt.savefig('../experiments/{}/{}/range_conf_mat_{}'.format(exp_name, model_path, key))
        plt.show()


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
                range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name))
                if len(range_preds) == 0:
                    range_preds = pd.DataFrame({col_names['id']: [], 'measurment': []}, index=[])
                range_preds = range_preds.set_index(col_names['id'])
                range_preds.columns = ['y_pred']
                range_res = range_preds.merge(range_y, left_index=True, right_index=True, how='inner')
            elif results['model'].iloc[i] == 'h2o':
                range_preds = pd.read_csv('../experiments/{}/{}/range_preds.csv'.format(exp_name, model_name)).drop('Unnamed: 0', axis=1)
                if len(range_preds) == 0:
                    range_preds = pd.DataFrame({'measurment': []}, index=[])
                range_preds.columns = ['y_pred']
                range_res = range_preds.merge(range_y.reset_index(), left_index=True, right_index=True, how='inner').set_index(col_names['id'])

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
            regression_res['stacked'] = False
        except (FileNotFoundError, OSError):
            regression_res = pd.DataFrame({}, index=[0])
            regression_res['exp_done'] = False
        regression_res.index = [i]
        results = pd.concat([results, pd.DataFrame(columns=regression_res.columns)])
        results.update(regression_res)
    return results


def shap_plots(i):
    res = pd.read_csv('../experiments/results_summery.csv')
    exp_name = res.loc[i, 'exp_path']
    model_type = res.loc[i, 'model']
    model_name = res.loc[i, 'model_path']
    data_path = res.loc[i, 'data_path']
    with open(data_path + '/col_names.json') as json_file:
        col_names = json.load(json_file)
    if os.path.exists('../experiments/{}/{}/shap_summery.png'.format(exp_name, model_name)):
        return pd.read_csv('../experiments/{}/{}/shap_values.csv'.format(exp_name, model_name))

    if model_type == 'h2o':
        model = Model_h2o(exp_name, model_name, data_path)
    elif model_type == 'autoxgb':
        model = Model_axgb(exp_name, model_name, data_path)
    else:
        print('model_type not supported: {}'.format(model_type))
    X = model.get_test()
    explainer = shap.KernelExplainer(model=model.predict, data=X)
    shap_values = explainer.shap_values(X=X)
    if len(shap_values.shape) > 2:
        shap_values = shap_values[1]
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    shap_df.to_csv('../experiments/{}/{}/shap_values.csv'.format(exp_name, model_name), index=False)
    shap.initjs()
    shap.summary_plot(shap_values=shap_values,
                      features=X, show=False)
    plt.savefig('../experiments/{}/{}/shap_summery'.format(exp_name, model_name), bbox_inches='tight')
    plt.show()
    return shap_values


def run_plots_single(species, antibiotic, criterion, ascending, plots, exp_dir_path):
    results = pd.read_csv('{}results_summery.csv'.format(exp_dir_path)).drop('Unnamed: 0', axis=1)

    results = results[results['size'] > 100][results['exact_size'] > 50]
    # results = results[results['learned_essential_agreement_test'] > 1.05][results['learned_RMSE_test'] < 0.95]
    if type(species) == int:
        if type(antibiotic) == int:
            i = get_exp_id_by_criterion(results, criterion, ascending, get_next=0)
        else:
            i = get_exp_id_by_criterion(results[results['antibiotic'] == antibiotic], criterion, ascending, get_next=species)
    else:
        if type(antibiotic) == int:
            i = get_exp_id_by_criterion(results[results['species'] == species], criterion, ascending, get_next=antibiotic)
        else:
            i = results[np.logical_and(results['species'] == species, results['antibiotic'] == antibiotic)].sort_values(
                by=criterion, ascending=ascending).iloc[0].dropna().name
    print(results.loc[i, 'exp_path'])
    for plot_func in plots:
        plot_func(i)
    print('Done {}'.format(results.loc[i, 'exp_path']))



def run_plots(species, antibiotic, criterion, ascending, plots, exp_dir_path='../experiments/'):
    if type(species) == list:
        for species_j in species:
            if type(antibiotic) == list:
                for antibiotic_i in antibiotic:
                    run_plots(species_j, antibiotic_i, criterion, ascending, plots, exp_dir_path)
            else:
                run_plots(species_j, antibiotic, criterion, ascending, plots, exp_dir_path)
    else:
        if type(antibiotic) == list:
            for antibiotic_i in antibiotic:
                run_plots(species, antibiotic_i, criterion, ascending, plots, exp_dir_path)
        else:
            run_plots_single(species, antibiotic, criterion, ascending, plots, exp_dir_path)


def main(args):
    plots = []
    if args.shap:
        plots.append(shap_plots)
    if args.range:
        plots.append(range_plots)
    if args.pa:
        plots.append(PA_plot)
    if args.exact:
        plots.append(exact_plots)

    criterion_ascending = {
        'accuracy': False,
        'learned_accuracy': False,
        'exact_accuracy': False,
        'exact_accuracy2': False,
        'range_accuracy': False,
        'range_accuracy2': False,
        'essential_agreement': False,
        'learned_essential_agreement': False,
        'exact_RMSE': True,
        'learned_RMSE': True,
    }
    if args.criterion:
        criterion = args.criterion
        ascending = criterion_ascending[criterion]
        print('criterion: {}, ascending: {}'.format(criterion, ascending))


    if type(args.anti_list) == list:
        anti_list = [int(anti) if anti.isnumeric() else anti for anti in args.anti_list]
    else:
        if args.anti_list.isnumeric():
            anti_list = int(args.anti_list)
        else:
            anti_list = args.anti_list

    if type(args.species_list) == list:
        species_list = [int(species) if species.isnumeric() else ' '.join(species.split('_')) for species in args.species_list]
    else:
        if args.species_list.isnumeric():
            species_list = int(args.species_list)
        else:
            species_list = args.species_list

    print('anti_list: {}, species_list: {}'.format(anti_list, species_list))

    run_plots(species_list, anti_list, criterion, ascending, plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # choose data to produce results for
    parser.add_argument('--criterion', dest='criterion', type=str, nargs='?', required=True)

    parser.add_argument('--species-list', dest='species_list', default=0, nargs='+', required=True)
    parser.add_argument('--anti-list', dest='anti_list', default=0, nargs='+', required=True)

    parser.add_argument('--shap', dest='shap', action="store_true")
    parser.add_argument('--exact', dest='exact', action="store_true")
    parser.add_argument('--range', dest='range', action="store_true")
    parser.add_argument('--PA', dest='pa', action="store_true")

    args = parser.parse_args()
    main(args)
