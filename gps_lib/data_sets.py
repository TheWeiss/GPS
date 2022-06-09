import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from scipy.stats.distributions import expon
from scipy.stats import uniform
from datetime import datetime
import os
import glob
import re
import h2o
from tqdm import tqdm
import pickle
import json
import gps_lib.parse_raw_utils as p_utils

from abc import ABC, abstractmethod
 
class MICDataSet(ABC):

    def __init__(self, name, path_dict, pre_params=None, saved_files_path='../pre_proccesing/',
                 species_dict_path="../resources/species_dict.json"):
        super().__init__()
        
        self.name = name
        with open(species_dict_path) as json_file:
            self.species_dict = json.load(json_file)
        self.path_dict = path_dict
        self.pre_params = pre_params
        
        if self.pre_params is None:
            pre_params_name = 'base_line'
        else:
            pre_params_name = str(' '.join([str(key) + '_' + str(value) for key, value in self.pre_param.items()]))
        self.saved_files_path = saved_files_path + pre_params_name + '/' + name
        if not os.path.exists(self.saved_files_path):
            os.makedirs(self.saved_files_path)
            
        self._load_geno()
        self._load_pheno()

    def _load_geno(self):
        try:
            genotypic = pd.read_csv(self.saved_files_path + '/geno.csv')
            self.geno = genotypic
        except:
            self._load_all_geno_data()
            self.geno.drop_duplicates(subset=['run_id'], keep='first', inplace=True)
            self.geno.to_csv(self.saved_files_path + '/geno.csv', index=False)



    def _load_all_geno_data(self):
        genotypic = pd.DataFrame({})
        error_id = []
        for SRR_dir in tqdm(os.listdir(self.path_dict['geno'])):
            srr_features = p_utils.get_isolate_features(self.path_dict['geno'] + '/' + SRR_dir)
            if type(srr_features) is not str:
                genotypic = pd.concat([genotypic, srr_features], axis=0)
            else:
                error_id += [srr_features]
        if len(error_id) == 0:
            error_id = None
        self.geno = genotypic

    def _load_pheno(self):
        try:
            self.all_ASR = pd.read_csv(self.saved_files_path + '/all_ASR.csv')
        except:

            self._load_all_phan_data()
            self._align_ASR()
            self._merge_all_meta()
            self.all_ASR = self.all_ASR.merge(right=self.geno['run_id'], how='inner', on='run_id')
            self._fix_general_values()
            self._calculate_multi_mic_aid()
            self._calculate_multi_mic_aid()
            
            self._test_phen()
            self.all_ASR.to_csv(self.saved_files_path + '/all_ASR.csv', index=False)
            
    
    def _load_all_phan_data(self):
        self.all_ASR = pd.DataFrame({})
        error_id = []
        for sam_dir in tqdm(os.listdir(self.path_dict['pheno'])):
            sam_phen = self._load_all_phen_data_per_file(self.path_dict['pheno']+'/'+sam_dir)
            if type(sam_phen) is not str:
                self.all_ASR = pd.concat([self.all_ASR, sam_phen], axis=0)
            else:
                error_id += [sam_dir]
        if len(error_id) == 0:
            error_id = None
    
    
    def _load_all_phen_data_per_file(self,
        path, 
        file_sep=',',
        file_columns=None,
        bio_id_sep='_'
    ):
        try:
            phen_df = pd.read_csv(path, sep=file_sep)
        except:
            return path
        
        biosample_id = path.split('/')[-1].split(bio_id_sep)[0]
        phen_df['biosample_id'] = biosample_id
        if file_columns is not None:
            phen_df.columns = file_columns
        if 'species' not in phen_df.columns:
            phen_df['species'] = np.nan


        phen_df = self._parse_measure_dash_per_file(phen_df)
        
        phen_df['DB'] = self.name

        return phen_df

    def _fix_general_values(self):
        self.all_ASR.drop(
            ['is_min_mic', 'is_max_mic', 'is_multi_mic', 'multi_dilution_distance', 'multi_too_different'], axis=1, errors='ignore')
        self.all_ASR['sign'].fillna('=', inplace=True)
        self.all_ASR['sign'].replace(inplace=True, to_replace='==', value='=')

        def choose_one_run_id(df):
            df['run_id'] = df.iloc[0]['run_id']
            return df
        self.all_ASR = self.all_ASR.groupby(by='biosample_id').apply(choose_one_run_id)

        self.all_ASR['species_fam'].replace(self.species_dict, inplace=True)
        
        def fix_ambiguse_sign(df):
            if '=' in df['sign'].values:
                df['sign'] = '='
            elif ('>=' in df['sign'].values) and ('>' in df['sign'].values):
                df['sign'] = '>'
            elif ('<=' in df['sign'].values) and ('<' in df['sign'].values):
                df['sign'] = '<='
            return df['sign']
        multi_sign = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name', 'measurement']).apply(
            fix_ambiguse_sign)
        multi_sign = multi_sign.reset_index().drop('level_3', axis=1)
        self.all_ASR.drop('sign', axis=1, inplace=True)
        self.all_ASR = self.all_ASR.merge(right=multi_sign, on=['biosample_id', 'antibiotic_name', 'measurement'])

        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].str.lower()
        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].replace(' ', '_', regex=True)
        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].replace('-', '_', regex=True)
        self.all_ASR['antibiotic_name'].replace(to_replace='trimethoprim_sulphamethoxazole', value='trimethoprim_sulfamethoxazole', inplace=True)

        self.all_ASR['units'].replace(to_replace='mg/l', value='mg/L', inplace=True)
        self.all_ASR['measurement_has_/'].fillna(False, inplace=True)

        self.all_ASR['measurement'] = self.all_ASR['measurement'].apply(np.log2)
        self.all_ASR = self.all_ASR.dropna(subset=['measurement'])
        self.all_ASR['measurement'] = self.all_ASR['measurement'].replace({-np.inf: -9})
        
        self.all_ASR['test_standard'].replace({'missing': None, np.nan: None}, inplace=True)
        self.all_ASR['test_standard'] = self.all_ASR['test_standard'].str.lower()
        self.all_ASR['test_standard'] = self.all_ASR['test_standard'].replace(' ', '_', regex=True)
        self.all_ASR['test_standard'] = self.all_ASR['test_standard'].replace('-', '_', regex=True)
        
        self.all_ASR['standard_year'].replace({'not_determined': None, 'M100-S24': None, 'as described in 2013/652/EU': '2013'}, inplace=True)
        self.all_ASR['standard_year'] = self.all_ASR['standard_year'].astype(float)

        self.all_ASR['platform'].replace({
            'manually': 'Manual',
            'missing': None,
            '-': None,
        }, inplace=True)
        self.all_ASR['platform'] = self.all_ASR['platform'].str.lower()
        if 'platform1' in self.all_ASR.columns:
            self.all_ASR['platform1'].replace({
                'biomerieux': 'Biomérieux',
                'BiomŽrieux': 'Biomérieux',
            }, inplace=True)
            self.all_ASR['platform1'] = self.all_ASR['platform1'].str.lower()
        if 'platform2' in self.all_ASR.columns:
            self.all_ASR['platform2'].replace({
                'missing': None,
            })
            self.all_ASR['platform2'] = self.all_ASR['platform2'].str.lower()
        if 'measurement_type' in self.all_ASR.columns:
            self.all_ASR['measurement_type'] = self.all_ASR['measurement_type'].str.lower()
            self.all_ASR['measurement_type'].replace({
                'broth_microdilution': 'mic',
                'microbroth dilution': 'mic',
                'mic broth microdilution': 'mic',
                'disk diffusion': 'disk_diffusion',
                'agar dilution': 'agar_dilution',
                'disc-diffusion': 'disk_diffusion',
            }, inplace=True)

        self.all_ASR['resistance_phenotype'].replace(
            {'non_susceptible': 'I',
             'Not defined': None,
             'Susceptible-dose dependent': 'S',
             'not defined': None,
             'resistant': 'R',
             'intermediate': 'I',
             'Susceptible': 'S',
             }, inplace=True)
        self.all_ASR['resistance_phenotype'] = \
            self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name', 'measurement'])[
                'resistance_phenotype'].transform(
                'first')

        def fix_ambiguse_standard(df):
            if len(df) > 1:
                if 'clsi' in df['test_standard'].values:
                    i = df[df['test_standard']=='clsi'].head(1).index
                    year = df.loc[i, 'standard_year']
                    df['test_standard'] = 'clsi'
                    df['standard_year'] = year

                if df[['test_standard', 'standard_year']].isna().all(axis=1).any():
                    if ~df[['test_standard', 'standard_year']].isna().all(axis=1).all():
                        i = df[~df['test_standard'].isna()].head(1).index
                        year = df.loc[i, 'standard_year'].iloc[0]
                        standard = df.loc[i, 'test_standard'].iloc[0]
                        df['test_standard'] = standard
                        df['standard_year'] = year
            return df
        self.all_ASR = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name', 'measurement']).apply(fix_ambiguse_standard)

        def is_unique(s):
            a = s.to_numpy()
            return (a[0] == a).all()
        def prefer_multi(df):
            if len(df) > 1:
                if df['measurement_has_/'].any(axis=0):
                    df = df[df['measurement_has_/']]
                    return df.iloc[0]
                elif 'clsi' in list(df['test_standard'].unique()):
                    if is_unique(df['test_standard']):
                        if not df['standard_year'].isna().all():
                            df = df.dropna(subset=['standard_year'])
                            # df = df[~df['standard_year'].isna()]
                            return df.iloc[0]
            return df

        self.all_ASR = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name', 'measurement']).apply(prefer_multi)
        if 'biosample_id' in self.all_ASR.index.names:
            self.all_ASR.drop(['biosample_id', 'antibiotic_name', 'measurement'], axis=1).reset_index()
        if 'level_3' in list(self.all_ASR.columns.values):
            self.all_ASR.drop(['level_3'], axis=1, inplace=True)
        if 0 in list(self.all_ASR.columns.values):
            self.all_ASR.drop(0, axis=1, inplace=True)

    def _calculate_multi_mic_aid(self):
        self.all_ASR.drop(
            ['is_min_mic', 'is_max_mic', 'is_multi_mic', 'multi_dilution_distance', 'multi_too_different'],
            axis=1, errors='ignore', inplace=True)
        self.all_ASR = self.all_ASR.drop_duplicates(
            subset=list(set(self.all_ASR.columns) - {'DB', 'is_min_mic', 'is_max_mic', 'measurement_type',
                                                     'platform', 'platform1', 'platform2', 'genome_id', 'Isolate',
                                                     'is_multi_mic', 'multi_dilution_distance'}),
            keep='first',
        )
        def is_multi_mic(df):
            if len(df) > 1:
                return True
            return False
        multi_mic = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name']).apply(is_multi_mic)
        multi_mic.name = 'is_multi_mic'
        multi_mic = multi_mic.reset_index()
        self.all_ASR = multi_mic.merge(self.all_ASR, on=['biosample_id', 'antibiotic_name'])

        self.all_ASR['exact_value'] = self.all_ASR['sign'] == '='

        def choose_multi_mic(df):
            df['is_max_mic'] = False
            df['is_min_mic'] = False
            df.sort_values(by='measurement', ascending=False, inplace=True)
            df.loc[df.head(1).index, 'is_max_mic'] = True
            df.loc[df.tail(1).index, 'is_min_mic'] = True
            return df

        self.all_ASR = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name']).apply(choose_multi_mic)
        if 'biosample_id' in self.all_ASR.index.names:
            self.all_ASR = self.all_ASR.drop(['biosample_id', 'antibiotic_name'], axis=1).reset_index()
            self.all_ASR.drop(['level_2'], axis=1, inplace=True)

        def how_bad(df):
            return df['measurement'].max() - df['measurement'].min()
        how_bad_multi = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name']).apply(how_bad)
        how_bad_multi.name = 'multi_dilution_distance'
        how_bad_multi = how_bad_multi.reset_index()

        self.all_ASR = how_bad_multi.merge(self.all_ASR, on=['biosample_id', 'antibiotic_name'])
        self.all_ASR['multi_too_different'] = self.all_ASR['multi_dilution_distance'] > 1.5
    
    def _test_phen(self):
        assert(len(self.all_ASR[self.all_ASR['is_multi_mic'] == True][self.all_ASR['multi_dilution_distance'] == 0])==0)

    @abstractmethod
    def _align_ASR(self):
        pass
    
    @abstractmethod
    def _merge_all_meta(self):
        pass

    def _parse_measure_dash_per_file(self, phen_df):
        if phen_df['measurement'].dtype == object:
            phen_df['measurement_has_/'] = phen_df['measurement'].apply(
                lambda x: len(re.findall("/(\\d+\.?\\d*)", str(x))) > 0)
            phen_df['measurement2'] = phen_df['measurement'].apply(MICDataSet._get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(
                lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

    @staticmethod
    def _get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
    def get_geno(self):
        return self.geno
    
    def get_pheno(self):
        return self.all_ASR

    def generate_dataset(self, ds_param=None, antibiotic=None, species=None):
        if ds_param is None:
            ds_param = {'species_sep': True, 'antibiotic_sep': True}
            if antibiotic is None:
                antibiotic = 0
            if species is None:
                species = 0

        ds_param_name = str(' '.join([str(key) + '_' + str(value) for key, value in ds_param.items()]))
        ds_param = MICDataSet._add_default_ds_param(ds_param)
        filtered, antibiotic_name, species_name = self._filter_data(ds_param, antibiotic, species)
        ds_param_files_path = self.saved_files_path + '/' + ds_param_name + '/' + species_name + '/' + antibiotic_name
        if not os.path.exists(ds_param_files_path):
            os.makedirs(ds_param_files_path)

        try:
            train = pd.read_csv(ds_param_files_path + '/train.csv')
            test = pd.read_csv(ds_param_files_path + '/test.csv')
            range = pd.read_csv(ds_param_files_path + '/range.csv')
            with open(ds_param_files_path + '/col_names.json') as json_file:
                col_names = json.load(json_file)
            return train, test, range, col_names, ds_param_files_path, antibiotic_name, species_name
        except:
            train_label, test_label, range_label, cv = self._split_train_valid_test(ds_param, filtered)
            train, test, range, col_names = self._merge_geno2pheno(train_label, test_label, range_label)
            train.to_csv(ds_param_files_path + '/train.csv')
            test.to_csv(ds_param_files_path + '/test.csv')
            range.to_csv(ds_param_files_path + '/range.csv')
            with open(ds_param_files_path + '/col_names.csv', "w") as fp:
                json.dump(col_names, fp)
            pd.DataFrame(ds_param, index=[0]).to_csv(ds_param_files_path + '/ds_param.csv')
            return train, test, range, col_names, ds_param_files_path, antibiotic_name, species_name


    @staticmethod
    def _add_default_ds_param(ds_param):
        default_values = {
            'species_sep': True,
            'antibiotic_sep': True,
            'handle_range': 'remove',  # remove/strip/move
            'handle_multi_mic': 'remove',  # remove/max/min/rand
            'ignore_small_dilu': True,  # remove/max/min/rand
            'task': 'regression',  # regression/classification/SIR
            'log2': True,
            'move_range_by': 5,
            'reg_stratified': True,
            'stratified_cv_num': 3,
            'random_seed': 42,
        }
        full_ds_param = {}
        for key, value in default_values.items():
            full_ds_param[key] = ds_param.get(key, value)
        return full_ds_param

    def _filter_data(self, ds_param, anti, spec):
        species = ''
        antibiotic = ''
        filtered = self.all_ASR.copy()
        filtered.set_index('run_id', inplace=True)
        if ds_param['species_sep']:
            if type(spec) == int:
                species_list = filtered.groupby(by='biosample_id').apply(
                    lambda x: x['species_fam'].iloc[0]).value_counts().drop(
                    ['Salmonella enterica', 'Streptococcus pneumoniae'], axis=0, errors='ignore').index.values
                species = species_list[spec]
            else:
                species = spec
            filtered = filtered[filtered['species_fam'] == species]
        else:
            species = 'all_species'

        if ds_param['antibiotic_sep']:
            if type(anti) == int:
                antibiotic_list = filtered.groupby(by='biosample_id').apply(
                    lambda x: x['antibiotic_name'].iloc[0]).value_counts().index.values
                antibiotic = antibiotic_list[anti]
            else:
                antibiotic = anti
            filtered = filtered[filtered['antibiotic_name'] == antibiotic]
        else:
            antibiotic = 'all_antibiotic'

        if ds_param['handle_multi_mic'] == 'remove':
            if ds_param['ignore_small_dilu']:
                filtered = filtered[~filtered['multi_too_different']]
                filtered = filtered[filtered['is_max_mic']]
            else:
                filtered = filtered[~filtered['is_multi_mic']]

        return filtered, antibiotic, species

    def _split_train_valid_test(self, ds_param, filtered):

        range_label = filtered[~filtered['exact_value']][['measurement', 'sign']]
        exact_label = filtered[filtered['exact_value']][['measurement', 'sign']]

        exact_y = exact_label['measurement'].copy()
        if ds_param['task'] == 'regression':
            exact_y = exact_label['measurement'].copy()
        elif ds_param['task'] == 'classification':
            exact_y = exact_label.apply(lambda row: str(' '.join([row['measurement'], row['sign']])))
            exact_y.name='measurement'

        range_y = pd.DataFrame({})
        if ds_param['handle_range'] != 'remove':
            if ds_param['task'] == 'regression':
                if ds_param['handle_range'] == 'strip':
                    range_y = range_label['measurement'].copy()
                elif ds_param['handle_range'] == 'move':
                    range_y = range_label['measurement'].copy().mask(
                        range_label['sign'].apply(lambda x: '>' in x),
                        range_label['measurement'] + ds_param['move_range_by'])
                    range_y = range_y.mask(
                        range_label['sign'].apply(lambda x: '<' in x),
                        range_y - ds_param['move_range_by'])
            else:
                raise Exception('regression not in the naive approach is not implemented yet.')
            range_y.name = 'measurement'

        range_train_ids = []
        range_test_ids = []
        if ds_param['reg_stratified']:
            exact_train_ids, exact_test_ids = MICDataSet._strat_id(exact_y, ds_param['random_seed'])
            if ds_param['handle_range'] != 'remove':
                range_train_ids, range_test_ids = MICDataSet._strat_id(range_y, ds_param['random_seed'])
        else:
            exact_train_ids, exact_test_ids = train_test_split(
                list(exact_y.index), test_size=0.2, random_state=ds_param['random_seed'])
            if ds_param['handle_range'] != 'remove':
                range_train_ids, range_test_ids = train_test_split(
                    list(range_y.index), test_size=0.2, random_state=ds_param['random_seed'])
        exact_y_train = exact_y.loc[exact_train_ids,]
        range_y_train = range_y.loc[range_train_ids,]
        train_label = pd.concat([exact_y.loc[exact_train_ids,], range_y.loc[range_train_ids,]])
        test_label = pd.concat([exact_y.loc[exact_test_ids,], range_y.loc[range_test_ids,]])

        # generate cv from train that is stratified
        cv = []
        if ds_param['stratified_cv_num']>1:
            train_ref = train_label.reset_index()
            for i in np.arange(ds_param['stratified_cv_num']):
                exact_cv_train_ids, exact_cv_test_ids = MICDataSet._strat_id(
                    exact_y_train, ds_param['random_seed'], seed_add=i)
                range_cv_train_ids, range_cv_test_ids = [], []
                if ds_param['handle_range'] != 'remove':
                    range_cv_train_ids, range_cv_test_ids = MICDataSet._strat_id(
                        range_y_train, ds_param['random_seed'], seed_add=i)
                cv_train_ids = exact_cv_train_ids + range_cv_train_ids
                cv_test_ids = exact_cv_test_ids + range_cv_test_ids
                cv.append((list(train_ref[train_ref['index'].isin(cv_train_ids)].index),
                           list(train_ref[train_ref['index'].isin(cv_test_ids)].index)))

        return train_label, test_label, range_label, cv


    @staticmethod
    def _strat_id(y, random_seed=42, seed_add=0):
        print(y)
        print(type(y))
        print(y.unique())
        print(type(y.unique()))
        train_ids = []
        test_ids = []
        for y_val in y.unique().values:
            sub_value_id = list(y[y == y_val])
            if len(sub_value_id) > 1:
                train_id, test_id = train_test_split(sub_value_id, test_size=0.2, random_state=random_seed+seed_add)
                train_ids = train_ids + train_id
                test_ids = test_ids + test_id
            else:
                train_ids = train_ids + sub_value_id
        return train_ids, test_ids

    def _merge_geno2pheno(self, train_label, test_label, range_label):
        train_data = self.geno.set_index('run_id').fillna(0).merge(right=train_label, left_index=True, right_index=True)
        test_data = self.geno.set_index('run_id').fillna(0).merge(right=test_label, left_index=True, right_index=True)
        range_data = self.geno.set_index('run_id').fillna(0).merge(right=range_label, left_index=True, right_index=True)

        col_names = {}
        col_names['features'] = list(self.geno.set_index('run_id').columns.values)
        col_names['id'] = 'run_id'
        col_names['label'] = 'measurement'
        return train_data, test_data, range_data, col_names
    

class PATAKICDataSet(MICDataSet):
    
    def __init__(self, path_dict, pre_params = None):
        super().__init__('PATAKI', path_dict, pre_params)

    def _load_all_phen_data_per_file(self, path):
        return super()._load_all_phen_data_per_file(
                path=path,
                file_sep='\t',
                file_columns=None,
                bio_id_sep='_',
            )

    def _align_ASR(self):
        self.all_ASR['platform'].fillna(self.all_ASR['platform '], inplace=True)

        self.all_ASR.drop(['platform ', 'biosample_id', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
               'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
               'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)    
        self.all_ASR.columns = [
            'biosample_id', 
            'species_fam',
            'antibiotic_name',
            'test_standard',
            'standard_year',
            'measurement_type',
            'measurement',
            'units',
            'sign',
            'resistance_phenotype',
            'platform',
            'DB',
            'measurement_has_/',
            'measurement2',
        ]
    
    def _merge_all_meta(self):
        run2bio = pd.read_excel(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']
        filtered_data.drop(['species_fam'], axis=1, inplace=True)

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')



class VAMPDataSet(MICDataSet):
    
    def __init__(self, path_dict, pre_params = None):
        super().__init__('VAMP', path_dict, pre_params)
        

    def _load_all_phen_data_per_file(self, path):
        return super()._load_all_phen_data_per_file(
                path=path,
                file_sep=',',
                file_columns=[
                    'antibiotic_name', 
                    'resistance_phenotype', 
                    'measurement_sign', 
                    'measurement', 
                    'units', 
                    'measurement_type',
                    'platform', 
                    'platform1', 
                    'platform2', 
                    'test_standard',
                    'biosample_id'
                ],
                bio_id_sep='.',
            )

    def _align_ASR(self):
        self.all_ASR.drop(['species'], axis=1, inplace=True)
        self.all_ASR.rename(columns={
            'measurement_sign': 'sign',
        }, inplace=True)
        self.all_ASR['standard_year'] = np.nan
    
    def _merge_all_meta(self):
        run2bio = pd.read_csv(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')
        
        
class PADataSet(MICDataSet):
    
    def __init__(self, path_dict, pre_params = None):
        super().__init__('PA', path_dict, pre_params)
        

    def _load_all_phan_data(self):
        anti_list = ['tobramycin', 'ciprofloxacin', 'meropenem', 'ceftazidime']
        anti_list_mic = [x + ' MIC' for x in anti_list]
        self.all_ASR = pd.read_excel(self.path_dict['pheno'], sheet_name='Strain and MIC', header=1, nrows=414).set_index(
            'Isolate')
        phen_cat = self.all_ASR[anti_list]
        phen_MIC = self.all_ASR[anti_list_mic]
        phen_MIC.columns = anti_list
        self.all_ASR = pd.concat([phen_cat.stack(), phen_MIC.stack()], axis=1)
        self.all_ASR.columns = ['resistance_phenotype', 'measurement']
        self.all_ASR['measurement_sign'] = '='
        self.all_ASR.loc[self.all_ASR['measurement'] == '≤0.125', 'measurement_sign'] = '<='
        self.all_ASR.loc[self.all_ASR['measurement'] == '≤0.5', 'measurement_sign'] = '<='
        self.all_ASR.replace({'≤0.125': 0.125, '≤0.5': 0.5}, inplace=True)
        self.all_ASR['measurement'] = self.all_ASR['measurement'].astype(float)
        self.all_ASR.reset_index(inplace=True)
        self.all_ASR['DB'] = self.name

    def _align_ASR(self):
        self.all_ASR.rename(columns={
            'level_1': 'antibiotic_name',
            'measurement_sign': 'sign',
        }, inplace=True)
        self.all_ASR['units'] = 'mg/L'
        self.all_ASR['measurement_has_/'] = False
        self.all_ASR['measurement2'] = np.nan
        self.all_ASR['measurement_type'] = 'MIC'
        self.all_ASR['test_standard'] = 'clsi'
        self.all_ASR['standard_year'] = 2018
    
    def _merge_all_meta(self):
        run2bio = pd.read_excel(self.path_dict['run2bio'])
        run2bio = run2bio[['Run', 'Platform', 'Model', 'BioSample', 'ScientificName', 'SampleName']]
        run2bio.rename(columns={
            'Run': 'run_id',
            'BioSample': 'biosample_id',
            'ScientificName': 'species_fam',
            'SampleName': 'Isolate',
            'Platform': 'platform',
            'Model': 'platform1',
            
        }, inplace=True)

        self.all_ASR = run2bio.merge(right=self.all_ASR, how='inner', on='Isolate')


class PATRICDataSet(MICDataSet):

    def __init__(self, path_dict, pre_params=None):
        super().__init__('PATRIC', path_dict, pre_params)
    
    def _load_all_phan_data(self):
        self.all_ASR = pd.read_excel(self.path_dict['pheno'])
        self.all_ASR['genome_id'] = self.all_ASR['genome_id'].astype(str)
        self.all_ASR['DB'] = self.name
        self._parse_measure_dash_per_file()

    def _parse_measure_dash_per_file(self):
        self.all_ASR['measurement'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, '1'), axis=1)
        self.all_ASR['measurement2'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, '2'), axis=1)
        self.all_ASR['measurement_has_/'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, 'has'), axis=1)

    def _align_ASR(self):
        self.all_ASR.drop(['genus', 'genome_name', 'taxon_id', 'measurement_value', 'source'], axis=1, inplace=True)
        self.all_ASR.rename(columns={
            'antibiotic': 'antibiotic_name',
            'measurement_sign': 'sign',
            'measurement_unit': 'units',
            'laboratory_typing_method': 'measurement_type',
            'laboratory_typing_method_version': 'platform2',
            'testing_standard': 'test_standard',
            'testing_standard_year': 'standard_year',
            'laboratory_typing_platform': 'platform',
            'vendor': 'platform1',
            'species': 'species_fam',
            'resistant_phenotype': 'resistance_phenotype'
        }, inplace=True)


    def _merge_all_meta(self):
        run2bio = pd.read_excel(self.path_dict['run2bio'])
        run2bio['PATRIC_ID'] = run2bio['PATRIC_ID'].astype(str)
        run2bio = run2bio[['run_accession', 'PATRIC_ID', 'sample_accession', 'Species']]
        run2bio.columns = ['run_id', 'genome_id', 'biosample_id', 'species_fam']
        run2bio.drop(['species_fam'], axis=1, inplace=True)
        run2bio['genome_id'] = run2bio['genome_id'].astype(str)
        self.all_ASR = self.all_ASR.merge(right=run2bio, how='inner', on='genome_id')

    @staticmethod
    def _fix_PATRIC_MIC_value(row, ans_type='1', log2=False, choose_first_dash=True):
        value = row['measurement_value']
        values_float = []
        values_str = []

        if type(value) == str:
            if choose_first_dash:
                values_float.append(float(value.split('/')[0]))
            else:
                values_float.append(float(value.split('/')[0]))
                values_float.append(float(value.split('/')[1]))
        elif type(value) == datetime:
            if value.date().year == 2022:
                values_float.append(value.date().month)
                values_float.append(value.date().day)
            else:
                values_float.append(value.date().month)
                values_float.append(int(str(value.date().year)[2:]))
        else:
            values_float.append(value)

        for val in values_float:
            values_str.append(str(val))
        if ans_type == '1':
            return float(values_str[0])
        elif ans_type == '2':
            if len(values_str) == 1:
                return np.nan
            else:
                return float(values_str[0])
        elif ans_type == 'has':
            return len(values_str) > 1

    
class CollectionDataSet(MICDataSet):

    def __init__(self, dbs_list: list):
        name = '_'.join([db.name for db in dbs_list])
        path_dict = {db.name: db.path_dict for db in dbs_list}
        self.dbs_list = dbs_list
        super().__init__(name, path_dict)

    def _load_all_geno_data(self):
        self.geno = None
        for db in self.dbs_list:
            if self.geno is None:
                self.geno = db.geno
            else:
                self.geno = pd.concat([self.geno, db.geno], axis=0)


    def _load_all_phan_data(self):
        self.all_ASR = None
        for db in self.dbs_list:
            if self.all_ASR is None:
                self.all_ASR = db.all_ASR
            else:
                self.all_ASR = pd.concat([self.all_ASR, db.all_ASR], axis=0)

        def fill_run_id(df):
            if len(df) > 1:
                if len(df['run_id'].unique()) > 1:
                    run_id = df.iloc[0]['run_id']
                    df['run_id'] = run_id
            return df
        self.all_ASR = self.all_ASR.groupby(by='biosample_id').apply(fill_run_id)

    def _align_ASR(self):
        pass

    def _merge_all_meta(self):
        pass
