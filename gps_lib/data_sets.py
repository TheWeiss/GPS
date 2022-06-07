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
        except:
            genotypic = pd.DataFrame({})
            error_id = []
            for SRR_dir in tqdm(os.listdir(self.path_dict['geno'])):
                srr_features = p_utils.get_isolate_features(self.path_dict['geno']+'/'+SRR_dir)
                if type(srr_features) is not str:
                    genotypic = pd.concat([genotypic, srr_features], axis=0)
                else:
                    error_id += [srr_features]
            if len(error_id) == 0:
                error_id = None
            genotypic.to_csv(self.saved_files_path + '/geno.csv', index=False)
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
            self.all_ASR = self.all_ASR.drop_duplicates(
                subset=list(set(self.all_ASR.columns) - set(['platform', 'platform1', 'platform2'])),
                keep='first'
            )
            self._calculate_multi_mic_aid()
            
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
        self.all_ASR['sign'].fillna('=', inplace=True)
        self.all_ASR['sign'].replace(inplace=True, to_replace='==', value='=')

        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].str.lower()
        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].replace(' ', '_', regex=True)
        self.all_ASR['antibiotic_name'] = self.all_ASR['antibiotic_name'].replace('-', '_', regex=True)

        self.all_ASR['units'].replace(to_replace='mg/l', value='mg/L', inplace=True)
        self.all_ASR['measurement_has_/'].fillna(False, inplace=True)

        self.all_ASR['measurement'] = self.all_ASR['measurement'].apply(np.log2)
        self.all_ASR['measurement'].fillna(-9, inplace=True)

        self.all_ASR['resistance_phenotype'].replace('not-defined', np.nan, inplace=True)
        self.all_ASR['resistance_phenotype'] = \
            self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name', 'measurement'])[
                'resistance_phenotype'].transform(
                'first')

        def choose_one_run_id(df):
            df['run_id'] = df.iloc[0]['run_id']
            return df
        self.all_ASR = self.all_ASR.groupby(by='biosample_id').apply(choose_one_run_id)

        self.all_ASR['species_fam'].replace(self.species_dict, inplace=True)

    def _calculate_multi_mic_aid(self):
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
        self.all_ASR = self.all_ASR.drop(['biosample_id', 'antibiotic_name'], axis=1).reset_index()

        def how_bad(df):
            return df['measurement'].max() - df['measurement'].min()

        how_bad_multi = self.all_ASR.groupby(by=['biosample_id', 'antibiotic_name']).apply(how_bad)
        how_bad_multi.name = 'multi_dilution_distance'
        how_bad_multi = how_bad_multi.reset_index()
        self.all_ASR = how_bad_multi.merge(self.all_ASR, on=['biosample_id', 'antibiotic_name'])

        self.all_ASR['multi_too_different'] = self.all_ASR['multi_dilution_distance'] > 1.5
        self.all_ASR.drop(['level_2'], axis=1, inplace=True)

    @abstractmethod
    def _align_ASR(self):
        pass
    
    @abstractmethod
    def _merge_all_meta(self):
        pass

    @abstractmethod
    def _parse_measure_dash_per_file(self):
        pass

    def generate_dataset(self):
        pass
    
    def get_geno(self):
        return self.geno
    
    def get_pheno(self):
        return self.all_ASR
    

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

    def _parse_measure_dash_per_file(self, phen_df):
        if phen_df['measurement'].dtype == object:
            phen_df['measurement_has_/'] = phen_df['measurement'].apply(lambda x: len(re.findall("/(\\d+\.?\\d*)", str(x)))>0)
            phen_df['measurement2'] = phen_df['measurement'].apply(PATAKICDataSet._get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

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

    @staticmethod
    def _get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
    def generate_data_set(self):
        print('hello')


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

    def _parse_measure_dash_per_file(self, phen_df):
        if phen_df['measurement'].dtype == object:
            phen_df['measurement_has_/'] = phen_df['measurement'].apply(lambda x: len(re.findall("/(\\d+\.?\\d*)", str(x)))>0)
            phen_df['measurement2'] = phen_df['measurement'].apply(VAMPDataSet._get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

    def _align_ASR(self):
        self.all_ASR.drop(['species'], axis=1, inplace=True)
        self.all_ASR.rename(columns={
            'measurement_sign': 'sign',
        }, inplace=True)
    
    def _merge_all_meta(self):
        run2bio = pd.read_csv(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')

    @staticmethod
    def _get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
    def generate_data_set(self):
        print('hello')
        
        
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


    def _parse_measure_dash_per_file(self):
        pass


    def _align_ASR(self):
        self.all_ASR.rename(columns={
            'level_1': 'antibiotic_name',
            'measurement_sign': 'sign',
        }, inplace=True)
        self.all_ASR['units'] = 'mg/L'
        self.all_ASR['measurement_has_/'] = False
        self.all_ASR['measurement2'] = np.nan
        self.all_ASR['measurement_type'] = 'MIC'

    
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

    
    def generate_data_set(self):
        print('hello')


class PATRICDataSet(MICDataSet):

    def __init__(self, path_dict, pre_params=None):
        super().__init__('PATRIC', path_dict, pre_params)

    def _load_all_phan_data(self):
        self.all_ASR = pd.read_excel(self.path_dict['pheno'])
        self.all_ASR['genome_id'] = self.all_ASR['genome_id'].astype(str)
        self.all_ASR['DB'] = self.name

    def _parse_measure_dash_per_file(self):
        self.all_ASR['measurement'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, '1'), axis=1)
        self.all_ASR['measurement2'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, '2'), axis=1)
        self.all_ASR['measurement_has_/'] = self.all_ASR.apply(lambda row: PATRICDataSet._fix_PATRIC_MIC_value(row, 'has'), axis=1)

    def _align_ASR(self):
        self.all_ASR.rename(columns={
            'antibiotic': 'antibiotic_name',
            'measurement_sign': 'sign'
        }, inplace=True)


    def _merge_all_meta(self):
        run2bio = pd.read_excel(self.path_dict['run2bio'])
        run2bio['PATRIC_ID'] = run2bio['PATRIC_ID'].astype(str)
        run2bio = run2bio[['PATRIC_ID', 'sample_accession', 'Species']]
        run2bio.columns = ['genome_id', 'biosample_id', 'species_fam']
        run2bio['genome_id'] = run2bio['genome_id'].astype(str)
        self.all_ASR = self.all_ASR.merge(right=run2bio, how='inner', on='genome_id')


    @staticmethod
    def _fix_PATRIC_MIC_value(row, ans_type='1', log2=True, choose_first_dash=True):
        sign = row['measurement_sign']
        value = row['measurement_value']
        values_float = []
        values_str = []
        # try:
        if sign == '==' or sign == '':
            sign = '='

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

            if log2:
                if val == 0:
                    sign = '<'
                    values_str.append('-7')
                else:
                    values_str.append(str(np.log2(val)))
            else:
                values_str.append(str(val))
        if ans_type == '1':
            return values_str[0]
        elif ans_type == '2':
            if len(values_str) == 1:
                return np.nan
            else:
                values_str[0]
        elif ans_type == 'has':
            return len(values_str) > 1
        else:
            return '{} {}'.format(sign, values_str[0])

    def generate_data_set(self):
        print('hello')
#         run2biosam = pd.read_excel(self.path_dict['run2bio'])
#         run2biosam.columns = ['run_id', 'biosample_id']
#         run2biosam = run2biosam.drop_duplicates(subset='biosample_id', keep='first')
#         filtered_data = pd.read_excel(self.path_dict['filter_list'])
#         filtered_data.columns = ['species_fam', 'run_id']
    
#         pheno = self.pheno.drop(['species_fam'], axis=1)
#         labels = list(set(pheno.columns)-set(['biosample_id', 'genome_id']))
        
#         filtered_data = filtered_data.merge(right=run2biosam, how='inner', on='run_id')
#         filtered_data = filtered_data.merge(right=pheno, how='inner', on='biosample_id')
#         filtered_data = filtered_data.merge(right=geno, how='inner', on='run_id')
#         filtered_data['DB'] = self.name
    
#         features = list(set(geno.columns)-set(['run_id']))
#         return filtered_data, labels, features
    
# class CollectionDataSet(MICDataSet):
    
#     def __init__(self, dbs_list):
#         self.name = '_'.join([db.name for db in dbs_list])
#         self.path_dict = path_dict
#         self.dbs_list = dbs_list

