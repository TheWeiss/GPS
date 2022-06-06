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
import gps_lib.parse_raw_utils as p_utils

from abc import ABC, abstractmethod
 
class MICDataSet(ABC):
 
    def __init__(self, name, path_dict, pre_params = None, saved_files_path = '../pre_proccesing/'):
        super().__init__()
        
        self.name = name
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
                
            self.align_ASR()
            self.all_ASR['units'].replace(to_replace='mg/l', value='mg/L', inplace=True)
            self.all_ASR['measurement_has_/'].fillna(False, inplace=True)
            self.merge_all_meta()
            
            self.all_ASR = self.all_ASR.drop_duplicates(keep='first')
            
            self.all_ASR.to_csv(self.saved_files_path + '/all_ASR.csv', index=False)
            
    
    def _load_all_phan_data(self):
        print(self.path_dict['pheno'])
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

        phen_df['measurement_sign'].fillna('=', inplace=True)
        phen_df['measurement_sign'].replace(inplace=True, to_replace='==', value='=')

        phen_df['antibiotic_name'] = phen_df['antibiotic_name'].str.lower()
        phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace(' ', '_', regex=True)
        phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace('-', '_', regex=True)
        
        phen_df = self._parse_measure_dash_per_file(phen_df)
        
        phen_df['DB'] = self.name

        return phen_df

    @abstractmethod
    def align_ASR(self):
        pass
    
    @abstractmethod
    def merge_all_meta(self):
        pass

    @abstractmethod
    def _parse_measure_dash_per_file(self):
        pass
    
    # @abstractmethod
    # def generate_dataset(self):
    #     pass
    
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
            phen_df['measurement2'] = phen_df['measurement'].apply(PATAKICDataSet.get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

    def align_ASR(self):
        self.all_ASR['platform'].fillna(self.all_ASR['platform '], inplace=True)
        self.all_ASR.drop(['platform ', 'biosample_id', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
               'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
               'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)    
        self.all_ASR.columns = [
            'biosample_id', 
            'species',
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
    
    def merge_all_meta(self):
        run2bio = pd.read_excel(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')

    @staticmethod
    def get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
    def generate_data_set(self):
        print('hello')

        
class PADataSet(MICDataSet):
    
    def __init__(self, path_dict, pre_params = None):
        super().__init__('PA', path_dict, pre_params)
        

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
            phen_df['measurement2'] = phen_df['measurement'].apply(PATAKICDataSet.get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

    def align_ASR(self):
        pass
        # self.all_ASR['platform'].fillna(self.all_ASR['platform '], inplace=True)
        # self.all_ASR.drop(['platform ', 'biosample_id', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
        #        'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
        #        'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)    
        # self.all_ASR.columns = [
        #     'biosample_id', 
        #     'species',
        #     'antibiotic_name',
        #     'test_standard',
        #     'standard_year',
        #     'measurement_type',
        #     'measurement',
        #     'units',
        #     'sign',
        #     'resistance_phenotype',
        #     'platform',
        #     'DB',
        #     'measurement_has_/',
        #     'measurement2',
        # ]
    
    def merge_all_meta(self):
        run2bio = pd.read_csv(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')

    @staticmethod
    def get_mes2(x):
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
            phen_df['measurement2'] = phen_df['measurement'].apply(PATAKICDataSet.get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df

    def align_ASR(self):
        pass
        # self.all_ASR['platform'].fillna(self.all_ASR['platform '], inplace=True)
        # self.all_ASR.drop(['platform ', 'biosample_id', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
        #        'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
        #        'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)    
        # self.all_ASR.columns = [
        #     'biosample_id', 
        #     'species',
        #     'antibiotic_name',
        #     'test_standard',
        #     'standard_year',
        #     'measurement_type',
        #     'measurement',
        #     'units',
        #     'sign',
        #     'resistance_phenotype',
        #     'platform',
        #     'DB',
        #     'measurement_has_/',
        #     'measurement2',
        # ]
    
    def merge_all_meta(self):
        run2bio = pd.read_csv(self.path_dict['run2bio'])
        run2bio.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(self.path_dict['filter_list'])
        filtered_data.columns = ['species_fam', 'run_id']

        filtered_data = filtered_data.merge(right=run2bio, how='inner', on='run_id')
        self.all_ASR = filtered_data.merge(right=self.all_ASR, how='inner', on='biosample_id')

    @staticmethod
    def get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
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

