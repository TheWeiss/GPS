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
        self.pre_param = pre_params
        
        if self.pre_params is None:
            pre_params_name = 'base_line'
        else:
            pre_params_name = str(' '.join([str(key) + '_' + str(value) for key, value in dic.items()]))
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
                srr_features = p_utils.get_isolate_features(path+'/'+SRR_dir)
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
            all_ASR = pd.read_csv(self.saved_files_path + '/all_ASR.csv')
            return all_ASR
        except:
            all_ASR = pd.DataFrame({})
            error_id = []
            for file_format, resources_dict in resources_path.items():
                if file_format in ['VAMP', 'PATAKI']:
                    for sam_dir in tqdm(os.listdir(resources_dict['pheno'])):
                        sam_phen = self._load_all_phen_data(resources_dict['pheno']+'/'+sam_dir)
                        if type(sam_phen) is not str:
                            all_ASR = pd.concat([all_ASR, sam_phen], axis=0)
                        else:
                            # print(sam_dir)
                            error_id += [sam_dir]
                else:
                    sam_phen = self._load_all_phen_data(path)
                    all_ASR = pd.concat([all_ASR, sam_phen], axis=0)
            if len(error_id) == 0:
                error_id = None
            all_ASR = self.align_ASR(all_ASR)
            all_ASR.to_csv(self.saved_files_path + '/all_ASR.csv', index=False)
        self.all_ASR  = all_ASR
    
    def _load_all_phen_data(
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
        
        phen_df = MICDataSet._parse_measure_dash_per_file(phen_df)
        
        phen_df['DB'] = self.name

        return phen_df
    
    @staticmethod
    @abstractmethod
    def _parse_measure_dash_per_file(self):
        pass
    
    @abstractmethod
    def generate_dataset(self):
        pass
    
    def get_geno(self):
        return self.geno
    
    def get_pheno(self):
        return self.pheno
    

class PATAKICDataSet(MICDataSet):
    
    def __init__(self, path_dict, pre_params = None):
        super().__init__('PATAKI', path_dict, pre_params)
        
    def _load_all_phen_data(self, path):
        return super()._load_all_phen_data(
                path=path,
                file_sep='\t',
                file_columns=None,
                bio_id_sep='_',
            )
        
    @staticmethod
    def _parse_measure_dash_per_file(phen_df):
        if phen_df['measurement'].dtype == object:
            phen_df['measurement_has_/'] = phen_df['measurement'].apply(lambda x: len(re.findall("/(\\d+\.?\\d*)", str(x)))>0)
            phen_df['measurement2'] = phen_df['measurement'].apply(PATAKICDataSet.get_mes2)
            phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        return phen_df
    
    @staticmethod
    def get_mes2(x):
        if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
            return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
        return np.nan
    
    def generate_data_set(self):
        pass
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

