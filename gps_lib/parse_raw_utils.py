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

####################################################################################
################################## Parse functions ##################################
####################################################################################

def save_filtered_metadata(tot_data, all_ASR, ds_name):
    full_ds_path = '../resources/{}'.format(ds_name)
    os.makedirs(full_ds_path, exist_ok=True)
    filtered_ASR = all_ASR[all_ASR['biosample_id'].isin(tot_data['biosample_id'])]
    filtered_ASR.to_csv('{}/filtered_metadata.csv'.format(full_ds_path))
    
    return filtered_ASR


'''
resources_path - a dictionary of db name as keys, and db ASR folders path as values.
    for example: {
        'PATAKI': {'pheno': '../resources/26.12.21/Pataki_paper/AST_2548_all'}, 
        'VAMP': {'pheno': "../resources/28.12.21/VAMPr_3400samples/VAMP_final_for_Amit.2021.12.28/VAMP_full_AST_data"},
    }
'''
def get_metadata(resources_path):
    all_ASR = pd.DataFrame({})
    error_id = []
    for file_format, resources_dict in resources_path.items():
        if file_format in ['VAMP', 'PATAKI']:
            for sam_dir in tqdm(os.listdir(resources_dict['pheno'])):
                sam_phen = load_all_phen_data(resources_dict['pheno']+'/'+sam_dir, file_format=file_format)
                if type(sam_phen) is not str:
                    all_ASR = pd.concat([all_ASR, sam_phen], axis=0)
                else:
                    # print(sam_dir)
                    error_id += [sam_dir]
        else:
            sam_phen = load_all_phen_data(resources_dict['pheno'], resources_dict['run2bio'], file_format=file_format)
            all_ASR = pd.concat([all_ASR, sam_phen], axis=0)
    if len(error_id) == 0:
        error_id = None
    all_ASR['platform'].fillna(all_ASR['platform '], inplace=True)
    all_ASR['biosample_id'].fillna(all_ASR['bioSample_ID'], inplace=True)
    all_ASR['units'].fillna(all_ASR['measurement_units'], inplace=True)
    all_ASR['measurement_type'].fillna(all_ASR['laboratory_typing_method'], inplace=True)
    all_ASR.drop(['Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
               'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
               'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1, inplace=True)    
    all_ASR['units'].replace(to_replace='mg/l', value='mg/L', inplace=True)
    all_ASR['measurement_has_/'].fillna(False, inplace=True)
    all_ASR.drop(['platform ', 'bioSample_ID', 'measurement_units', 'laboratory_typing_method', ], axis=1, inplace=True)
    # all_ASR.sort_values('measurement', ascending=False).drop_duplicates(subset=['biosample_id', 'antibiotic_name'], keep='first', inplace=True)
    
    return all_ASR, error_id


def load_all_phen_data(
    path, 
    run2bio_match_path=None,
    file_format='PATAKI'
):
    try:
        if file_format=='PATAKI':
            phen_df = pd.read_csv(path, sep='\t')
        elif file_format=='VAMP':
            phen_df = pd.read_csv(path, sep=',')
        elif file_format=='PATRIC':
            phen_df = pd.read_excel(path)
            phen_df['genome_id'] = phen_df['genome_id'].astype(str)
            run2biosam = pd.read_excel(run2bio_match_path)
            run2biosam['PATRIC_ID'] = run2biosam['PATRIC_ID'].astype(str)
            run2biosam = run2biosam[['PATRIC_ID', 'sample_accession', 'Species']]
            run2biosam.columns = ['genome_id', 'biosample_id', 'species_fam']
            run2biosam['genome_id'] = run2biosam['genome_id'].astype(str)
            phen_df = phen_df.merge(right=run2biosam, how='inner', on='genome_id').drop(['genome_id'], axis=1)
            phen_df.rename(columns = {'antibiotic': 'antibiotic_name'}, inplace = True)
    except:
        # print(path + ': problem loading file')
        return path
    if file_format=='PATAKI':
        biosample_id = phen_df['bioSample_ID'][0]
        species_fam = phen_df['species'][0]
    elif file_format=='VAMP':
        phen_df.columns = [
            'antibiotic_name', 
            'resistance_phenotype', 
            'measurement_sign', 
            'measurement', 
            'units', 
            'measurement_type',
            '1', 
            '1.1', 
            '1.2', 
            'ast_standard',
        ]
        
        biosample_id = path.split('/')[-1].split('.')[0]
        phen_df['biosample_id'] = biosample_id
    
    phen_df['measurement_sign'].fillna('=', inplace=True)
    phen_df['measurement_sign'].replace(inplace=True, to_replace='==', value='=')
    
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].str.lower()
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace(' ', '_', regex=True)
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace('-', '_', regex=True)
    
    if file_format == 'PATRIC':
        phen_df['measurement'] = phen_df.apply(lambda row: fix_PATRIC_MIC_value(row, '1'), axis=1)
        phen_df['measurement2'] = phen_df.apply(lambda row: fix_PATRIC_MIC_value(row, '2'), axis=1)
        phen_df['measurement_has_/'] = phen_df.apply(lambda row: fix_PATRIC_MIC_value(row, 'has'), axis=1)
        
    if phen_df['measurement'].dtype == object:
        phen_df['measurement_has_/'] = phen_df['measurement'].apply(lambda x: len(re.findall("/(\\d+\.?\\d*)", str(x)))>0)
        phen_df['measurement2'] = phen_df['measurement'].apply(get_mes2)
        phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
        
    phen_df['DB'] = file_format
    
    return phen_df


def save_dataset(tot_data, labels, features, ds_name):
    full_ds_path = '../resources/{}'.format(ds_name)
    os.makedirs(full_ds_path, exist_ok=True)
    
    tot_data.to_csv('{}/tot_filtered_data.csv'.format(full_ds_path))
    with open("{}/labels".format(full_ds_path), "wb") as fp:
        pickle.dump(labels, fp)
    with open("{}/features".format(full_ds_path), "wb") as fp:
        pickle.dump(features, fp)

def merging_dbs(filtered_data_list, labels_list, features_list):
    tot_data = pd.concat(filtered_data_list, axis=0)
    tot_data = tot_data.drop_duplicates(subset=['biosample_id'], keep='first')

    labels = {}
    for label in labels_list:
        labels = list(set(labels).union(set(label)))
    
    features = {}
    for feature in features_list:
        features = list(set(features).union(set(feature)))
        
    return tot_data, labels, features
    

def merge_gen2phen(geno, pheno, run2bio_match_path, filter_list_path, db_name):
    run2biosam = pd.read_excel(run2bio_match_path)
    if db_name == 'PATRIC':
        run2biosam['PATRIC_ID'] = run2biosam['PATRIC_ID'].astype(str)
        run2biosam = run2biosam[['run_accession', 'PATRIC_ID', 'sample_accession', 'Species']]
        run2biosam.columns = ['run_id', 'genome_id', 'biosample_id', 'species_fam']
        run2biosam['genome_id'] = run2biosam['genome_id'].astype(str)
    else:
        run2biosam.columns = ['run_id', 'biosample_id']
        filtered_data = pd.read_excel(filter_list_path)
        filtered_data.columns = ['species_fam', 'run_id']
        
    if db_name == 'PATAKI':
        pheno = pheno.drop(['species_fam'], axis=1)
        
    labels = list(set(pheno.columns)-set(['biosample_id', 'genome_id']))
    
    if db_name == 'PATRIC':
        filtered_data = run2biosam.merge(right=pheno, how='inner', on='genome_id')
        filtered_data = filtered_data.drop(['genome_id'], axis=1)
    else:
        filtered_data = filtered_data.merge(right=run2biosam, how='inner', on='run_id')
        filtered_data = filtered_data.merge(right=pheno, how='inner', on='biosample_id')
    filtered_data = filtered_data.merge(right=geno, how='inner', on='run_id')
    filtered_data['DB'] = db_name
    filtered_data = filtered_data.drop_duplicates(subset='biosample_id', keep='first')
    
    features = list(set(geno.columns)-set(['run_id']))
    return filtered_data, labels, features
    

'''
db_name - either Pataki/VAMP for now
'''
def get_phenotype_per_db(AST_dir, db_name='PATAKI'):
    print('updated')
    if db_name == 'VAMP' or db_name == 'PATAKI':
        phenotypic = pd.DataFrame({})
        error_id = []
        for sam_dir in tqdm(os.listdir(AST_dir)):
            sam_labels = get_isolate_labels(AST_dir+'/'+sam_dir, file_format=db_name)
            if type(sam_labels) is not str:
                phenotypic = pd.concat([phenotypic, sam_labels], axis=0)
            else:
                print(sam_dir)
                error_id += [sam_dir]
        if len(error_id) == 0:
            error_id = None
    elif db_name == 'PATRIC':
        phenotypic = pd.read_excel(AST_dir)
        phenotypic['antibiotic'] = phenotypic['antibiotic'].str.lower()
        phenotypic['antibiotic'] = phenotypic['antibiotic'].replace(' ', '_', regex=True)
        phenotypic['antibiotic'] = phenotypic['antibiotic'].replace('-', '_', regex=True)
        phenotypic['measurement_sign'].fillna('=', inplace=True)
        phenotypic['measurement_sign'].replace('==', '=', inplace=True)
        phenotypic['genome_id'] = phenotypic['genome_id'].astype(str)
        phenotypic['measurement'] = phenotypic.apply(fix_PATRIC_MIC_value, axis=1)
        phenotypic.drop_duplicates(subset=['genome_id', 'antibiotic'], keep='first', inplace=True)
        phenotypic = phenotypic.pivot(index='genome_id', values='measurement', columns = 'antibiotic')
        error_id = None
    else:
        print('error parsing data')
    return (phenotypic, error_id)


'''
'''
def get_genotype_per_db(path):
    genotypic = pd.DataFrame({})
    error_id = []
    for SRR_dir in tqdm(os.listdir(path)):
        srr_features = get_isolate_features(path+'/'+SRR_dir)
        if type(srr_features) is not str:
            genotypic = pd.concat([genotypic, srr_features], axis=0)
        else:
            error_id += [srr_features]
    if len(error_id) == 0:
        error_id = None
    return genotypic, error_id


'''
A function that parse the raw WGS matching output and turns it to model features per sample.
'''
def get_isolate_features(
    path, 
    with_confidence=False, 
    sortby='SeqCov', 
    cov_thresh=None, 
    id_thresh=None, 
    depth_thresh=None, 
):
    if sortby is None:
        sortby = 'SeqCov'
    if with_confidence is None:
        with_confidence = False
    run_id = re.findall("(.RR\\d+)\.", path)[0]
    try:
        csv_file = glob.glob(path+'/*.csv')[0]
    except:
        print(run_id + ': is missing csv file')
        return run_id
    gene_df = pd.read_csv(csv_file)
    
    if cov_thresh is not None:
        gene_df = gene_df[gene_df['SeqCov']>cov_thresh]
    if id_thresh is not None:
        gene_df = gene_df[gene_df['SeqID']>id_thresh]
    if depth_thresh is not None:
        gene_df = gene_df[gene_df['Depth']>depth_thresh]
        
    gene_df = gene_df.groupby(by='Gene').apply(lambda x: x.iloc[x[sortby].argmax()])
    gene_df = gene_df[['Contig', 'Start', 'End', 'Depth', 'SeqID', 'SeqCov', 'Match_Start', 'Match_End', 'Ref_Gene_Size']].reset_index()
    
    try:
        txt_file = glob.glob(path+'/*.txt')[0]
    except:
        print(run_id + ': is missing txt file')
        return run_id
    contig_df = pd.read_csv(txt_file, sep=" ", header=None)
    contig_df = contig_df.iloc[:,:3]
    contig_df.columns = ['contig_id', 'contig_size', 'avg_depth']
    contig_df['contig_id'] = contig_df['contig_id'].apply(lambda x: x[1:])
    contig_df['contig_size'] = contig_df['contig_size'].apply(lambda x: int(x[4:]))
    contig_df['avg_depth'] = contig_df['avg_depth'].apply(lambda x: float(x[4:]))

    features = gene_df.merge(right=contig_df, how='left', left_on='Contig', right_on='contig_id').drop(['Contig', 'contig_id'], axis=1)
    features['dist_contig_end'] = np.minimum(features['Start'], features['contig_size']-features['End'])
    features['contig_end_partition'] = features['dist_contig_end'] / features['contig_size']
    features['relative_depth'] = features['Depth'] / features['avg_depth']
    features = features[['Gene', 'Depth', 'avg_depth', 'relative_depth', 'dist_contig_end', 'contig_end_partition', 'SeqID', 'SeqCov']]
    features.columns = ['gene', 'depth', 'avg_depth', 'relative_depth', 'dist_contig_end', 'contig_end_partition', 'seq_id', 'seq_cov']
    if not with_confidence:
        features = features.drop(['relative_depth', 'depth', 'avg_depth', 'dist_contig_end', 'contig_end_partition'], axis=1)
    features = features.set_index('gene').stack().reset_index()
    features['col_name'] = features['gene'].values + '->' + features['level_1'].values
    features = features.set_index('col_name').loc[:, 0].to_frame().T
    features['run_id'] = run_id
    return features


def get_isolate_labels(
    path, 
    MIC_val=True, 
    log_scale=True,
    file_format='PATAKI'
):
    try:
        if file_format=='PATAKI':
            phen_df = pd.read_csv(path, sep='\t')
        elif file_format=='VAMP':
            phen_df = pd.read_csv(path, sep=',')
    except:
        print(path + ': problem loading file')
        return path
    if file_format=='PATAKI':
        biosample_id = phen_df['bioSample_ID'][0]
        species_fam = phen_df['species'][0]        
    elif file_format=='VAMP':
        phen_df.columns = [
            'antibiotic_name', 
            'resistance_phenotype', 
            'measurement_sign', 
            'measurement', 
            'units', 
            'measurement_type',
            '1', 
            '1.1', 
            '1.2', 
            'MIC_format',
        ]
        biosample_id = path.split('/')[-1].split('.')[0]
        phen_df['measurement_sign'].replace(inplace=True, to_replace='==', value='=')
        
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].str.lower()
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace(' ', '_', regex=True)
    phen_df['antibiotic_name'] = phen_df['antibiotic_name'].replace('-', '_', regex=True)
    if MIC_val:
        phen_df = phen_df[['antibiotic_name', 'measurement', 'measurement_sign']]
    else:
        phen_df = phen_df[['antibiotic_name', 'resistance_phenotype']]
    
    if phen_df['measurement'].dtype == object:
        print(biosample_id + ': not excepted measure values')
        phen_df['measurement'] = phen_df['measurement'].apply(lambda x: float(re.findall("(\\d+\.?\\d*)", str(x))[0]))
    
    if log_scale:
        try:
            phen_df['label'] = np.log2(phen_df['measurement'])
        except:
            print('log still Error')
    else:
        phen_df['label'] = phen_df['measurement']
        
    phen_df['label'] =  phen_df['measurement_sign'].values + ' ' + phen_df['label'].astype(str).values
    phen_df = phen_df.sort_values('measurement', ascending=False).drop_duplicates(subset=['antibiotic_name'], keep='first')
    
    labels = phen_df[['label', 'antibiotic_name']].set_index('antibiotic_name').T
    labels['biosample_id'] = biosample_id
    if file_format=='PATAKI':
        labels['species_fam'] = species_fam.lower()
    labels.set_index('biosample_id', inplace=True)
    return labels


def get_mes2(x):
    if len(re.findall("/(\\d+\.?\\d*)", str(x)))>0:
        return float(re.findall("/(\\d+\.?\\d*)", str(x))[0])
    else:
        return np.nan


def fix_PATRIC_MIC_value(row, ans_type='1', log2=True, choose_first_dash=True):
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

    # if len(values_str)>1:
    #     return '{} {}/{}'.format(sign, values_str[0], values_str[1])
    # else:
        # return '{} {}'.format(sign, values_str[0])


def save_species_dict(path="../resources/species_dict.json"):
    species_dict = {
        'escherichia coli': 'Escherichia coli',
        'Salmonella enterica subsp. enterica serovar Typhimurium': 'Salmonella enterica',
        'senterica': 'Salmonella enterica',
        'abaumannii': 'Acinetobacter baumannii',
        'ecoli': 'Escherichia coli',
        'spneumoniae': 'Streptococcus pneumoniae',
        'kpneumoniae': 'Klebsiella pneumoniae',
        'paeruginosa': 'Pseudomonas aeruginosa',
        'Pseudomonas aeuruginosa': 'Pseudomonas aeruginosa',
        'saureus': 'Staphylococcus aureus',
        'ecloacae': 'Enterobacter sp.',
        'Enterobacter hormaechei': 'Enterobacter sp.',
        'Enterobacter cloacae': 'Enterobacter sp.',
        'Enterobacter roggerkampi': 'Enterobacter sp.',
        'Enterobacter asburiae': 'Enterobacter sp.',
        'Enterobacter chengduensi': 'Enterobacter sp.',
        'Enterobacter bugandensis': 'Enterobacter sp.',
        'Enterobacter sichuanensis': 'Enterobacter sp.',
        'Enterobacter kobei': 'Enterobacter sp.',
    }
    with open(path, "w") as fp:
        json.dump(species_dict, fp)


def save_resources_dict(path="../resources/resources_dict.json"):
    resources_dict = {
        'PATAKI': {
            'geno': '../resources/28.12.21/Pataki_paper/PATAKI_final_for_Amit.2021.12.28/Pataki.results.for.Amit',
            'pheno': '../resources/26.12.21/Pataki_paper/AST_2548_all',
            'run2bio': '../resources/28.12.21/Pataki_paper/PATAKI_final_for_Amit.2021.12.28/PATAKI_full_SAM_and_SRR_list.xlsx',
            'filter_list': '../resources/28.12.21/Pataki_paper/PATAKI_final_for_Amit.2021.12.28/PATAKI_filtered_SRR_list_for_Amit.xlsx',
        },
        'VAMP': {
            'geno': '../resources/28.12.21/VAMPr_3400samples/VAMP_final_for_Amit.2021.12.28/VAMPr.results.for.Amit',
            'pheno': '../resources/28.12.21/VAMPr_3400samples/VAMP_final_for_Amit.2021.12.28/VAMP_full_AST_data',
            'run2bio': '../resources/28.12.21//VAMPr_3400samples/VAMP_final_for_Amit.2021.12.28/VAMP_full_SAM_and_SRR_list.csv',
            'filter_list': '../resources/28.12.21//VAMPr_3400samples/VAMP_final_for_Amit.2021.12.28/VAMP_filtered_SRR_list.20211228.xlsx',
        },
        'PA': {
            'geno': "../resources/data/PA.dataset.400.for.Amit/",
            'pheno': '../resources/data/Pseudomonas_paper_AST.xlsx',
            'run2bio': '../resources/data/PA.dataset.400.RunInfo.xlsx',
            'filter_list': '',
        },
        'PATRIC': {
            'geno': '/sise/liorrk-group/AmitdanwMaranoMotroy/all.QC.passed.spades.20220313/',
            'pheno': '../resources/data/PATRIC_AMR_ESKAPE_etal_with_numericalAST_only.xlsx',
            'run2bio': '../resources/data/PATRIC_genome_final_db.20220223.xlsx',
            'filter_list': '',
        },
    }
    with open(path, "w") as fp:
        json.dump(resources_dict, fp)