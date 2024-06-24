#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='muted', font='sans-serif',
              font_scale=1, color_codes=True, rc=None)
sns.set_style("whitegrid")
sns.despine(left=True)     
# sns.set()
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import ttest_ind, ttest_rel
import json
def remove_duplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.get_legend().remove()
    ax.legend(by_label.values(), by_label.keys(), prop={'size': 10})
    return ax
def print_terminal_and_file(text, path, sentinel='a'):
    print( text ) 
    with open(path, sentinel) as text_file:
        print( text, file=text_file ) 
"""----- SETTINGS  -----"""


deltaStep = 1

distancesForFaiRatio = [ [2,3] ,[5,6], [10,11], [12,13] ]
min_volume_far = 300
min_volume_close = 300
y_lim_diff = (-7,23)
y_lim_ratio = (0.71,1.1)

no_legend = True
inputFolder = None#'NIFTI'

outputFolder_list = [
                        'BAA_vs_K_Bifurc-Truncus_ohneBAABereich',
                        'BAA_vs_K_Bifurc-Truncus_mitBAABereich',
                        'BAA_vs_K_BAA-Bereich_vs_Bifurc-Tuncus',
                        'BAA_vs_K_BAA-Bereich_vs_Bifurc-Nierenarterie',
                        'BAA_Bereich_vs_Bifurc-Truncus_intraindividuell',
                     ]



colorsPlot_list = ['tab:blue', 'tab:orange']

additionalKeyAppendix = '_fat_'
valToAnalyse = '_meanHU'

#get all possible ranges
all_ranges = []
for i in np.arange(distancesForFaiRatio[0][0], distancesForFaiRatio[0][1], deltaStep):
    for j in np.arange(distancesForFaiRatio[1][0], distancesForFaiRatio[1][1], deltaStep):
        for k in np.arange(distancesForFaiRatio[2][0], distancesForFaiRatio[2][1], deltaStep):
            for m in np.arange(distancesForFaiRatio[3][0], distancesForFaiRatio[3][1], deltaStep):
                if i < j and j < k and k < m: all_ranges.append( [i,j,k,m] )


for outputFolder in outputFolder_list:
    

    if outputFolder == 'BAA_vs_K_BAA-Bereich_vs_Bifurc-Tuncus':
        independent = True
        excelFile_appendix_list = ['BAA_', 'K_']
        group_appendix_list = ['BAA', 'K']
        nameSectionToCompare_list = ['unteres_Ende_BAA-oberes_Ende_BAA',
                                     'Bifurkation-Truncus_coeliacus'] 
    elif outputFolder == 'BAA_vs_K_BAA-Bereich_vs_Bifurc-Nierenarterie':
        independent = True
        excelFile_appendix_list = ['BAA_', 'K_']
        group_appendix_list = ['BAA', 'K']
        nameSectionToCompare_list = ['unteres_Ende_BAA-oberes_Ende_BAA',
                                     'Bifurkation-Abgang_rechte_Nierenarterie'] 
                                             
    elif outputFolder == 'BAA_vs_K_Bifurc-Truncus_mitBAABereich':
        independent = True
        excelFile_appendix_list = ['BAA_', 'K_']
        group_appendix_list = ['BAA', 'K']
        nameSectionToCompare_list = ['Bifurkation-Truncus_coeliacus_mitBAA',
                                      'Bifurkation-Truncus_coeliacus'] 
    elif outputFolder == 'BAA_vs_K_Bifurc-Truncus_ohneBAABereich':
        independent = True
        excelFile_appendix_list = ['BAA_', 'K_']
        group_appendix_list = ['BAA', 'K']
        nameSectionToCompare_list = [ 'Bifurkation-Truncus_coeliacus',
                                      'Bifurkation-Truncus_coeliacus']     
   
    elif outputFolder == 'BAA_Bereich_vs_Bifurc-Truncus_intraindividuell':
        independent = False
        excelFile_appendix_list = ['BAA_', 'BAA_']
        group_appendix_list = ['BAA', 'bifurc-Truncus']
        nameSectionToCompare_list = ['unteres_Ende_BAA-oberes_Ende_BAA',
                                      'Bifurkation-Truncus_coeliacus'] 

    
    """----- COMPUTE  -----"""
    

    basePath = os.path.dirname(os.path.realpath(__file__))+os.sep
    inputPath = basePath
    if inputFolder is not None: inputPath += inputFolder+os.sep
    
    totalDiffsBetweenGroups = []
    totalDiffFaiRatiosBetweenGroups = []
    pvals_diffsBetweenGroups = []
    pvals_faiRatiosBetweenGroups = []
    meanValuesClose0 = []
    meanValuesClose1 = []
    meanValuesFar0 = []
    meanValuesFar1 = []
    meanDiff0 = []
    meanDiff1 = []  
    meanRatio0 = []
    meanRatio1 = []  
    meanVolumeClose0 = []
    meanVolumeClose1 = []
    meanVolumeFar0 = []
    meanVolumeFar1 = [] 
    stdValuesClose0 = []
    stdValuesClose1 = []
    stdValuesFar0 = []
    stdValuesFar1 = []
    stdDiff0 = []
    stdDiff1 = []  
    stdRatio0 = []
    stdRatio1 = []  
    stdVolumeClose0 = []
    stdVolumeClose1 = []
    stdVolumeFar0 = []
    stdVolumeFar1 = [] 
    pvals_meanValuesCloseBetweenGroups = []
    pvals_meanValuesFarBetweenGroups = []
    numFailed0 = []
    numFailed1 =[]
    all_foldernames = []
    
    for curRange in tqdm(all_ranges):
        
        folder_str = '_'.join([ str(x) for x in curRange]).replace('.0','').replace('.','-')
        all_foldernames.append(folder_str)
        outputFolder_path = inputPath+outputFolder+os.sep
        outputPath = outputFolder_path+folder_str+os.sep
        log_path = outputFolder_path+folder_str+os.sep+folder_str+'.txt'
        Path(outputPath).mkdir(parents=True, exist_ok=True)
        print_terminal_and_file('', log_path, sentinel='w')
        
        dict_out = { 'excelFile_appendix' : excelFile_appendix_list,
                    'group_appendix' : group_appendix_list,
                    'sections': nameSectionToCompare_list,
                    'valToAnalyse': valToAnalyse,
                    'color' : colorsPlot_list,
                    'distances' : [],
                    'values' : [],
                    'volumes': [],
                    'distances_close' : [],
                    'distances_far' : [],
                    'steps_close' : [],
                    'steps_far' : [],
                    'volume_close' : [],
                    'volume_far' : [],
                    'meanValues_close' : [],
                    'meanValues_far' : [],
                    'FaiRatios' : [],
                    'diff' : [],
                    'names' : []
                    }
        failed = False
        for gr_idx in range(len(group_appendix_list)):
            
            excelFile_appendix = excelFile_appendix_list[gr_idx]
            nameSectionToCompare = nameSectionToCompare_list[gr_idx]
        
            df = pd.read_excel(inputPath+excelFile_appendix+'results.xlsx')
            keys = [ key for key in df if nameSectionToCompare in key]
            keys_splitForStep = [x.split(nameSectionToCompare)[-1] for x in keys]
            
            all_steps = [ re.findall(r'\d+', x) for x in keys_splitForStep if valToAnalyse in x]
            all_steps = [ int(x[-1])  for x in all_steps if x != []  ]
            all_steps = np.unique(np.array(all_steps))
            
            distances_perCase = []
            values_perCase = []
            volumes_perCase = []
            steps_close_perCase = []
            steps_far_perCase = []
            distances_close_perCase = []
            distances_far_perCase = []
            meanValues_close_perCase = []
            meanValues_far_perCase = []
            volume_close_perCase = []
            volume_far_perCase = [] 
            FaiRatios_perCase = []
            diff_perCase = []
            numFailed_perCase = 0
            
            for i in range(len(df)):
                curDistances = all_steps*df['numVoxelOneStep'][i]*df['voxSpacing_x'][i]

                cur_steps_close_perCase = [ x for x,y in zip(all_steps,curDistances) if (y > int(curRange[0]) and y <= int(curRange[1])) ]
                cur_steps_far_perCase = [ x for x,y in zip(all_steps,curDistances) if (y > int(curRange[2]) and y <= int(curRange[3])) ]
                
                if len(cur_steps_close_perCase) == 0 or len(cur_steps_far_perCase) == 0:
                    numFailed_perCase += 1
                    values_perCase.append( np.nan )
                    volumes_perCase.append( np.nan )
                    distances_perCase.append( np.nan )
                    steps_close_perCase.append( np.nan )
                    steps_far_perCase.append( np.nan )
                    distances_close_perCase.append( np.nan )
                    distances_far_perCase.append( np.nan )
                    volume_close_perCase.append( np.nan )
                    volume_far_perCase.append( np.nan )
                    meanValues_close_perCase.append( np.nan )
                    meanValues_far_perCase.append( np.nan )
                    FaiRatios_perCase.append( np.nan )
                    diff_perCase.append( np.nan )
                    continue

                cur_distances_close_perCase = list(np.array(cur_steps_close_perCase)*df['numVoxelOneStep'][i]*df['voxSpacing_x'][i])
                cur_distances_far_perCase = list(np.array(cur_steps_far_perCase)*df['numVoxelOneStep'][i]*df['voxSpacing_x'][i])
            
                curDistances = cur_distances_close_perCase+cur_distances_far_perCase
            
                keys_close = [ nameSectionToCompare+additionalKeyAppendix+str(x)+valToAnalyse for x in cur_steps_close_perCase ]
                keys_volumes_close = [ nameSectionToCompare+additionalKeyAppendix+str(x)+'_volume' for x in cur_steps_close_perCase ]
                
                keys_far = [ nameSectionToCompare+additionalKeyAppendix+str(x)+valToAnalyse for x in cur_steps_far_perCase ]
                keys_volumes_far = [ nameSectionToCompare+additionalKeyAppendix+str(x)+'_volume' for x in cur_steps_far_perCase ]
                
                keys_all = keys_close+keys_far
                keys_volumes_all = keys_volumes_close+keys_volumes_far

                total_volume_close = 0
                for key in keys_volumes_close: total_volume_close += df[key][i]
                if total_volume_close < min_volume_close: 
                    print_terminal_and_file(f'EXCLUDED {df["name"][i]}: only volume_close {total_volume_close}', log_path)
                    numFailed_perCase += 1
                    values_perCase.append( np.nan )
                    volumes_perCase.append( np.nan )
                    distances_perCase.append( np.nan )
                    steps_close_perCase.append( np.nan )
                    steps_far_perCase.append( np.nan )
                    distances_close_perCase.append( np.nan )
                    distances_far_perCase.append( np.nan )
                    volume_close_perCase.append( np.nan )
                    volume_far_perCase.append( np.nan )
                    meanValues_close_perCase.append( np.nan )
                    meanValues_far_perCase.append( np.nan )
                    FaiRatios_perCase.append( np.nan )
                    diff_perCase.append( np.nan )
                    continue
                
                # weight by voxels per key
                val_volumeWeighted_close = 0
                for key, key_volume in zip(keys_close,keys_volumes_close): 
                    val_volumeWeighted_close += df[key][i]*df[key_volume][i]
                val_volumeWeighted_close /= total_volume_close

                total_volume_far = 0
                for key in keys_volumes_far: total_volume_far += df[key][i]
                if total_volume_far < min_volume_far: 
                    print_terminal_and_file(f'EXCLUDED {df["name"][i]}: only volume_far {total_volume_far}', log_path)
                    numFailed_perCase += 1
                    values_perCase.append( np.nan )
                    volumes_perCase.append( np.nan )
                    distances_perCase.append( np.nan )
                    steps_close_perCase.append( np.nan )
                    steps_far_perCase.append( np.nan )
                    distances_close_perCase.append( np.nan )
                    distances_far_perCase.append( np.nan )
                    volume_close_perCase.append( np.nan )
                    volume_far_perCase.append( np.nan )
                    meanValues_close_perCase.append( np.nan )
                    meanValues_far_perCase.append( np.nan )
                    FaiRatios_perCase.append( np.nan )
                    diff_perCase.append( np.nan )
                    continue
            
                val_volumeWeighted_far = 0
                for key, key_volume in zip(keys_far,keys_volumes_far) : 
                    val_volumeWeighted_far += df[key][i]*df[key_volume][i]
                val_volumeWeighted_far /= total_volume_far

                cur_values_perCase = []
                for key in keys_all:
                    cur_values_perCase.append( df[key][i] )
                cur_volumes_perCase = []
                for key in keys_volumes_all:
                    cur_volumes_perCase.append( df[key][i] )
                    
                values_perCase.append( cur_values_perCase )
                volumes_perCase.append( cur_volumes_perCase )
                distances_perCase.append( curDistances )
                steps_close_perCase.append( cur_steps_close_perCase )
                steps_far_perCase.append( cur_steps_far_perCase )
                distances_close_perCase.append( cur_distances_close_perCase )
                distances_far_perCase.append( cur_distances_far_perCase )
                volume_close_perCase.append( total_volume_close )
                volume_far_perCase.append( total_volume_far )
                meanValues_close_perCase.append( val_volumeWeighted_close )
                meanValues_far_perCase.append( val_volumeWeighted_far )
                FaiRatios_perCase.append(val_volumeWeighted_close/val_volumeWeighted_far)
                diff_perCase.append(val_volumeWeighted_close-val_volumeWeighted_far)
        
            # if failed: break
        
            dict_out['names'].append( df['name'] )
            dict_out['values'].append( values_perCase )
            dict_out['volumes'].append( volumes_perCase )
            dict_out['steps_close'].append( steps_close_perCase )
            dict_out['steps_far'].append( steps_far_perCase )
            dict_out['distances'].append( distances_perCase )
            dict_out['distances_close'].append( distances_close_perCase )
            dict_out['distances_far'].append( distances_far_perCase )
            dict_out['volume_close'].append( volume_close_perCase )
            dict_out['volume_far'].append( volume_far_perCase )
            dict_out['meanValues_close'].append( meanValues_close_perCase )
            dict_out['meanValues_far'].append( meanValues_far_perCase )
            dict_out['FaiRatios'].append(FaiRatios_perCase)
            dict_out['diff'].append(diff_perCase)  

            if gr_idx == 0: numFailed0.append( int(numFailed_perCase) )
            else: numFailed1.append( int(numFailed_perCase ) )
            

        if len(dict_out['values']) == 0:
            totalDiffsBetweenGroups.append(np.nan)
            totalDiffFaiRatiosBetweenGroups.append(np.nan)
            pvals_diffsBetweenGroups.append(np.nan)
            pvals_faiRatiosBetweenGroups.append(np.nan)
            meanValuesClose0.append(np.nan)
            meanValuesClose1.append(np.nan)
            meanValuesFar0.append(np.nan)
            meanValuesFar1.append(np.nan)
            meanVolumeClose0.append(np.nan)
            meanVolumeClose1.append(np.nan)
            meanDiff0.append(np.nan)
            meanDiff1.append(np.nan)
            meanRatio0.append(np.nan)
            meanRatio1.append(np.nan) 
            meanVolumeFar0.append(np.nan)
            meanVolumeFar1.append(np.nan)
            stdValuesClose0.append(np.nan)
            stdValuesClose1.append(np.nan)
            stdValuesFar0.append(np.nan)
            stdValuesFar1.append(np.nan)
            stdVolumeClose0.append(np.nan)
            stdVolumeClose1.append(np.nan)
            stdVolumeFar0.append(np.nan)
            stdVolumeFar1.append(np.nan)
            stdDiff0.append(np.nan)
            stdDiff1.append(np.nan)
            stdRatio0.append(np.nan)
            stdRatio1.append(np.nan)
            pvals_meanValuesCloseBetweenGroups.append(np.nan)
            pvals_meanValuesFarBetweenGroups.append(np.nan)
            continue
        
        
        fig, ax = plt.subplots(1,1)
        for gr_idx in range(len(dict_out['group_appendix'])-1, -1, -1):
            for values, distances in zip(dict_out['values'][gr_idx], dict_out['distances'][gr_idx]):
                line, = ax.plot(distances, values, dict_out['color'][gr_idx])
            line.set_label( dict_out['group_appendix'][gr_idx].replace('_','') )
            ax.legend()
        ax.set_xlabel('distance to vessel [mm]')#, fontsize=cfg.fontsize_axis)
        ax.set_ylabel(valToAnalyse.replace('_',''))#, fontsize=cfg.fontsize_axis)
        if no_legend: ax.get_legend().remove()
        fig.savefig(outputPath+'FAI_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig)

        gr_str = dict_out['group_appendix'][0].replace('_','')
        dict_out['FaiRatios_all'] = []
        dict_out['volume_close_all'] = []
        dict_out['volume_far_all'] = []
        dict_out['meanValues_close_all'] = []
        dict_out['meanValues_far_all'] = []
        dict_out['diff_all'] = []
        dict_out['group_all'] = []
        keys_toCheckForOutliers = [ 'volume_close', 'volume_far', 'meanValues_close', 'meanValues_far', 'FaiRatios','diff' ]
  
        for i in range(len(dict_out['group_appendix'])):
            names = np.asarray(dict_out['names'][i])
            dict_out['FaiRatios_all'] += dict_out['FaiRatios'][i]
            dict_out['diff_all'] += dict_out['diff'][i]
            dict_out['volume_close_all'] += dict_out['volume_close'][i]
            dict_out['volume_far_all'] += dict_out['volume_far'][i]
            dict_out['meanValues_close_all'] += dict_out['meanValues_close'][i]
            dict_out['meanValues_far_all'] += dict_out['meanValues_far'][i]
            
            for key in keys_toCheckForOutliers:
                mean = np.nanmean(dict_out[key][i])
                std =  np.nanstd(dict_out[key][i])
                
                if 'volume' in key:
                    range_ok = [1000, np.inf]
                    print_terminal_and_file(f'Outliers {key}: volume < {range_ok[0]}', log_path)
                else:
                    range_ok = [mean-2*std, mean+2*std]
                    print_terminal_and_file(f'Outliers {key}: mean {np.round(mean,2)} std {np.round(std,2)}', log_path)
                for name_idx, name in enumerate(names):
                    if dict_out[key][i][name_idx] < range_ok[0] or dict_out[key][i][name_idx] > range_ok[1]:
                        print_terminal_and_file(f'{name}={np.round(dict_out[key][i][name_idx],2)}', log_path)

                            
            group_appendix = dict_out['group_appendix'][i]
            if gr_str == group_appendix: group_list = [ gr_str for x in dict_out['FaiRatios'][i]]
            else: group_list = [ dict_out['group_appendix'][1].replace('_','') for x in dict_out['FaiRatios'][i]]
            dict_out['group_all'] += group_list        
        
        dict_out['group_all'] = pd.Categorical(dict_out['group_all'],
                                                ordered=False,
                                                categories=[gr_str, dict_out['group_appendix'][1].replace('_','')])
        
            
        
        if independent: ttest = ttest_ind
        else: ttest = ttest_rel
        
        isGroup0 = dict_out['group_all'] == dict_out['group_appendix'][0]
                                          
        totalDiffsBetweenGroups.append( np.nanmean(np.asarray(dict_out['diff_all'])[isGroup0]) - \
                                        np.nanmean(np.asarray(dict_out['diff_all'])[~isGroup0]) )
        
        totalDiffFaiRatiosBetweenGroups.append( np.nanmean(np.asarray(dict_out['FaiRatios_all'])[isGroup0]) - \
                                               np.nanmean(np.asarray(dict_out['FaiRatios_all'])[~isGroup0]) )      
            
        pvals_diffsBetweenGroups.append( ttest(np.asarray(dict_out['diff_all'])[isGroup0],
                                               np.asarray(dict_out['diff_all'])[~isGroup0] ,
                                               nan_policy='omit').pvalue   ) 
        
        pvals_faiRatiosBetweenGroups.append( ttest(np.asarray(dict_out['FaiRatios_all'])[isGroup0],
                                                   np.asarray(dict_out['FaiRatios_all'])[~isGroup0] ,
                                                   nan_policy='omit').pvalue   ) 
        
        meanVolumeClose0.append( np.nanmean(np.asarray(dict_out['volume_close_all'])[isGroup0]))
        meanVolumeClose1.append( np.nanmean(np.asarray(dict_out['volume_close_all'])[~isGroup0]))
        meanVolumeFar0.append( np.nanmean(np.asarray(dict_out['volume_far_all'])[isGroup0]))
        meanVolumeFar1.append( np.nanmean(np.asarray(dict_out['volume_far_all'])[~isGroup0]))
        meanValuesClose0.append( np.nanmean(np.asarray(dict_out['meanValues_close_all'])[isGroup0]))
        meanValuesClose1.append( np.nanmean(np.asarray(dict_out['meanValues_close_all'])[~isGroup0]))
        meanValuesFar0.append( np.nanmean(np.asarray(dict_out['meanValues_far_all'])[isGroup0]))
        meanValuesFar1.append( np.nanmean(np.asarray(dict_out['meanValues_far_all'])[~isGroup0]))
        
        meanDiff0.append( np.nanmean(np.asarray(dict_out['diff_all'])[isGroup0]))
        meanDiff1.append( np.nanmean(np.asarray(dict_out['diff_all'])[~isGroup0]))
        meanRatio0.append( np.nanmean(np.asarray(dict_out['FaiRatios_all'])[isGroup0]))
        meanRatio1.append( np.nanmean(np.asarray(dict_out['FaiRatios_all'])[~isGroup0]))
         
        stdVolumeClose0.append( np.nanstd(np.asarray(dict_out['volume_close_all'])[isGroup0]))
        stdVolumeClose1.append( np.nanstd(np.asarray(dict_out['volume_close_all'])[~isGroup0]))
        stdVolumeFar0.append( np.nanstd(np.asarray(dict_out['volume_far_all'])[isGroup0]))
        stdVolumeFar1.append( np.nanstd(np.asarray(dict_out['volume_far_all'])[~isGroup0]))
        stdValuesClose0.append( np.nanstd(np.asarray(dict_out['meanValues_close_all'])[isGroup0]))
        stdValuesClose1.append( np.nanstd(np.asarray(dict_out['meanValues_close_all'])[~isGroup0]))
        stdValuesFar0.append( np.nanstd(np.asarray(dict_out['meanValues_far_all'])[isGroup0]))
        stdValuesFar1.append( np.nanstd(np.asarray(dict_out['meanValues_far_all'])[~isGroup0]))
        
        stdDiff0.append( np.nanstd(np.asarray(dict_out['diff_all'])[isGroup0]))
        stdDiff1.append( np.nanstd(np.asarray(dict_out['diff_all'])[~isGroup0]))
        stdRatio0.append( np.nanstd(np.asarray(dict_out['FaiRatios_all'])[isGroup0]))
        stdRatio1.append( np.nanstd(np.asarray(dict_out['FaiRatios_all'])[~isGroup0]))
        
        
        pvals_meanValuesCloseBetweenGroups.append( ttest(np.asarray(dict_out['meanValues_close_all'])[isGroup0],
                                               np.asarray(dict_out['meanValues_close_all'])[~isGroup0] ,
                                               nan_policy='omit').pvalue   ) 

        pvals_meanValuesFarBetweenGroups.append( ttest(np.asarray(dict_out['meanValues_far_all'])[isGroup0],
                                               np.asarray(dict_out['meanValues_far_all'])[~isGroup0] ,
                                               nan_policy='omit').pvalue   ) 

        fig, ax = plt.subplots(1,1) 
        sns.scatterplot(data=dict_out, x='meanValues_close_all', y='meanValues_far_all', hue='group_all')
        ax.set_xlabel('close')
        ax.set_ylabel('far')
        if no_legend: ax.get_legend().remove()
        fig.savefig(outputPath+'scatter_meanValues_far_close_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig)
        
        dict_out['dummy']=['' for x in dict_out['group_all']]#https://stackoverflow.com/questions/33745746/split-violinplot-in-seaborn-with-hue-parameter   
        fig, axes = plt.subplots(1,2) 
        axes[1].axis('off')
        sns.violinplot(x='dummy', y='FaiRatios_all', hue='group_all', data=dict_out, split=True,# cut=0,
                        linewidth=0, saturation=1,bw=0.15, inner=None, scale='count', ax=axes[0]) 
        sns.boxplot(x='dummy', y='FaiRatios_all', hue='group_all', data=dict_out, color='dimgray', width=0.125, 
                    linewidth=0.5, boxprops={'zorder': 2, 'alpha':0.5}, ax=axes[0]) 
        # iterate over boxes #https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn
        for b_idx, box in enumerate(axes[0].artists):
            box.set_edgecolor('black')       
            # iterate over whiskers and median lines
            for j in range(6*b_idx,6*(b_idx+1)):
                  axes[0].lines[j].set_color('black')     
        axes[0] = remove_duplicate_legend(axes[0])
        axes[0].set_xlabel("")
        axes[0].set_ylim(y_lim_ratio)
        if no_legend: axes[0].get_legend().remove()
        fig.savefig(outputPath+'Violoin_ratio_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig)
        
        fig, axes = plt.subplots(1,2) 
        axes[1].axis('off')
        sns.violinplot(x='dummy', y='diff_all', hue='group_all', data=dict_out, split=True,# cut=0,
                        linewidth=0, saturation=1,bw=0.15, inner=None, scale='count', ax=axes[0]) 
        sns.boxplot(x='dummy', y='diff_all', hue='group_all', data=dict_out, color='dimgray', width=0.125, 
                    linewidth=0.5, boxprops={'zorder': 2, 'alpha':0.5}, ax=axes[0]) 
        # iterate over boxes #https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn
        for b_idx, box in enumerate(axes[0].artists):
            box.set_edgecolor('black')       
            # iterate over whiskers and median lines
            for j in range(6*b_idx,6*(b_idx+1)):
                  axes[0].lines[j].set_color('black')     
        axes[0] = remove_duplicate_legend(axes[0])
        axes[0].set_xlabel("")
        axes[0].set_ylim(y_lim_diff)
        if no_legend: axes[0].get_legend().remove()
        fig.savefig(outputPath+'Violoin_diff_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig)
        
        fig, axes = plt.subplots(1,2) 
        axes[1].axis('off')
        sns.swarmplot(x='dummy', y='FaiRatios_all', hue='group_all', data=dict_out, dodge=True, ax=axes[0]) 
        sns.boxplot(x='dummy', y='FaiRatios_all', hue='group_all', data=dict_out, color='dimgray', width=0.125, 
                    linewidth=0.5, boxprops={'zorder': 2, 'alpha':0.5}, ax=axes[0], fliersize=0) 
        # iterate over boxes #https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn
        for b_idx, box in enumerate(axes[0].artists):
            box.set_edgecolor('black')       
            # iterate over whiskers and median lines
            for j in range(6*b_idx,6*(b_idx+1)):
                  axes[0].lines[j].set_color('black')     
        axes[0] = remove_duplicate_legend(axes[0])
        axes[0].set_xlabel("")
        axes[0].set_ylim(y_lim_ratio)
        if no_legend: axes[0].get_legend().remove()
        fig.savefig(outputPath+'Swarm_ratio_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig) 
        
        fig, axes = plt.subplots(1,2) 
        axes[1].axis('off')
        sns.swarmplot(x='dummy', y='diff_all', hue='group_all', data=dict_out, dodge=True, ax=axes[0]) 
        sns.boxplot(x='dummy', y='diff_all', hue='group_all', data=dict_out, color='dimgray', width=0.125, 
                    linewidth=0.5, boxprops={'zorder': 2, 'alpha':0.5}, ax=axes[0], fliersize=0) 
        # iterate over boxes #https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn
        for b_idx, box in enumerate(axes[0].artists):
            box.set_edgecolor('black')       
            # iterate over whiskers and median lines
            for j in range(6*b_idx,6*(b_idx+1)):
                  axes[0].lines[j].set_color('black')     
        axes[0] = remove_duplicate_legend(axes[0])
        axes[0].set_xlabel("")
        axes[0].set_ylim(y_lim_diff)
        if no_legend: axes[0].get_legend().remove()
        fig.savefig(outputPath+'Swarm_diff_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig) 
        
        fig, axes = plt.subplots(1,2) 
        axes[1].axis('off')
        sns.stripplot(x='dummy', y='diff_all', hue='group_all', data=dict_out, dodge=True, ax=axes[0])  
        axes[0].set_xlabel("")
        if no_legend: axes[0].get_legend().remove()
        fig.savefig(outputPath+'Strip_diff_plot'+valToAnalyse+'.png', dpi=300)
        plt.close(fig) 
        
    totalDiffsBetweenGroups = [np.round(x,3) for x in totalDiffsBetweenGroups]
    totalDiffFaiRatiosBetweenGroups = [np.round(x,3) for x in totalDiffFaiRatiosBetweenGroups]
    meanVolumeClose0 = [np.round(x,3) for x in meanVolumeClose0]
    meanVolumeClose1 = [np.round(x,3) for x in meanVolumeClose1]
    meanVolumeFar0 = [np.round(x,3) for x in meanVolumeFar0]
    meanVolumeFar1 = [np.round(x,3) for x in meanVolumeFar1]
    meanValuesClose0 = [np.round(x,3) for x in meanValuesClose0]
    meanValuesClose1 = [np.round(x,3) for x in meanValuesClose1]
    meanValuesFar0 = [np.round(x,3) for x in meanValuesFar0]
    meanValuesFar1 = [np.round(x,3) for x in meanValuesFar1]
    meanDiff0 = [np.round(x,3) for x in meanDiff0]
    meanDiff1 = [np.round(x,3) for x in meanDiff1]
    meanRatio0 = [np.round(x,3) for x in meanRatio0]
    meanRatio1 = [np.round(x,3) for x in meanRatio1]
    
    
    stdVolumeClose0 = [np.round(x,3) for x in stdVolumeClose0]
    stdVolumeClose1 = [np.round(x,3) for x in stdVolumeClose1]
    stdVolumeFar0 = [np.round(x,3) for x in stdVolumeFar0]
    stdVolumeFar1 = [np.round(x,3) for x in stdVolumeFar1]
    stdValuesClose0 = [np.round(x,3) for x in stdValuesClose0]
    stdValuesClose1 = [np.round(x,3) for x in stdValuesClose1]
    stdValuesFar0 = [np.round(x,3) for x in stdValuesFar0]
    stdValuesFar1 = [np.round(x,3) for x in stdValuesFar1]
    stdDiff0 = [np.round(x,3) for x in stdDiff0]
    stdDiff1 = [np.round(x,3) for x in stdDiff1]
    stdRatio0 = [np.round(x,3) for x in stdRatio0]
    stdRatio1 = [np.round(x,3) for x in stdRatio1]
    
    pvals_diffsBetweenGroups = [np.round(x,3) for x in pvals_diffsBetweenGroups]
    pvals_meanValuesCloseBetweenGroups = [np.round(x,3) for x in pvals_meanValuesCloseBetweenGroups]
    pvals_meanValuesFarBetweenGroups = [np.round(x,3) for x in pvals_meanValuesFarBetweenGroups]
    pvals_faiRatiosBetweenGroups = [np.round(x,3) for x in pvals_faiRatiosBetweenGroups]

    df_out =  pd.DataFrame({ 'foldernames' :all_foldernames,
                             'meanHU_G0_close' :meanValuesClose0,
                             'meanHU_G1_close' :meanValuesClose1,
                             'meanHU_G0_far' :meanValuesFar0,
                             'meanHU_G1_far' :meanValuesFar1,                  
                             'meanDiff_G0' :meanDiff0,
                             'meanDiff_G1' :meanDiff1, 
                             'meanRatio_G0' :meanRatio0,
                             'meanRatio_G1' :meanRatio1, 
                             
                             'meanVolume_G0_close' :meanVolumeClose0,
                             'meanVolume_G1_close' :meanVolumeClose1,
                             'meanVolume_G0_far' :meanVolumeFar0,
                             'meanVolume_G1_far' :meanVolumeFar1,

                             'stdHU_G0_close' :stdValuesClose0,
                             'stdHU_G1_close' :stdValuesClose1,
                             'stdHU_G0_far' :stdValuesFar0,
                             'stdHU_G1_far' :stdValuesFar1,
                             'stdDiff_G0' :stdDiff0,
                             'stdDiff_G1' :stdDiff1,
                             'stdRatio_G0' :stdRatio0,
                             'stdRatio_G1' :stdRatio1,
                             
                             'stdVolume_G0_close' :stdVolumeClose0,
                             'stdVolume_G1_close' :stdVolumeClose1,
                             'stdVolume_G0_far' :stdVolumeFar0,
                             'stdVolume_G1_far' :stdVolumeFar1,
                             'failed0' :numFailed0,
                             'failed1' :numFailed1,
                             'diffMeanHu' :totalDiffsBetweenGroups,
                             'diffFaiRatois' :totalDiffFaiRatiosBetweenGroups,
                             'p_meanVal_close' :pvals_meanValuesCloseBetweenGroups,
                             'p_meanVal_far' :pvals_meanValuesFarBetweenGroups,
                             'p_diffMeanHu' :pvals_diffsBetweenGroups,
                             'p_diffFaiRatio' :pvals_faiRatiosBetweenGroups,

                            })
    if len(all_ranges) == 1: excel_name = folder_str+'.xlsx'
    else: 
        excel_name = 'sorted_results.xlsx'
        with open(outputFolder_path+excel_name, 'w') as file:
            for name, diff, p in zip(all_foldernames, totalDiffsBetweenGroups, pvals_diffsBetweenGroups):
                print(f'{p} {diff} {name}', file=file)
                
    df_out.to_excel(outputFolder_path+excel_name)
    
