# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:06:57 2023

@author: lucas
"""

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler


def filter_videos(file_name):
    if file_name[-3:] == 'mp4':
        return True
    return False


def extract_time_series(file):
    bluetotal = np.array([[]])
    redtotal = np.array([[]])
    greentotal = np.array([[]])
    
    cap = cv2.VideoCapture(file)
    (grabbed, frame) = cap.read()
    if not grabbed:
        raise Exception("File not found")
        return

    while True:
        (b, g, r) = cv2.split(frame)
        bluetotal = np.append(bluetotal, b.sum())
        redtotal = np.append(redtotal, r.sum())
        greentotal = np.append(greentotal,g.sum())

        (grabbed, frame) = cap.read()
        
        if not grabbed:
            break
        
    cap.release()
    cv2.destroyAllWindows()

    bluetotal = np.reshape(bluetotal,[bluetotal.shape[0],1])
    redtotal = np.reshape(redtotal,[redtotal.shape[0],1])
    greentotal = np.reshape(greentotal,[greentotal.shape[0],1])
    
    normalizador = MinMaxScaler(feature_range = (0,1))
    bluetotalnorm = normalizador.fit_transform(bluetotal)
    greentotalnorm = normalizador.fit_transform(greentotal)
    redtotalnorm=normalizador.fit_transform(redtotal)

    return redtotalnorm, greentotalnorm, bluetotalnorm


def create_time_series_dataset(
    input_path,
    output_path,
    directories_ids = {'HU Enfermaria':'HUE','Maricondi':'M', 'Santa Casa Enfermaria':'SCE', 'Santa Casa PA':'SCPA'},
    fingers = 'both'
):
    
    if fingers not in ['right','left','both']:
        raise Exception('fingers argument can either be right, left or both')
        
    for directory in directories_ids.keys():
        
        directory_path = input_path+'/'+directory
        patinets_folders = sorted(os.listdir(directory_path),key=int)
        
        for folder in patinets_folders:
            folder_path = directory_path+'/'+folder
            recordings = list(filter(filter_videos,os.listdir(folder_path)))
        
            if fingers == 'right':
                select_fingers = ['dedo_direito']
            elif fingers == 'left':
                select_fingers = ['dedo_esquerdo']
            else:
                select_fingers = ['dedo_direito','dedo_esquerdo']
            
            filterd_fingers = list(filter(lambda file_name: file_name[:-4] in select_fingers,recordings))
            
            fingers_abbreviations= {'dedo_direito.mp4':'dd', 'dedo_esquerdo.mp4':'de'}
            
            path_to_time_series = output_path+'/'+directories_ids[directory]+str(folder)
            
            if not os.path.exists(path_to_time_series):
                os.makedirs(path_to_time_series)
            for finger_recording in filterd_fingers:
                r_series,b_series,g_series = extract_time_series(folder_path +'/'+finger_recording)
                
                
                r_series = np.reshape(r_series,(r_series.shape[0]*r_series.shape[1]))
                g_series = np.reshape(g_series,(g_series.shape[0]*g_series.shape[1]))
                b_series = np.reshape(b_series,(b_series.shape[0]*b_series.shape[1]))
                
                time_series_dict = {'R':r_series,'B':b_series,'G':g_series}
                
                time_series_df = pd.DataFrame.from_dict(time_series_dict)

                
                time_series_df.to_csv(path_to_time_series+'/'+
                                     fingers_abbreviations[finger_recording]+
                                      '_rgb_time_series.csv',columns =['R','G','B'],index=False)
   

def crop_time_series(dataset_dict,size = 900):
    for series in list(dataset_dict):
        if dataset_dict[series].shape[0] < size:
            dataset_dict.pop(series)
        else:
            dataset_dict[series] = dataset_dict[series][0:size]
            
    return dataset_dict


def read_time_series(time_series_path, patient_ids, fingers='both'):    
    if fingers not in ['right','left','both']:
        raise Exception('fingers argument can either be right, left or both')
    elif fingers == 'right':
        select_fingers = ['dd']
    elif fingers == 'left':
        select_fingers = ['de']
    else:
        select_fingers = ['dd','de']
        
    
    time_series_set = []
    indexes = []

    for patient_id in patient_ids:
        for finger in select_fingers:
            rgb_time_series = pd.read_csv(time_series_path+'/'+patient_id+
                                          '/'+finger+'_rgb_time_series.csv')

            if rgb_time_series.shape[0] >= 900:
                indexes.append(patient_id)
                time_series_set.append(rgb_time_series.to_numpy()[0:900])
                
    return np.array(indexes), np.array(time_series_set)
            
        
def read_parameters(parameters_file):
    parameter = pd.read_csv(parameters_file)
    return parameter

        
def create_dataset(time_series_path, parameters_file):
    parameters = read_parameters(parameters_file).dropna()
    idxs, time_series = read_time_series(time_series_path, parameters['Paciente'])
    y=[]
    for i in range(len(time_series)):
        y.append(parameters[parameters['Paciente']==idxs[i]][parameters.columns[-1]].iloc[0])

    return time_series,np.array(y)

    
def normalize_parameters(parameters):
    norm = MinMaxScaler(feature_range = (0,1))
    normalized_param = np.reshape(parameters,(parameters.shape[0],1))    
    normalized_param = norm.fit_transform(normalized_param)
    
    return normalized_param, norm
