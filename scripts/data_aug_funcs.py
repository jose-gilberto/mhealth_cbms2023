# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:04:06 2023

@author: lucas
"""
import numpy as np
import random
import tsaug


def windowing(time_series,parameters,number_of_parts):
    if number_of_parts == 1:
        return time_series, parameters
    
    time_series_with_windowing = np.zeros(shape = (number_of_parts*time_series.shape[0],
                                               300 ,time_series.shape[2]))
    parameters_with_windowing = []
    for i in range(time_series.shape[0]):
        for j in range(number_of_parts):
            start = random.randint(0, time_series.shape[1] - 300)
            time_series_with_windowing[i] = time_series[i][start:start+300]
            parameters_with_windowing.append(parameters[i])
            
    return time_series_with_windowing,np.array(parameters_with_windowing)


def time_series_augmentations(time_series,parameters, multiple, augmenters):
    if(multiple == 0):
        return time_series,parameters
    
    sum_augmenters = augmenters[0]*multiple
    
    for i in range(1,len(augmenters)):
        sum_augmenters += augmenters[i]
    
    augmenter = (
        sum_augmenters
    )
    augmented_time_series = augmenter.augment(time_series)
    augmented_time_series = np.concatenate((time_series,augmented_time_series))
    augmented_parameters = np.repeat(parameters, multiple)
    augmented_parameters = np.concatenate((parameters,augmented_parameters))

    return augmented_time_series, augmented_parameters