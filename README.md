
# Hemoglobin Estimation from Smartphone-Based Photoplethysmography with Small Data

Repository containing all the code for the article published on the CBMS 2023.

## Authors:

Diego Furtado Silva (email: diegofsilva@usp.br)  
José Gilberto Barbosa de Medeiros Júnior (email: gilberto.barbosa@usp.br)  
Lucas V. Domingues (email: lucasvd@estudante.ufscar.br)  
Thiago Mazzu-Nascimento (email: thiagomazzu@estudante.ufscar.br)

## Abstract
Photoplethysmography (PPG) is a well-known technique to estimate blood pressure, oxygen saturation, and heart frequency. Recent efforts aim to obtain PPG from wearable and mobile devices, allowing more democratic access. This paper explores the potential of using a smartphone camera as a PPG sensor, getting a time series based on the RGB values of video recordings of patients’ fingertips. Through this PPG, we apply machine learning for the non-invasive estimation of hemoglobin levels. We assume a realistic scenario where the data has a low volume and potentially a low quality. The generalization capacity of the models built on these scenarios usually achieves undesirable performance. This paper presents a novel dataset that comprises real-world mobile phone-based PPG and a comprehensive experiment on how different techniques may improve hemoglobin estimation using deep neural architectures. In general, cleaning, augmentation, and ensemble positively affect the results. In some cases, these techniques reduced the mean absolute error by more than thirty percent.

## Structure

```
|-data/
    |- M1/
    |- ...
|-notebooks/
|-scripts/
|-results/
```

- `data` contains all the data instances used, each folder in this directory represents a different pacient, the .csv file contains all hemoglobin rates;
- `notebooks` contains all the jupyter notebooks used to produce the results and experiments;
- `scripts` contains all the python scripts used in the experiments;
- `results` contains all the csv files with the experiments results.

## Training models

To train the deep learning models use the training_models notebook. Remember to check the paths, devices and other environment configs in the files. Any question just email me on: gilberto.barbosa@usp.br