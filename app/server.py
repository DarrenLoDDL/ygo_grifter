from fastapi import FastAPI
#import datasets
import json
import pandas as pd
import torch
import torch.nn as nn
from pandas import read_json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, BertConfig
from torch.utils.data import DataLoader
from datasets import Dataset
import io
from io import StringIO
import json
import re
import os
import requests
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from transformers import pipeline
#import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#import mpl_scatter_density
#from matplotlib.colors import LinearSegmentedColormap
#from matplotlib import cm
#import math 

model = AutoModelForSequenceClassification.from_pretrained("app/ygo_nlp_Version2_RMSLE",num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#input = tokenizer(test_ds["text"][i], return_tensors="pt")
#output = model(**input).logits

app = FastAPI()

def list_unique(samplelist, category):
    output = {}
    hidden_out = []
    for element in samplelist:
        if str(element) != 'nan':
            if str(element) not in hidden_out:
                open_q = '"'
                close_q = '"'
                stringed = open_q + str(element) + close_q ##small adjustment on thel isted elements
                output[stringed] =  category
                output[str(element)] = category
                hidden_out.append(str(element))
    return output


def preprocess(json_file): #takes an api query as string
    res = requests.get(json_file + "&tcgplayer_data=true")
    data = json.loads(res.text)
    print(data)
    data = json.dumps(data)
    data = data[8:-1]
    #print(type(data)
    print(type(data))
    df = pd.read_json(data)
    print(type(df))
    
    sample = df.explode('card_sets').reset_index(drop=True)
    del sample["card_prices"]
    del sample["card_images"]
    secondsample = pd.json_normalize(json.loads(sample.to_json(orient = 'records')), max_level = 1)
    print(sample.columns.values)
    sample = secondsample

    #handling self referencing
    try:
        atlist = sample['archetype']
    except:
        atlist = [" "]
    namelist = sample['name']
    print(len(namelist))
    unq_atlist = list_unique(atlist, "archetypal")
    unq_namelist = list_unique(namelist, "card with this name")

    for name in unq_namelist:
        unq_atlist[name] = unq_namelist[name]

    #replacing archetypes and names in desc
    for i in range(len(sample)):
        sections = sample['desc'][i].split('"')
        #print(sections)
        for j in range(len(sections)):
            if sections[j] in unq_atlist:
                sections[j] = unq_atlist[sections[j]]
            else:
                if sections[j][0] != ' ' and sections[j][-1]  != ' ':
                    term = sections[j].split(" ")
                    checker = False
                    for k in range(len(term)):
                        if term[k] in unq_atlist:
                            checker = True
                    if checker == True:
                        sections[j] = "archetype card with a name"
                    else:
                        sections[j] = "outer-archetype card"
                    
        sample['desc'][i] = ''.join([str(chunk) for chunk in sections])
        print('\r',"rows preprocessed: "+ str(i) + "/" + str(len(sample)), end="")
        
    print()
    print("preprocessing done")
    return sample 

def concat(sample):
    sample = sample
    text_and_target = []
    headers = []
    reject_list = ["card_sets", 'name', 'set_name', 'card_sets.set_name','card_sets.set_url','card_sets.set_code', 'set_rarity', 'id', 'pend_desc', 'monster_desc', 
                   'set price', 'archetype', 'linkval', 'ygoprodeck_url', 'card_sets.set_url']
    
    sample = sample.drop(sample[sample['frameType'] == 'token'].index)
    sample = sample.drop(sample[sample['frameType'] == 'skill'].index)
    sample = sample.drop(sample[sample['frameType'] == 'normal'].index)
    #sample = sample.drop(sample[sample['card_sets.set_price'].astype(float) > 100].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Quarter Century Secret Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == "Collector's Rare"].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Ultimate Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Ghost Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Platinum Secret Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == "Prismatic Collector's Rare"].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Prismatic Ultimate Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Mosaic Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Prismatic Ultimate Rare'].index)
    #sample = sample.drop(sample[sample['card_sets.set_rarity'] == 'Starlight Rare'].index)
    #ssample = sample[sample['card_sets.set_rarity'] == "Common"]
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Super Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Ultra Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Secret Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Gold Rare"]) 
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Premium Gold Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Gold Secret Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Duel Terminal Normal Parallel Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Duel Terminal Rare Parallel Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Duel Terminal Super Parallel Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Duel Terminal Ultra Parallel Rare"])
    #ssample._append(sample[sample['card_sets.set_rarity'] == "Starfoil Rare"])
    
    #sample = ssample
    print(len(sample))
    meta_data = []
    for i in range (len(sample)):
        meta_builder = []
        text = ""
        target = 0
        for j in range(len(sample.columns.values)):
            if sample.columns.values[j] == "card_sets.set_price":
                stripped = str(sample.iloc[i][j]).replace(",", "") #some cards are 4 digit prices
                target = float("{:.2f}".format(float(stripped)))
                if target == 0:
                    target = float("{:.2f}".format(0.20))
            if str(sample.columns.values[j]) == "name":
                meta_builder.append(str(sample.iloc[i][j]))
            if str(sample.columns.values[j]) == "card_sets.set_name":
                meta_builder.append(str(sample.iloc[i][j]))
            if str(sample.columns.values[j]) == "card_sets.set_rarity":
                meta_builder.append(str(sample.iloc[i][j]))
            if str(sample.columns.values[j]) not in reject_list and sample.columns.values[j] != "card_sets.set_price":
                text = text  + "[SEP]" + "AAA"+ sample.columns.values[j] + " " + str(sample.iloc[i][j]) + " "
            if "AAA" + sample.columns.values[j] not in headers:
                heading = "AAA" + str(sample.columns.values[j])
                headers.append(heading)
        text_and_target.append([text[:-1],target])
        meta_data.append(meta_builder)
        print('\r',"rows done: " + str(len(text_and_target)) + "/" + str(len(sample)), end ="")
    for i in range(len(headers)):
        headers[i] = headers[i].replace("_","").replace(".","")
    print()
    print("concatenation complete")

    complete_dataset = pd.DataFrame(text_and_target, columns = ['text','label'])
    complete_dataset.dropna(subset=['label'], inplace=True)

    print("Finished")

    return complete_dataset, headers, meta_data

def RegressionTest(title, model,test_ds, meta_data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pred = []
    true = []
    comparison = []
    max_log = 0 
    for i in range(len(test_ds)):   
        input = tokenizer(test_ds["text"][i], return_tensors="pt")
        output = model(**input).logits
        comparison.append([float(output[0,0].item())*100, float(test_ds["label"][i])])
        pred.append(float(output[0,0].item())*100)
        true.append(float(test_ds["label"][i])*100)
        print('\r',"tests done: "+ str(i) + "/" + str(len(test_ds)), end="")
        comp_table = []
    for i in range(len(comparison)):
        adder = []
        adder.append(comparison[i])
        adder.append(meta_data[i])
        comp_table.append(adder)
    return compute_metrics_for_regression(true, pred), comp_table

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,logits,labels):
        return torch.sqrt(torch.nn.functional.mse_loss(logits, labels))

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def compute_metrics_for_regression(labels, logits):
    #logits, labels, dummy = eval_pred #had a gander at the output of eval pred (returned an object so i just googled what it returned), it returned three things, just gotta put one of them into a dummy var
    #labels = labels.reshape(-1, 1)
    rmse_metric = RMSELoss()
    mse = mean_squared_error(labels, logits)
    rmse = (mse)**(1/2)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    mape = mean_absolute_percentage_error(labels, logits)

    #single_squared_errors = ((logits - labels).flatten()**2).tolist()
    #smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)
    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < (whatever e is less than
    #accuracy = sum([1 for e in single_squared_errors if e < 0.15]) / len(single_squared_errors)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

@app.get('/')
def reed_root():
    return{'message': 'YGO Grifter API'}

@app.post('/predict for name')
def predict(data: str):
    link = "https://db.ygoprodeck.com/api/v7/cardinfo.php?name=" + data
    test_prep = preprocess(link)
    test, test_headers, meta_data = concat(test_prep)
    test.to_json('input.json', orient = "records")
    loader = pd.read_json('input.json')
    test_ds = Dataset.from_pandas(loader)
    return RegressionTest('BERT-WS Scatterplot', model, test_ds, meta_data)
    
