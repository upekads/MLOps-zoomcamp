#!/usr/bin/env python
# coding: utf-8




import pickle
import pandas as pd
import numpy as np
import sys


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model



def read_data(filename, categorical):
    
    
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df, dv,model,categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("mean predict", np.mean(y_pred))
    return y_pred

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:03d}-{month:02d}.parquet'

    categorical = ['PULocationID', 'DOLocationID']
    dv, model = load_model()
    df = read_data(filename,categorical)
    y_pred = predict(df,dv,model,categorical)

if __name__ == "__main__":
    run()








