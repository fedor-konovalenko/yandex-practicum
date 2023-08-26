import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def temp_pred(path: str):
    '''running pretrained LightGBMModel'''
    try:
        data = pd.read_csv(path, index_col=[0])
    except:
        predictions = []
        status = 'Not valid data'
        return predictions, status
    if data.shape[1] != 47:
        predictions = []
        status = 'Incorrect data shape'
    else:
        scaler = StandardScaler()
        scaler.fit(data)
        features = scaler.transform(data)
        model = lgb.Booster(model_file='best_lgb.txt')
        predictions = model.predict(features)
        status = 'Succesfully predicted'
        res_data = {'key number': data.index, 'temperature': predictions}
        result = pd.DataFrame(data=res_data)
    return result, status



