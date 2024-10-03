
# general imports
import pandas as pd
import datetime
import numpy as np
import warnings
import pickle
import gurobipy as gp
import time
import os

# what we need from darts
from darts import TimeSeries
from darts.models import XGBModel
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.metrics import mape
from darts.metrics import mae
from darts.metrics import rmse

# help function for processing the data, from csv to darts.TimeSeries
def csv2TimeSeries(csvPath, timeColumn, valueColumn, freq = None):
    df = pd.read_csv(csvPath)
    # freq is optional, if not provided, it will be inferred
    ts = TimeSeries.from_dataframe(df, timeColumn, [valueColumn], freq=freq, fill_missing_dates=True)
    return ts


class EasyBldLoad:

    def __init__(self, forecastMethod = 'XGBForecast',
                 loadData = None, valueColumn = None, timeColumn = None, freq = None,
                 modelParams = None, generalSettings = None, saveSettings = None
                 ):
        """
        forecastMethod: str, the method to forecast the data
        loadData: pandas dataframe or darts TimeSeries, the latter is preferred

        valueColumn: str, the column name of the value, not required if the loadData is a darts TimeSeries
        timeColumn: str, the column name of the time, not required if the loadData is a darts TimeSeries
        freq: str, the frequency of the data, not required if the loadData is a darts TimeSeries

        modelParams: dict, the parameters for the model, refer to the used method for details
        generalSettings: dict, the general settings for the model, refer to the used method for details
        saveSettings: dict, the settings for saving the model and prediction, refer to the used method for details
        """
        
        self.forecastMethod = forecastMethod
        if isinstance(loadData, pd.DataFrame):
            assert valueColumn is not None and timeColumn is not None and freq is not None, 'The valueColumn, timeColumn and freq should be provided.'
        elif isinstance(loadData, TimeSeries):
            ...
        self.loadData = loadData
        if forecastMethod == 'XGBForecast':
            self.forecaster = XGBForecast(loadData = loadData, freq = freq, valueColumn = valueColumn, timeColumn = timeColumn, **modelParams, **generalSettings, **saveSettings)
        else:
            raise ValueError('The forecast method is not supported.')
    
    def fit(self):
        self.forecaster.fit()

    def predict(self, availableHistory, horizon):
        '''
        availableHistory: TimeSeries, the available historical data
        !!! The availableHistory should be darts.timeseries, not pandas dataframe, please use the help function to deal with it !!!
        horizon: int, the number of steps to predict
        '''
        return self.forecaster.predict(availableHistory, horizon)
        


class XGBForecast:

    def __init__(self, loadData, freq = None, valueColumn = None, timeColumn = None,
                 # model parameters
                 lags_past = None, lags_future = None, adaptiveLags=True, 
                 thresholdHours=6, output_chunk_length=60, historicalMaxWindowDays=7,
                 # other parameters for the XGBModel
                 otherParams = None,
                 # general settings
                 platformInfo='MyLaptop', muteRuntimeNotification=False,
                 # save settings
                 savePrediction=True, saveModel=False, 
                 savePath='XGBModelTraining'
                 ):
        
        """
        loadData: pandas dataframe or darts TimeSeries
        freq: str, the frequency of the data

        *** parameters for the XGBModel itself ***
        please refer to the darts.models.XGBModel for details of the parameters: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html
        lags_past: list of int, the lags of the past data
        lags_future: list of int, the lags of the future data
        thresholdHours: int, the minimum hours of data for training
        output_chunk_length: int, the length of the output chunk

        *** other model parameters ***
        adaptiveLags: bool, whether to use adaptive lags, which means the lags are generated based on the length of the data
        historicalMaxWindowDays: int, the maximum days of historical data, data beyond this window will be truncated

        platformInfo: str, the platform information
        savePrediction: bool, whether to save the prediction
        saveModel: bool, whether to save the model
        savePath: str, the path to save the model and prediction, better named according to the controller
        """

        # convert pd.dataframe to TimeSeries
        if isinstance(loadData, pd.DataFrame):
            self.loadData = TimeSeries.from_dataframe(loadData, timeColumn, [valueColumn],freq=freq,fill_missing_dates=True)
            #self.loadData = transformer.transform(loadData)
        elif isinstance(loadData, TimeSeries):
            self.loadData = loadData
            ...
        else:
            raise ValueError('The input data should be either a pandas dataframe or a darts TimeSeries.')

        # check the length of the data
        hoursData = round((self.loadData.time_index[-1]-self.loadData.time_index[0]).total_seconds()/3600)
        if hoursData/24 > historicalMaxWindowDays:
            self.loadData.drop_before(self.loadData.time_index[-1]-datetime.timedelta(days=historicalMaxWindowDays))
            if not muteRuntimeNotification:
                print('The historical data is truncated to the last {} days.'.format(historicalMaxWindowDays))

        if hoursData < thresholdHours: 
            raise ValueError('There are less than {} hours of data for training.'.format(thresholdHours))

        if lags_past is None or lags_future is None:
            # generate lags
            lags_past=[]
            lags_future=[]

            # adaptive lags for cold start when there are no enough historical data
            if adaptiveLags: 
                for ii in range(1, min(hoursData, 25)):
                    lags_past.append(-ii*60)
                    lags_future.append(ii*60)
            else:
                for ii in range(1, 25):
                    lags_past.append(-ii*60)
                    lags_future.append(ii*60)

            self.lags_past = lags_past       
            self.lags_future = lags_future
        
        else:
            self.lags_past = lags_past
            self.lags_future = lags_future

        # initialize the XGBmodel
        model = XGBModel(lags=lags_past,
                    add_encoders={
            'cyclic': {'future': ['month']},
            'datetime_attribute': {'future': ['hour', 'dayofweek']},
            'position': {'past': ['relative'], 'future': ['relative']},
            'tz': 'CET'
            },
            lags_future_covariates=lags_future,
            lags_past_covariates=lags_past,
            output_chunk_length=output_chunk_length,
            **otherParams,
            tree_method="hist", device="cuda" # enable GPU by default, if not available, it will fall back to CPU
        )

        # general settings
        self.model = model
        self.saveModel = saveModel
        self.savePath = savePath   
        self.prediction = None
        self.savePrediction = savePrediction

        # check if the savepath exists
        if self.saveModel or self.savePrediction:
            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

        # only one model will be trained of each instance, so the log is unique
        self.log = {
            'trainingStartTime': self.loadData.time_index[0],
            'trainingEndTime': self.loadData.time_index[-1],
            'fitComsumedTimeSeconds': None,
            'saveModeld': saveModel,
            'modelID': None,
            'modelPath': None,
            'modelSize': None,
            'platformInfo': platformInfo
        }


        # one model can have multiple predictions, so the prediction log is a dataframe
        # the error metrics are calculated when the prediction is made and there are available true values
        self.predictionLog = pd.DataFrame(columns=[ 'predictionID', 'predictionStartTime', 'predictionEndTime', 'savePredictiond', 'predictionPath', 'predictionConsumedTimeSeconds', 'MAPE', 'MAE', 'RMSE'])

        if not self.savePrediction:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!! The prediction will not be saved.         !!!')
            print('!!! The metrics will not be calculated either !!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def generateModelID(self):
        return 'XGBModel_{}_{}'.format(self.loadData.time_index[0].strftime('%Y%m%d%H%M'), self.loadData.time_index[-1].strftime('%Y%m%d%H%M'))

    def fit(self):

        self.log['modelID'] = self.generateModelID()
        self.log['modelPath'] = self.savePath+'/model'

        if os.path.exists(self.log['modelPath']+'/'+self.log['modelID']+'.pkl'): # load the model if it exists
            self.model = XGBModel.load(self.log['modelPath']+'/'+self.log['modelID']+'.pkl')
            print('The model is loaded from {}'.format(self.log['modelPath']+'/'+self.log['modelID']))
        else: 
            print("Training the model...")
            trainStartTime = time.time()
            self.model.fit(self.loadData)
            trainEndTime = time.time()
            self.log['fitComsumedTimeSeconds'] = trainEndTime - trainStartTime
            
            if self.saveModel:
                self.log['saveModeld'] = True
                self.log['modelID'] = self.generateModelID()
                self.log['modelPath'] = self.savePath+'/model'
                if not os.path.exists(self.log['modelPath']):
                    os.makedirs(self.log['modelPath'])
                self.model.save(self.log['modelPath']+'/'+self.log['modelID']+'.pkl')
                self.log['modelSize'] = os.path.getsize(self.log['modelPath']+'/'+self.log['modelID']+'.pkl')
                self.saveModelLog()
                #print('The model is saved at {}'.format(self.log['modelPath']+'/'+self.log['modelID']))

    def predict(self, availableHistory, horizon):

        predictStartTime = time.time()
        self.prediction = self.model.predict(n=horizon, series=availableHistory, show_warnings=False)
        predictEndTime = time.time()

        predictionConsumedTimeSeconds = predictEndTime - predictStartTime
        savePredictiond = True
        predictionID = 'XGBPrediction_{}_{}steps'.format(self.loadData.time_index[-1].strftime('%Y%m%d%H%M'), horizon)
        predictionPath = self.savePath+'/prediction'

        self.predictionLog = pd.concat(
            [self.predictionLog, pd.DataFrame(
                {'predictionID': [predictionID], 
                 'predictionStartTime': [self.prediction.time_index[0]],
                 'predictionEndTime': [self.prediction.time_index[-1]],
                 'savePredictiond': [savePredictiond], 'predictionPath': [predictionPath], 
                 'predictionConsumedTimeSeconds': [predictionConsumedTimeSeconds] })], ignore_index=True)
        
        # calculate the error metrics
        if self.savePrediction:
            for irow in range(len(self.predictionLog)):
                if self.predictionLog.loc[irow]['predictionEndTime'] <= availableHistory.time_index[-1]:
                    if pd.isna(self.predictionLog.loc[irow]['MAPE']) or pd.isna(self.predictionLog.loc[irow]['MAE'])\
                         or pd.isna(self.predictionLog.loc[irow]['RMSE']):
                        # read pickle file by default
                        previousPrediction = pickle.load(open(self.predictionLog.loc[irow]['predictionPath']+'/'+self.predictionLog.loc[irow]['predictionID']+'.pkl', 'rb')) 
                    if pd.isna(self.predictionLog.loc[irow]['MAPE']):
                        self.predictionLog.loc[irow, 'MAPE'] = mape(availableHistory, previousPrediction)
                    if pd.isna(self.predictionLog.loc[irow]['MAE']):
                        self.predictionLog.loc[irow, 'MAE'] = mae(availableHistory, previousPrediction)
                    if pd.isna(self.predictionLog.loc[irow]['RMSE']):
                        self.predictionLog.loc[irow, 'RMSE'] = rmse(availableHistory, previousPrediction)
   
        if self.savePrediction:
            if not os.path.exists(predictionPath):
                os.makedirs(predictionPath)
            self.prediction.to_pickle(predictionPath+'/'+predictionID+'.pkl')
            self.prediction.to_csv(predictionPath+'/'+predictionID+'.csv')
            #print('The prediction is saved at {} as both pickle and csv'.format(predictionPath+'/'+predictionID))

        self.savePredictionLog()
        
        return self.prediction
    
    def savePredictionLog(self, path = None):
        if path == None:
            path = self.savePath+'/predictionLog'
        if not os.path.exists(path):
            os.makedirs(path)
        self.predictionLog.to_csv(path+'/'+self.log['modelID']+'_predictionLog.csv')
        #print('The prediction log is saved at {}'.format(path))
        
    def getLog(self):
        return self.log
    
    def saveModelLog(self, path = None):
        if path == None:
            path = self.savePath+'/modelLog'
        
        if not os.path.exists(self.savePath+'/modelLog'):
            os.makedirs(self.savePath+'/modelLog')
        pd.DataFrame(self.log, index=[0]).to_csv(path+'/'+self.log['modelID']+'_modelLog.csv')
