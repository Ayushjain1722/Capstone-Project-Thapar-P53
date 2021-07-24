import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# preparing independent and dependent features


def prepare_data(timeseries_data, n_features):
    X, y = [], []
    for i in range(len(timeseries_data)):
        # find the end of this pattern
        end_ix = i + n_features
        # check if we are beyond the sequence
        if end_ix > len(timeseries_data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data = pd.read_csv(
    'https://docs.google.com/spreadsheets/d/1Sy34s0VXEBxLzoAQsCPNZR51pHKAfYwEzM-ZmGZshFY/edit#gid=797493677', error_bad_lines=False)
# define the scope
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('./keys.json', scope)
# authorize the clientsheet
client = gspread.authorize(creds)
# get the instance of the Spreadsheet
sheet = client.open('database')
# get the first sheet of the Spreadsheet
sheet_instance = sheet.get_worksheet(2)
# get all the records of the data
records_data = sheet_instance.get_all_records()
records_df = pd.DataFrame.from_dict(records_data)


# choose a number of time steps
n_steps = 5
n_features = 1


def getPredictions(numberOfPredictions, x_input, model):
    # x_input = pd.Series(records_df['Four-wheeler'].tail(4)).to_numpy()
    temp_input = list(x_input)
    lst_output = []
    i = 0
    while(i < numberOfPredictions):
        if(len(temp_input) > n_steps):
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            # print(x_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.append(yhat[0][0])
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i = i+1

    return np.ceil(lst_output)


def updateData(data, type_vehicle, time):
    cell_row = 2
    cell_column = 1

    if type_vehicle == 'Two-wheeler' and time == 'Daily':
        cell_column = 1
    if type_vehicle == 'Two-wheeler' and time == 'Weekly':
        cell_column = 4
    if type_vehicle == 'Two-wheeler' and time == 'Monthly':
        cell_column = 7
    if type_vehicle == 'Four-wheeler' and time == 'Daily':
        cell_column = 2
    if type_vehicle == 'Four-wheeler' and time == 'Weekly':
        cell_column = 5
    if type_vehicle == 'Four-wheeler' and time == 'Monthly':
        cell_column = 8
    if type_vehicle == 'Pedestrian' and time == 'Daily':
        cell_column = 3
    if type_vehicle == 'Pedestrian' and time == 'Weekly':
        cell_column = 6
    if type_vehicle == 'Pedestrian' and time == 'Monthly':
        cell_column = 9

    update_sheet_instance = sheet.get_worksheet(4)
    # Update Cell
    update_sheet_instance.update_cell(cell_row, cell_column, str(data[0]))


def calculateAndUploadData(field, time):

    data = pd.Series(records_df[field].tail(n_steps)).to_numpy()
    print(data)
    dataToPrepareModel = pd.Series(
        records_df[field].tail(n_steps*4)).to_numpy()
    X, y = prepare_data(dataToPrepareModel, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True,
                   input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=400, verbose=1)

    calculated_data = getPredictions(1, data, model)

    updateData(calculated_data, field, time)


calculateAndUploadData('Two-wheeler', 'Hourly')
calculateAndUploadData('Two-wheeler', 'Daily')
calculateAndUploadData('Two-wheeler', 'Weekly')

calculateAndUploadData('Four-wheeler', 'Hourly')
calculateAndUploadData('Four-wheeler', 'Daily')
calculateAndUploadData('Four-wheeler', 'Weekly')

calculateAndUploadData('Pedestrian', 'Hourly')
calculateAndUploadData('Pedestrian', 'Daily')
calculateAndUploadData('Pedestrian', 'Weekly')

