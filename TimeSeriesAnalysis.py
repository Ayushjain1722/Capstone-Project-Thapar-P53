import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# preparing independent and dependent features
class TimeSeriesPrediction:
    def __init__(self):
        self.data = pd.read_csv(
            'https://docs.google.com/spreadsheets/d/1Sy34s0VXEBxLzoAQsCPNZR51pHKAfYwEzM-ZmGZshFY/edit#gid=797493677', error_bad_lines=False)
        # define the scope
        self.scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        # add credentials to the account
        self.creds = ServiceAccountCredentials.from_json_keyfile_name('./keys.json', self.scope)
        # authorize the clientsheet
        self.client = gspread.authorize(self.creds)
        # get the instance of the Spreadsheet
        self.sheet = self.client.open('database')
        # get the first sheet of the Spreadsheet
        self.sheet_instance = self.sheet.get_worksheet(2)
        # get all the records of the data
        self.records_data = self.sheet_instance.get_all_records()
        self.records_df = pd.DataFrame.from_dict(self.records_data)


        # choose a number of time steps
        self.n_steps = 5
        self.n_features = 1

    def prepare_data(self, timeseries_data, n_features):
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

    def getPredictions(self,numberOfPredictions, x_input, model):
        # x_input = pd.Series(records_df['Four-wheeler'].tail(4)).to_numpy()
        temp_input = list(x_input)
        lst_output = []
        i = 0
        while(i < numberOfPredictions):
            if(len(temp_input) > self.n_steps):
                x_input = np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                # print(x_input)
                x_input = x_input.reshape((1, self.n_steps, self.n_features))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.append(yhat[0][0])
                i = i+1
            else:
                x_input = x_input.reshape((1, self.n_steps, self.n_features))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i+1
        if np.ceil(lst_output)[0] < 0:
            return 0
        else:
            return np.ceil(lst_output[0])
        # return max(0,np.ceil(lst_output))

    def updateData(self, data, type_vehicle, time):
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

        update_sheet_instance = self.sheet.get_worksheet(4)
        # Update Cell
        update_sheet_instance.update_cell(cell_row, cell_column, str(data))

    def calculateAndUploadData(self, field, time):

        data = pd.Series(self.records_df[field].tail(self.n_steps)).to_numpy()
        print(data)
        dataToPrepareModel = pd.Series(self.records_df[field].tail(self.n_steps*4)).to_numpy()
        X, y = self.prepare_data(dataToPrepareModel, self.n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True,
                    input_shape=(self.n_steps, self.n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=400, verbose=1)

        calculated_data = self.getPredictions(1, data, model)

        self.updateData(calculated_data, field, time)


# calculateAndUploadData('Two-wheeler', 'Hourly')
# calculateAndUploadData('Two-wheeler', 'Daily')
# calculateAndUploadData('Two-wheeler', 'Weekly')

# calculateAndUploadData('Four-wheeler', 'Hourly')
# calculateAndUploadData('Four-wheeler', 'Daily')
# calculateAndUploadData('Four-wheeler', 'Weekly')

# calculateAndUploadData('Pedestrian', 'Hourly')
# calculateAndUploadData('Pedestrian', 'Daily')
# calculateAndUploadData('Pedestrian', 'Weekly')