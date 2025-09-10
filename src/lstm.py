import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.callbacks import EarlyStopping

def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map

def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]

def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]

def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output

def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output
    
def load_subjects(root='../PAMAP2_Dataset/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    
    for i in range(101,110):
        path = root + str(i) +'.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols 
        subject['id'] = i
        output = pd.concat([output,subject],ignore_index = True)
    output.reset_index(drop=True, inplace=True)
    return output

def fix_data(data):
    data = data.drop(data[data['activity_id']==0].index)
    data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    activity_mean = data.groupby(['activity_id']).mean().reset_index()
    return data

def feature_engineer(data):
    new_data = data.copy().reset_index()
    new_cols = None 
    for subject in range(101,110):
        prev_act_1 = new_data[new_data['id'] == subject]
        start = prev_act_1.head(2).index[1]
        end = prev_act_1.tail(1).index[0]
        prev_act_1 = prev_act_1.loc[start:end+1]
        new_cols_1 = pd.DataFrame()
        new_cols_1['prev_aid'] = prev_act_1['activity_id']
        new_cols_1['prev_hr'] = prev_act_1['heart_rate']
        new_cols_1['index'] = prev_act_1['index'] + 1
        if new_cols is None:
            new_cols = new_cols_1
        else:
            new_cols = pd.concat([new_cols,new_cols_1])
    new_cols = new_data.merge(new_cols, on='index', how='left')
    new_cols = new_cols.dropna()

    return new_cols

def train_test_split(new_cols):
    X_train, X_test, y_train, y_test = split_train_test(new_cols)
    print('Train shape X :',X_train.shape,' y ', y_train.shape)
    print('Test shape X :',X_test.shape,' y ', y_test.shape)
    
    X_lstm_train, y_lstm_train = create_lstm_data(X_train, y_train)
    X_lstm_test, y_lstm_test = create_lstm_data(X_test, y_test)
    hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hot.fit(y_lstm_train)
    hot.fit(y_lstm_test)
    
    y_lstm_train = hot.transform(y_lstm_train)
    y_lstm_test = hot.transform(y_lstm_test)
    print('Train shape X lstm :',X_lstm_train.shape,' y ', y_lstm_train.shape)
    print('Test shape X lstm :',X_lstm_test.shape,' y ', y_lstm_test.shape)

    return X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test

def create_model():
    lstm_model = Sequential()
    lstm_model.add(LSTM(16,input_shape=(X_lstm_train.shape[1],X_lstm_train.shape[2])))
    lstm_model.add(Dense(64 ,activation='relu'))
    lstm_model.add(Dense(64 ,activation='relu'))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(64 ,activation='relu'))
    lstm_model.add(Dense(64 ,activation='relu'))
    lstm_model.add(Dense(y_lstm_train.shape[1], activation='softmax'))
    
    lstm_model.summary()
    lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return lstm_model

def train_model(lstm_model):
    early_stopping_monitor = EarlyStopping(patience=3)
    history = lstm_model.fit(X_lstm_train, y_lstm_train, validation_split = 0.2 , epochs = 10, callbacks=[early_stopping_monitor])

def quick_plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


    
def main():
    data = fix_data(load_subjects())
    new_cols = feature_engineer(data)
    X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test = train_test_split(new_cols)
    lstm_model = create_model()
    history = train_model(lstm_mdoel)
    quick_plot_history(history)

    #accuracy test
    y = y_test[5:-1]
    preds = lstm_model.predict(X_lstm_test)
    preds_cat = np.argmax(preds,axis=1)
    # building a map of result to activity
    result = np.unique(preds_cat).tolist() 
    expected = np.unique(y).tolist() 
    combined = list(zip(result,expected))
    conf_map = dict(combined)
    # transfoms the prediction to an activity
    results = [conf_map[x] for x in preds_cat]
    print('model accuracy on test :',accuracy_score(y,results)*100)
    
if name == "__main__":
    main()