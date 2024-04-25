import pickle
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import scipy.io
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.backend import squeeze

DNN_Model = 40
scheme = 'LS_Interpolation' 
mod    = 'OTFS' 
preamble_size = 1
no_midamble = 0
nSym = 14
nSym_total = nSym + preamble_size + no_midamble
nDSC = 44
pilots = 8
#nDSC = 48
#pilots = 4
rate = 2
In = 2*(nDSC/int(rate) + pilots)
LSTM_size = (nDSC/int(rate) + pilots)
Out =  2*(nDSC + pilots)
MLP_size = 15

SNR_index = np.arange(1, 10)

for j in SNR_index:
    mat = loadmat('data\{}_{}_Less_{}_Dataset_{}.mat'.format(scheme,mod,rate, j))
    Testing_Dataset = mat['DNN_Datasets']
    Testing_Dataset = Testing_Dataset[0, 0]
    X = Testing_Dataset['Test_X']
    Y = Testing_Dataset['Test_Y']
    print('Loaded Dataset Inputs: ', X.shape)
    print('Loaded Dataset Outputs: ', Y.shape)
    # Normalizing Datasets
    scalerx = StandardScaler()
    scalerx.fit(X)
    scalery = StandardScaler()
    scalery.fit(Y)
    XS = scalerx.transform(X)
    YS = scalery.transform(Y)
    XS = XS.transpose()
    YS = YS.transpose()

    # To use LSTM networks, the input needs to be reshaped to be [samples, time steps, features]
    XS = np.reshape(XS,(2000, nSym_total, int(In)))
    print(XS.shape)

    # Loading trained DNN
    model = load_model('data\{}_Less_DNN_{}_{}{}_{}.h5'.format(scheme, mod, int(LSTM_size), MLP_size, DNN_Model))
    print('Model Loaded: ', DNN_Model)

    # Testing the model
    Y_pred = model.predict(XS)
   
    XS = np.reshape(XS,(2000*nSym_total, int(In)))
    Y_pred = np.reshape(Y_pred,(2000*nSym_total, int(Out)))

    XS = XS.transpose()
    YS = YS.transpose()
    Y_pred = Y_pred.transpose()

    # Calculation of Mean Squared Error (MSE)

    Original_Testing_X = scalerx.inverse_transform(XS)
    Original_Testing_Y = scalery.inverse_transform(YS)
    Prediction_Y = scalery.inverse_transform(Y_pred)

    Error = mean_squared_error(Original_Testing_Y, Prediction_Y)
    print('MSE: ', Error)
   
    # Saving the results and converting to .mat
    result_path = 'data\{}_LSTM_Less_DNN_{}_{}{}_Results_{}.pickle'.format(scheme, mod, int(LSTM_size), MLP_size, j)
    with open(result_path, 'wb') as f:
        pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

    dest_name = 'data\{}_LSTM_Less_DNN_{}_{}{}_Results_{}.mat'.format(scheme, mod,  int(LSTM_size), MLP_size, j)
    a = pickle.load(open(result_path, "rb"))
    scipy.io.savemat(dest_name, {
        '{}_LSTM_Less_DNN_{}_{}{}_test_x_{}'.format(scheme,mod, int(LSTM_size), MLP_size, j): a[0],
        '{}_LSTM_Less_DNN_{}_{}{}_test_y_{}'.format(scheme,mod,  int(LSTM_size), MLP_size, j): a[1],
        '{}_LSTM_Less_DNN_{}_{}{}_corrected_y_{}'.format(scheme,mod,  int(LSTM_size), MLP_size, j): a[2]
    })
    
    print("Data successfully converted to .mat file ")
