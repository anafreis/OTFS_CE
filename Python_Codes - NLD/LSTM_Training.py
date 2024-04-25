from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from numpy.random import seed
import numpy as np
import warnings
warnings.filterwarnings("ignore")
SNR_array = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

scheme = 'LS_Interpolation' 
mod    = 'OTFS' 
preamble_size = 1
no_midamble = 0
nSym = 14
nSym_total = nSym + preamble_size + no_midamble
snr = 9
nDSC = 44
pilots = 8
#nDSC = 48
#pilots = 4
rate = 2
In = 2*(nDSC/int(rate) + pilots)
LSTM_size = (nDSC/int(rate) + pilots)
MLP_size = 15
Out =  2*(nDSC + pilots)
epoch = 500
batch_size = 128

# Training with the highest SNR
mat = loadmat('data\{}_{}_Less_{}_Dataset_{}.mat'.format(scheme,mod,rate, snr))
Training_Dataset = mat['DNN_Datasets']
Training_Dataset = Training_Dataset[0, 0]
X = Training_Dataset['Train_X']
Y = Training_Dataset['Train_Y']
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
XS     = np.reshape(XS,(8000, nSym_total, int(In)))
YS     = np.reshape(YS,(8000, nSym_total,  int(Out)))
print('Training shape', XS.shape)

X_train, X_test, y_train, y_test = train_test_split(XS, YS, test_size=0.10)

# Build the model.
# The weighs are initialized by the Glorot Initializer
      #http://proceedings.mlr.press/v9/glorot10a.html
init = glorot_uniform(seed=1)
# Add a LSTM layer with 128 internal units.
model = Sequential([LSTM(units= int(LSTM_size), activation='relu',
      kernel_initializer=init,
      bias_initializer=init,return_sequences=True),
      Dense(units= int(MLP_size), activation='relu',
      kernel_initializer=init,
      bias_initializer=init),
      Dense(units=int(Out), kernel_initializer=init,
      bias_initializer=init)
      ])

# Compile the model.
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])

# Train the models
for i in range(0,epoch):
      seed(1)
      model_path = 'data\models\{}_Less_{}_DNN_{}_{}{}_{}_{}.h5'.format(scheme,rate,mod, int(LSTM_size),MLP_size, SNR_array[int(snr)-1],i)
      checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                             verbose=1, save_best_only=True,
                             mode='max')
      callbacks_list = [checkpoint]
      model.fit(X_train, y_train, epochs=1, batch_size=int(batch_size), verbose=2, validation_data=(X_test, y_test),  callbacks=callbacks_list)
   