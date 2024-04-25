clc
clearvars
close all
warning('off','all')

path = pwd;

mod = 'QPSK';
ChType = 'VehicularA';
v = 1000;                    % Moving speed of user in km/h
IBO = 2;

nSym                    = 14;        % Number of symbols within one frame
N_CH                    = [2000; 2000; 2000; 2000; 2000; 2000; 2000; 2000; 10000]; % Number of channel realizations
EbN0dB                  = 0:5:40;    % bit to noise ratio

pathdata = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];

% Generating testing and training indices
% idx = randperm(N_CH);
% training_indices          = idx(1:floor(0.8*N_CH));               % number of channel realizations for training
% testing_indices           = idx(length(training_indices)+1:end);  % number of channel realizations for testing
% save('indices','testing_indices','training_indices')
load('indices');
Testing_Data_set_size     = 2000;
Training_Data_set_size    = 8000;

% Define Simulation parameters
SNR                       = 0:5:40;
nUSC                      = 52;
preamble_size             = 1;

algo                    = 'LS_Interpolation';
wave                    = 'OTFS';

ppositions                = [3,7,14,21,32,39,46,50];    % Pilots positions in Kset

% X_sub will define the quantity of active subcarriers to be used,
% maintaining all the pilots

rate = 2;
X_Sub_aux = [1:rate:52 ppositions];
X_Sub_aux = sort(X_Sub_aux, 'ascend');
X_Sub = unique(X_Sub_aux);

nSym_total = (nSym+preamble_size);

Train_X                   = zeros(length(X_Sub)*2, Training_Data_set_size * nSym_total);
Train_Y                   = zeros(nUSC*2, Training_Data_set_size * nSym_total);
Test_X                    = zeros(length(X_Sub)*2, Testing_Data_set_size * nSym_total);
Test_Y                    = zeros(nUSC*2, Testing_Data_set_size * nSym_total);

% testing datasets
for n_snr = 1:size(SNR,2)-1

    % Load simulation data according to the defined configurations (Ch, mod, algorithm) 
    load(['data_' pathdata '\Simulation_' num2str(n_snr),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
    Algo_Channels_Structure = eval([algo '_Structure']);
    
    Testing_DatasetX  =  Algo_Channels_Structure(X_Sub,:,:);
    Testing_DatasetY  =  True_Channels_Structure;

    Testing_DatasetX_1nch  =  Testing_DatasetX(:,:,1);
   
    % Expend Testing and Training Datasets
    Testing_DatasetX_expended  = reshape(Testing_DatasetX, length(X_Sub), nSym_total * Testing_Data_set_size);
    Testing_DatasetY_expended  = reshape(Testing_DatasetY, nUSC, nSym_total * Testing_Data_set_size);

    Test_X(1:length(X_Sub),:)                       = real(Testing_DatasetX_expended);
    Test_X(length(X_Sub)+1:2*length(X_Sub),:)       = imag(Testing_DatasetX_expended);
    Test_Y(1:nUSC,:)                                = real(Testing_DatasetY_expended);
    Test_Y(nUSC+1:2*nUSC,:)                         = imag(Testing_DatasetY_expended);
    
    % Save training and testing datasets to the DNN_Datasets structure
    DNN_Datasets.('Test_X')  =  Test_X;
    DNN_Datasets.('Test_Y')  =  Test_Y;

    % Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
    save([path '\Python_Codes - NLD\data\' algo '_' wave '_Less_' num2str(rate) '_Dataset_' num2str(n_snr)],  'DNN_Datasets');
    disp(['Data generated for ' algo ', SNR = ', num2str(SNR(n_snr))]);

end

% training dataset
max_index = find(SNR == max(SNR));
% Load simulation data according to the defined configurations (Ch, mod, algorithm) 
load(['data_' pathdata '\Simulation_' num2str(max_index),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
Algo_Channels_Structure = eval([algo '_Structure']);

Training_DatasetX =  Algo_Channels_Structure(X_Sub,:,training_indices);
Training_DatasetY =  True_Channels_Structure(:,:,training_indices);
Testing_DatasetX  =  Algo_Channels_Structure(X_Sub,:,testing_indices);
Testing_DatasetY  =  True_Channels_Structure(:,:,testing_indices);
   
% Expend Testing and Training Datasets
Training_DatasetX_expended = reshape(Training_DatasetX, length(X_Sub), nSym_total * Training_Data_set_size);
Training_DatasetY_expended = reshape(Training_DatasetY, nUSC, nSym_total * Training_Data_set_size);
Testing_DatasetX_expended  = reshape(Testing_DatasetX, length(X_Sub), nSym_total * Testing_Data_set_size);
Testing_DatasetY_expended  = reshape(Testing_DatasetY, nUSC, nSym_total * Testing_Data_set_size);

% Complex to Real domain conversion
Train_X(1:length(X_Sub),:)                    = real(Training_DatasetX_expended);
Train_X(length(X_Sub)+1:2*length(X_Sub),:)    = imag(Training_DatasetX_expended);
Train_Y(1:nUSC,:)                             = real(Training_DatasetY_expended);
Train_Y(nUSC+1:2*nUSC,:)                      = imag(Training_DatasetY_expended);

Test_X(1:length(X_Sub),:)                       = real(Testing_DatasetX_expended);
Test_X(length(X_Sub)+1:2*length(X_Sub),:)       = imag(Testing_DatasetX_expended);
Test_Y(1:nUSC,:)                                = real(Testing_DatasetY_expended);
Test_Y(nUSC+1:2*nUSC,:)                         = imag(Testing_DatasetY_expended);

% Save training and testing datasets to the DNN_Datasets structure
DNN_Datasets.('Train_X') =  Train_X;
DNN_Datasets.('Train_Y') =  Train_Y;
DNN_Datasets.('Test_X')  =  Test_X;
DNN_Datasets.('Test_Y')  =  Test_Y;

% Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
save([path '\Python_Codes - NLD\data\' algo '_' wave '_Less_' num2str(rate) '_Dataset_' num2str(max_index)],  'DNN_Datasets');
disp(['Data generated for ' algo ', SNR = ', num2str(max_index)]);
