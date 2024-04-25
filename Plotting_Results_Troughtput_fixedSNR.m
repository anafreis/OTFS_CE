clc
clear all
% close all
addpath('OTFS_functions');

N = [14 30 50];
nFFT                   = 64;             % FFT size 
nDSC                   = 44;             % Number of data subcarriers
preamble_size = 1;                       % preamble considered in the proposed scheme

v = 500; % Moving speed of user in km/h
mod = '16QAM';
IBO = 4;
EbN0dB_all                    = 0:5:40;
EbN0dB                        = 30;
i                             = find(EbN0dB_all==EbN0dB);

coderate = 1/2;
if isequal(mod,'16QAM')
    MCS_rate = log2(16)*coderate;
elseif isequal(mod,'QPSK')
    MCS_rate = log2(4)*coderate;
end

%% Physical Layer Specifications 
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)

% 14 symbols
nSym = N(1);
pathdata_NLD = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];
load(['data_' pathdata_NLD '/Results_OTFS_NLD'],'BER_Raviteja_Est_NLD');
load(['data_' pathdata_NLD '/LSTM_NN_Results_Less_OTFS_3015'])

n_Raviteja = (nSym*nDSC)/((nSym)*nFFT);
n_proposal = (nSym*nDSC)/((nSym+preamble_size)*nFFT);

trRaviteja_14Sym = n_Raviteja * MCS_rate * (1 - BER_Raviteja_Est_NLD(i));
trProposal_14Sym = n_proposal  * MCS_rate * (1 - BER_LSTM_NN_TF(i));
clearvars nSym pathdata_NLD BER_Raviteja_Est_NLD rate_Raviteja BER_LSTM_NN_TF rate_proposal

% 30 symbols
nSym = N(2);
pathdata_NLD = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];
load(['data_' pathdata_NLD '/Results_OTFS_NLD'],'BER_Raviteja_Est_NLD');
load(['data_' pathdata_NLD '/LSTM_NN_Results_Less_OTFS_3015'])

n_Raviteja = (nSym*nDSC)/((nSym)*nFFT);
n_proposal = (nSym*nDSC)/((nSym+preamble_size)*nFFT);

trRaviteja_30Sym = n_Raviteja * MCS_rate  * (1 - BER_Raviteja_Est_NLD(i)); 
trProposal_30Sym = n_proposal  * MCS_rate * (1 - BER_LSTM_NN_TF(i));

clearvars nSym BER_Raviteja_Est_NLD rate_Raviteja BER_LSTM_NN_TF rate_proposal

% 50 symbols
nSym = N(3);
pathdata_NLD = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];
load(['data_' pathdata_NLD '/Results_OTFS_NLD'],'BER_Raviteja_Est_NLD');
load(['data_' pathdata_NLD '/LSTM_NN_Results_Less_OTFS_3015'])

n_Raviteja = (nSym*nDSC)/((nSym)*nFFT);
n_proposal = (nSym*nDSC)/((nSym+preamble_size)*nFFT);

trRaviteja_50Sym = n_Raviteja  * MCS_rate  * (1 - BER_Raviteja_Est_NLD(i));
trProposal_50Sym = n_proposal  * MCS_rate * (1 - BER_LSTM_NN_TF(i));

figure
colorOrder = get(gca, 'ColorOrder');

results = [trRaviteja_14Sym trProposal_14Sym;
           trRaviteja_30Sym trProposal_30Sym;
           trRaviteja_50Sym trProposal_50Sym];

bar(results);
colororder(colorOrder(2:4,:))
grid on
name = {'14';'30'; '50'};
set(gca,'xticklabel',name)
legend('TCE [6]','LS-LSTM-NN','Location','best','Interpreter','latex');
xlabel('$N$');
ylabel('Throughput (bps/Hz)');
ylim([0 2])
grid on
set(gca, 'FontSize',18)
hAxes.TickLabelInterpreter = 'latex';
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
