clc
clear all
% close all
EbN0dB = 0:5:40;
addpath('OTFS_functions');

v = 1000; % Moving speed of user in km/h
mod = '16QAM';
nSym = 14;
IBO = 4;
coderate = 1/2;
%% Physical Layer Specifications 
nFFT                   = 64;             % FFT size 
nDSC                   = 44;             % Number of data subcarriers
if isequal(mod,'16QAM')
    MCS_rate = log2(16)*coderate;
elseif isequal(mod,'QPSK')
    MCS_rate = log2(4)*coderate;
end

preamble_size = 1;                       % preamble considered in the proposed scheme
%% Nonlinear case
pathdata_NLD = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];
load(['data_' pathdata_NLD '/Results_OTFS_NLD'],'BER_Raviteja_Est_NLD');
load(['data_' pathdata_NLD '/LSTM_NN_Results_Less_OTFS_3015'])


%% Throughput
rate_Raviteja = (nSym*nDSC)/((nSym)*nFFT);
rate_proposal = (nSym*nDSC)/((nSym+preamble_size)*nFFT);

figure
colorOrder = get(gca, 'ColorOrder');
trRaviteja = rate_Raviteja * MCS_rate * (1 - BER_Raviteja_Est_NLD); 
trProposal = rate_proposal * MCS_rate * (1 - BER_LSTM_NN_TF);

plot(EbN0dB,trRaviteja,'-s','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
hold on
plot(EbN0dB,trProposal,'-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
grid on


legend('TCE','LS-LSTM-NN','Location','best','Interpreter','latex');

set(gcf,'position',[10,10,600,500])
axis([0 30 0 1.5])
xticks(0:5:30)

set(gca, 'FontSize',18)
hAxes.TickLabelInterpreter = 'latex';
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xlabel('SNR [dB]');
ylabel('Throughput (bps/Hz)');
