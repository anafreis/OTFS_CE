clc
clear all
% close all
EbN0dB = 0:5:40;

mod = '16QAM';
ChType = 'VehicularA';
v = 300;                    % Moving speed of user in km/h
IBO = 4;
nSym                    = 14;   % Number of symbols within one frame

%% Linear case
pathdata_lin = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_Linear'];
load(['data_' pathdata_lin '/Results_OTFS_Linear'],'BER_Raviteja_Est_linear');
load(['data_' pathdata_lin '/LSTM_NN_Results_Less_OTFS_3015'])

% BER
figure
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, BER_IDEAL_TF,'k*-','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
hold on
% semilogy(EbN0dB, BER_IDEAL_DD,'k^--','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_Raviteja_Est_linear,'-s','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_LSTM_NN_TF,'-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
if strcmp(mod,'QPSK')
axis([min(EbN0dB) 30 10^-8 10^0])
yticks([10^-8 10^-6 10^-4 10^-2 10^0])
elseif strcmp(mod,'16QAM')
axis([min(EbN0dB) 30 0.5*10^-4 10^0])
yticks([10^-4 10^-3 10^-2 10^-1 10^0])
end
xticks(min(EbN0dB):5:30)
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
legend('OTFS with perfect CSI','TCE','LS-LSTM-NN','FontSize',14,'Location','best','Interpreter','latex');
subtitle('Linear');
set(gca, 'FontSize',16)

%% Nonlinear case

pathdata_NLD = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];
load(['data_' pathdata_NLD '/Results_OTFS_NLD'],'BER_Raviteja_Est_NLD');
load(['data_' pathdata_NLD '/LSTM_NN_Results_Less_OTFS_3015'])

% BER
figure
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, BER_IDEAL_TF,'k*-','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
hold on
% semilogy(EbN0dB, BER_IDEAL_DD,'k^--','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_Raviteja_Est_NLD,'-s','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_LSTM_NN_TF,'-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
% semilogy(EbN0dB, BER_LSTM_NN_DD,'-<','MarkerFaceColor',colorOrder(5,:),'color',colorOrder(5,:),'MarkerSize',8,'LineWidth',2);

if strcmp(mod,'QPSK')
    axis([min(EbN0dB) 30 10^-6 10^0])
    yticks([10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 10^0])
end
if strcmp(mod,'16QAM')
    axis([min(EbN0dB) 30 10^-5 10^0])
    yticks([10^-5 10^-4 10^-3 10^-2 10^-1 10^0])
end
set(0,'defaulttextinterpreter','latex')
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
xticks(min(EbN0dB):5:30)
xlabel('SNR [dB]');
ylabel('BER');
legend('OTFS with perfect CSI','TCE','LS-LSTM-NN','FontSize',14,'Location','best','Interpreter','latex');
set(gca, 'FontSize',18)
title(['HPA, IBO = ' num2str(IBO) ' dB'])
grid on


