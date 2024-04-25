clc
clear all
close all

M = 64;
N = 5:5:50;
M_d = 44;
M_on = 52;
M_p = 8;

% MMSE-DD
Threshold_CE = M .* N + M_d^3 * N.^3;
% AMP-FO
% Threshold_CE = M .* N + 15*N*M*9 + 15*N*M*16;
% MMSE-TF
LSLSTMNN = M_on^2 + M_p^2 + M_on*M_p + M_d.*N;       

colorOrder = get(gca, 'ColorOrder');
semilogy(N, Threshold_CE,'-s','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',10,'LineWidth',2);
hold on;
semilogy(N, LSLSTMNN, '-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
legend('TCE','LS-LSTM-NN','Location','best','Interpreter','latex');
xlabel('$N$');
ylabel('Complexity');
axis([5 50 10^1 2*10^10])
xticks(N)
yticks([10^1 10^4 10^7 10^10])
grid on
set(gca, 'FontSize',18)
hAxes.TickLabelInterpreter = 'latex';
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')