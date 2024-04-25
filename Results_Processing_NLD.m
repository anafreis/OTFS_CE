clc
clearvars
% close all
warning('off','all')
addpath('OTFS_functions');
addpath('Base_functions');

path = pwd;

mod = '16QAM';
ChType = 'VehicularA';
v = 300;                    % Moving speed of user in km/h
IBO = 4;

nSym                    = 14;   % Number of symbols within one frame

pathdata = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)];

algo                     = 'LS_Interpolation';
wave                     = 'OTFS';

% Loading Simulation Data
load(['data_' pathdata '\Simulation_variables.mat']);
%% ------ Bits Modulation Technique------------------------------------------
if(strcmp(mod,'QPSK') == 1)
     nBitPerSym            = 2; 
elseif (strcmp(mod,'16QAM') == 1)
     nBitPerSym            = 4; 
elseif (strcmp(mod,'64QAM') == 1)
     nBitPerSym            = 6; 
end
M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
load('indices.mat');
N_Test_Frames = length(testing_indices);
EbN0dB                    = (0:5:40)';
Pow                       = mean(abs(qammod(0:(M-1),M)).^2);
nFFT                      = 64;
preamble_size             = 1;
nDSC                      = 44;
nPSC                      = 8;              % Number of pilot subcarriers
nUSC                      = nDSC + nPSC;    % Number of total used subcarriersnUSC                     
K_cp                      = 16;
numtx                     = 1;
numrx                     = 1;
N_SNR                     = size(EbN0dB,1);
Phf                       = zeros(N_SNR,1);
%% --------- Scrambler Parameters ---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% --------- Convolutional Coder Parameters -----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;

%% Bit to Noise Ratio
EbN0Lin                   = 10.^(EbN0dB/10);
noiseVar = 1./EbN0Lin;  % Total power
 
NMSE_proposed_TF           = zeros(N_SNR,N_Test_Frames);
NMSE_proposed_DD           = zeros(N_SNR,N_Test_Frames);

Ber_TF_Ideal     = zeros(N_SNR,1);
Ber_LSTM_DNN_TF  = zeros(N_SNR,1);
Ber_LSTM_DNN_DD  = zeros(N_SNR,1);
Ber_DD_Ideal     = zeros(N_SNR,1);

dpositions              = [1:2, 4:6, 8:13, 15:20, 22:31, 33:38, 40:45, 47:49, 51:52].';  % Data positions in the set of allocated subcarriers Kset
fftshift(dpositions);

% OTFS Precoder
hPrecoder = OtfsPrecoder( ...
    'NumDelayBins', nFFT, ...
    'NumDopplerBins', nSym, ...
    'NumTransmitAntennas', numtx, ...
    'NumReceiveAntennas', numrx);

% DFT matrices
W_nDSC =  fft(eye(nDSC))/sqrt(nDSC);
W_nSym =  fft(eye(nSym))/sqrt(nSym);
W_nFFT =  fft(eye(nDSC))/sqrt(nDSC);
W_Spred = eye(nDSC,nDSC) * W_nFFT;  % DFT spreading matrix 

% OFDM Transmitter and Receiver Matrix for the channel
GTX_OFDM = kron((eye(nSym)), W_Spred');
GRX_OFDM =  kron((eye(nSym)), W_Spred);

%Matrix for all L or K (Kronecker product with identity)
WnDSC = kron(W_nSym,sparse(eye(nDSC)));  
WnSym = kron(sparse(eye(nSym)),W_nDSC); 

% Encoding matrix
C =  WnSym  * WnDSC';
GTX_OTFS = GTX_OFDM * C;
GRX_OTFS = C' *  GRX_OFDM;

rate = 2;
LSTM_size = (nDSC/rate + nPSC);

for ii = 1:N_SNR 
    i = ii;
    % Cyclic prefix loss
    cpl     = 1 - (1/K_cp);
    Pn      = 1/cpl * 1/log2(M) * 10^(-EbN0dB(ii)/10); 
    
    % Loading Simulation Parameters Results
    load(['data_' pathdata '\Simulation_' num2str(i) '.mat']);
      
    % Loading Results
    load([path '\Python_Codes - NLD\data\' algo '_LSTM_Less_DNN_' wave '_' num2str(LSTM_size) '15_Results_' num2str(i),'.mat']);
    LSTM_DNN_preamble = eval([algo '_LSTM_Less_DNN_' wave '_' num2str(LSTM_size) '15_corrected_y_',num2str(i)]);
    LSTM_DNN_preamble = reshape(LSTM_DNN_preamble(1:52,:) + 1i*LSTM_DNN_preamble(53:104,:), nUSC, nSym+preamble_size, N_Test_Frames);   
    LSTM_DNN = LSTM_DNN_preamble(:,(preamble_size+1):end,:);
    tic;
    for u = 1:N_Test_Frames
        % testing dataset (2000)
        if ii ~= find(EbN0dB == max(EbN0dB))
            c = u;
        % training dataset (10000)
        else 
            c = testing_indices(1,u);
        end 

        True_Channels_Structure_wopreamble = True_Channels_Structure(:,(preamble_size+1:end),:);
        Phf(ii)  = Phf(ii)  + norm(True_Channels_Structure_wopreamble(:,:,c))^ 2;

        H_LSTM_DNN = LSTM_DNN(:,:,u);
        H_LSTM_DNN_tf = H_LSTM_DNN(dpositions,:);
        % TF domain
        est_error_TF = abs(True_Channels_Structure_wopreamble(dpositions,:,c)-H_LSTM_DNN_tf).^2;
        NMSE_proposed_TF(ii,u) = sum(est_error_TF(:))/numel(H_LSTM_DNN_tf);
        % One-tap (scaled) MMSE followed by despreading
        H_ideal = True_Channels_Structure_wopreamble(dpositions,:,c);
        Scaling_OFDM_Ideal     =  repmat(1./(mean(1 ./( 1 + Pn./abs( H_ideal ).^2 ),1)), nDSC,1);
        e_OFDM_Ideal     = Scaling_OFDM_Ideal     .*   conj(H_ideal)    ./( abs(H_ideal).^2     + Pn );
        Equalized_Symbols_LSTM_DNN_TF_Ideal = Received_Symbols_FFT_Structure(dpositions,:,c) .* e_OFDM_Ideal;

        Scaling_OFDM_est     =  repmat(1./(mean(1 ./( 1 + Pn./abs( H_LSTM_DNN(dpositions,:) ).^2 ),1)), nDSC,1);
        e_OFDM_est     = Scaling_OFDM_est     .*   conj(H_LSTM_DNN(dpositions,:))    ./( abs(H_LSTM_DNN(dpositions,:)).^2     + Pn );
        Equalized_Symbols_LSTM_DNN_TF = Received_Symbols_FFT_Structure(dpositions,:,c) .* e_OFDM_est;

        % OTFS decodification SFFT
        x_est_LSTM_DNN_TF_Ideal = hPrecoder.decode(Equalized_Symbols_LSTM_DNN_TF_Ideal);
        x_est_LSTM_DNN_TF = hPrecoder.decode(Equalized_Symbols_LSTM_DNN_TF);

        % DD domain
        H_ideal_DD = sparse(double((GTX_OFDM*diag(H_ideal(:))*GRX_OFDM)));
        Hest_DD = sparse(double((GTX_OFDM*diag(H_LSTM_DNN_tf(:))*GRX_OFDM)));
        est_error_DD = abs(H_ideal_DD-Hest_DD).^2;
        NMSE_proposed_DD(ii,u) = sum(est_error_DD(:))/numel(Hest_DD);

        H_LSTM_DNN_DD_Ideal = GRX_OTFS*H_ideal_DD*GTX_OTFS;
        H_LSTM_DNN_DD       = GRX_OTFS*Hest_DD*GTX_OTFS;
        
        % OTFS decodification SFFT
        y_OTFS = hPrecoder.decode(Received_Symbols_FFT_Structure(dpositions,:,c));
        % MMSE Equalizer for OTFS in the delay-Doppler domain
        e_OTFS_Ideal =      (H_LSTM_DNN_DD_Ideal' / (H_LSTM_DNN_DD_Ideal*H_LSTM_DNN_DD_Ideal' + Pn*eye(nDSC*nSym)));
        e_OTFS       =      (H_LSTM_DNN_DD' / (H_LSTM_DNN_DD*H_LSTM_DNN_DD' + Pn*eye(nDSC*nSym)));
        
        Equalized_Symbols_DD_Ideal =  e_OTFS_Ideal * y_OTFS(:);
        Equalized_Symbols_LSTM_DNN_DD       =  e_OTFS * y_OTFS(:);
                
        x_est_DD_Ideal     = reshape(Equalized_Symbols_DD_Ideal,nDSC,nSym);
        x_est_LSTM_DNN_DD  = reshape(Equalized_Symbols_LSTM_DNN_DD,nDSC,nSym);

        % QAM - DeMapping
        De_Mapped_TF_Ideal    = qamdemod(sqrt(Pow) * x_est_LSTM_DNN_TF_Ideal,M);
        De_Mapped_LSTM_DNN_TF = qamdemod(sqrt(Pow) * x_est_LSTM_DNN_TF,M);
        De_Mapped_DD_Ideal    = qamdemod(sqrt(Pow) * x_est_DD_Ideal,M);
        De_Mapped_LSTM_DNN_DD = qamdemod(sqrt(Pow) * x_est_LSTM_DNN_DD,M);

        % Bits Extraction
        Bits_TF_Ideal          = zeros(nDSC,nSym,log2(M));
        Bits_LSTM_DNN_TF       = zeros(nDSC,nSym,log2(M));
        Bits_LSTM_DNN_DD       = zeros(nDSC,nSym,log2(M));
        Bits_DD_Ideal          = zeros(nDSC,nSym,log2(M));

        for b = 1:nSym
           Bits_TF_Ideal(:,b,:)      = de2bi(De_Mapped_TF_Ideal(:,b),nBitPerSym);
           Bits_LSTM_DNN_TF(:,b,:)   = de2bi(De_Mapped_LSTM_DNN_TF(:,b),nBitPerSym);
           Bits_DD_Ideal(:,b,:)      = de2bi(De_Mapped_DD_Ideal(:,b),nBitPerSym);
           Bits_LSTM_DNN_DD(:,b,:)   = de2bi(De_Mapped_LSTM_DNN_DD(:,b),nBitPerSym);

        end
        
        % Viterbi decoder
        Decoded_TF_Ideal          = vitdec(Bits_TF_Ideal(:),trellis,tbl,'trunc','hard');
        Decoded_Bits_LSTM_DNN_TF  = vitdec(Bits_LSTM_DNN_TF(:),trellis,tbl,'trunc','hard');
        Decoded_DD_Ideal          = vitdec(Bits_DD_Ideal(:),trellis,tbl,'trunc','hard');
        Decoded_Bits_LSTM_DNN_DD  = vitdec(Bits_LSTM_DNN_DD(:),trellis,tbl,'trunc','hard');

        % De-scrambler Data
        DeScramble_Bits_TF_Ideal     = wlanScramble(Decoded_TF_Ideal,scramInit);
        DeScramble_Bits_LSTM_DNN_TF  = wlanScramble(Decoded_Bits_LSTM_DNN_TF,scramInit);
        DeScramble_Bits_DD_Ideal     = wlanScramble(Decoded_DD_Ideal,scramInit);
        DeScramble_Bits_LSTM_DNN_DD  = wlanScramble(Decoded_Bits_LSTM_DNN_DD,scramInit);

        % BER Calculation
        ber_TF_Ideal    = biterr(DeScramble_Bits_TF_Ideal,TX_Bits_Stream_Structure(:,c));
        ber_LSTM_DNN_TF = biterr(DeScramble_Bits_LSTM_DNN_TF,TX_Bits_Stream_Structure(:,c));
        ber_DD_Ideal    = biterr(DeScramble_Bits_DD_Ideal,TX_Bits_Stream_Structure(:,c));
        ber_LSTM_DNN_DD = biterr(DeScramble_Bits_LSTM_DNN_DD,TX_Bits_Stream_Structure(:,c));

        Ber_TF_Ideal(ii)           = Ber_TF_Ideal(ii) + ber_TF_Ideal; 
        Ber_LSTM_DNN_TF(ii)        = Ber_LSTM_DNN_TF(ii) + ber_LSTM_DNN_TF; 
        Ber_DD_Ideal(ii)           = Ber_DD_Ideal(ii) + ber_DD_Ideal; 
        Ber_LSTM_DNN_DD(ii)        = Ber_LSTM_DNN_DD(ii) + ber_LSTM_DNN_DD; 

    end
    toc;
end

%% Bit Error Rate (BER)
BER_IDEAL_TF        = Ber_TF_Ideal / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_LSTM_NN_TF      = Ber_LSTM_DNN_TF / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_IDEAL_DD        = Ber_DD_Ideal / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_LSTM_NN_DD      = Ber_LSTM_DNN_DD / (N_Test_Frames * nSym * nDSC * nBitPerSym);
 
colorOrder = get(gca, 'ColorOrder');
subplot(1,2,1)
semilogy(EbN0dB, BER_IDEAL_TF,'k*-','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
hold on
semilogy(EbN0dB, BER_IDEAL_DD,'k^--','MarkerFaceColor','k','MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_LSTM_NN_TF,'-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_LSTM_NN_DD,'-<','MarkerFaceColor',colorOrder(5,:),'color',colorOrder(5,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
axis([min(EbN0dB) max(EbN0dB) 10^-8 10^0])
yticks([10^-8 10^-6 10^-4 10^-2 10^0])
xticks(min(EbN0dB):5:max(EbN0dB))
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
legend('OTFS CSI TF','OTFS CSI DD','LS-LSTM-NN - TF eq','LS-LSTM-NN - DD eq','FontSize',14,'Location','best','Interpreter','latex');
set(gca, 'FontSize',16)


%% Normalized Mean Square Error
ERR_LSTM_NN_TF      = mean(NMSE_proposed_TF,2);
ERR_LSTM_NN_DD      = mean(NMSE_proposed_DD,2);

colorOrder = get(gca, 'ColorOrder');
subplot(1,2,2)
semilogy(EbN0dB, ERR_LSTM_NN_TF,'-h','MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:),'MarkerSize',8,'LineWidth',2);
hold on
semilogy(EbN0dB, ERR_LSTM_NN_DD,'-<','MarkerFaceColor',colorOrder(5,:),'color',colorOrder(5,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
axis([min(EbN0dB) max(EbN0dB) 10^-8 10^0])
yticks([10^-8 10^-6 10^-4 10^-2 10^0])
xticks(min(EbN0dB):5:max(EbN0dB))
xlabel('SNR ($\xi$) [dB]');
ylabel('NMSE');
legend('LS-LSTM-NN - TF eq','LS-LSTM-NN - DD eq','FontSize',14,'Interpreter','latex');
set(gca, 'FontSize',16)
save(['data_' pathdata '\LSTM_NN_Results_Less_' wave '_' num2str(LSTM_size) '15'],'BER_IDEAL_TF','BER_IDEAL_DD','BER_LSTM_NN_TF','BER_LSTM_NN_DD','ERR_LSTM_NN_TF','ERR_LSTM_NN_DD');
