clc
clearvars
close all
ch_func = Channel_functions();
addpath('OTFS_functions');

mod = '16QAM';
ChType = 'VehicularA';
v = 300;                    % Moving speed of user in km/h

nSym                    = 14;   % Number of symbols within one frame
N_CH                    = 1000; % Number of channel realizations
EbN0dB                  = 40; % bit to noise ratio
%% Physical Layer Specifications 
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 44;             % Number of data subcarriers
nPSC                   = 8;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
%% Proposed estimator: Preamble and pilots
pilots_locations       = [4,8,15,22,44,51,58,62].'; % Pilot subcarriers positions
pilots                 = [1 1 1 1 1 1 -1 -1].';
data_locations         = [2:3 5:7, 9:14, 16:21, 23:27, 39:43, 45:50, 52:57, 59:61, 63:64].'; % Data subcarriers positions
% Pre-defined preamble in frequency domain
dp = [ 0 0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % preamble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);               % set of allocated subcarriers                  
Kon                    = length(Kset);              % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
xp                     = ifft(dp);
xp_cp                  = [xp(end-K_cp+1:end); xp];  % Adding CP to the time domain preamble
preamble_size          = 1;
preamble_80211p        = repmat(xp_cp,1,preamble_size);         % IEEE 802.11p preamble symbols (tow symbols)

%% Benchmark Channel Estimator: Pilot power
pilotPower = max(EbN0dB);
pilotGuardSize = floor(0.32*nFFT); 

%% Bits Modulation Technique
Mod_Type                  = 1; % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(mod,'QPSK') == 1)
         nBitPerSym            = 2; 
    elseif (strcmp(mod,'16QAM') == 1)
         nBitPerSym            = 4; 
    elseif (strcmp(mod,'64QAM') == 1)
         nBitPerSym            = 6; 
    end
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM        
end

%% Bit to Noise Ratio
EbN0Lin                   = 10.^(EbN0dB/10);
%snr_p = Ep/KN0 => N0 = Ep/(K*snr_p)
N0 = Ep./(nFFT*EbN0Lin);

%% --------- Scrambler Parameters ---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% --------- Convolutional Coder Parameters -----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
rate                      = 1/2;

%% Simulation Parameters 
N_SNR                   = length(EbN0Lin); % SNR length

%% OTFS Precoder
hPrecoder = OtfsPrecoder( ...
    'NumDelayBins', nFFT, ...
    'NumDopplerBins', nSym, ...
    'NumTransmitAntennas', 1, ...
    'NumReceiveAntennas', 1);

% DFT matrices
W_nSym =  fft(eye(nSym))/sqrt(nSym);
W_nFFT =  fft(eye(nFFT))/sqrt(nFFT);
W_Spred = eye(nFFT,nFFT) * W_nFFT;  % DFT spreading matrix 

% CP insertion matrix
T_CP = [zeros(K_cp,nFFT-K_cp) eye(K_cp) ; eye(nFFT)];

% CP Removal Matrix
R_CP = [zeros(nFFT,K_cp)  eye(nFFT)];

% OFDM Transmitter and Receiver Matrix for the channel
% Proposed, considering preamble
GTX_OFDM_proposed = kron((eye(nSym+size(preamble_80211p,2))), T_CP*W_Spred');
GRX_OFDM_proposed =  kron((eye(nSym+size(preamble_80211p,2))), W_Spred*R_CP);
% Benchmark
GTX_OFDM_raviteja = kron(eye(nSym), T_CP*W_Spred');
GRX_OFDM_raviteja =  kron(eye(nSym), W_Spred*R_CP);

% Matrix for all L or K (Kronecker product with identity)
WnDSC = kron(W_nSym,sparse(eye(nFFT)));  
WnSym = kron(sparse(eye(nSym)),W_nFFT); 

% Encoding matrix
C =  WnSym  * WnDSC';
GTX_OTFS = GTX_OFDM_raviteja * C;
GRX_OTFS = C' *  GRX_OFDM_raviteja;

% Transmitted signals
s_OTFS_TF_CP_raviteja          = zeros(nFFT+K_cp,nSym,N_CH);
s_OTFS_TF_CP_Preamble_proposed = zeros(K+K_cp,nSym+preamble_size,N_CH);
output_HPA_raviteja_final      = zeros(nFFT+K_cp,nSym,N_CH);
output_HPA_proposed            = zeros(K+K_cp,nSym+preamble_size,N_CH);

%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    for n_ch = 1:N_CH(n_snr) % loop over channel realizations
        %% ----------- Transmitter -------------
        %% Delay-Doppler domain
        % Bits Stream Generation 
        txBits = randi(2, nDSC * nSym  * nBitPerSym * rate,1)-1;
        % Data Scrambler 
        scrambledData = wlanScramble(txBits,scramInit);
        % Convolutional Encoder
        dataEnc = convenc(scrambledData,trellis);
        % Bits Mapping: M-QAM Modulation
        TxBits_Coded = reshape(dataEnc,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData_Coded = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m)*2^(m-1);
        end 
        % M-QAM Modulation
        Modulated_Bits_Coded  = 1/sqrt(Pow) * qammod(TxData_Coded,M); 

        % Benchmark: Pilots and band guard insertion in the DD domain
        txBlocks = zeros(K,nSym);
        txBlocks(pilotGuardSize+1:end,:) = Modulated_Bits_Coded;
        txBlocks(1,1) = pilotPower;

        %% OTFS precodification ISFFT
        x_OTFS_data_raviteja = hPrecoder.encode(txBlocks);
        x_OTFS_data_proposed = hPrecoder.encode(Modulated_Bits_Coded);
        % Proposed: Pilots and band guard insertion in the TF domain
        s_OTFS_proposed = zeros(K,nSym);
        s_OTFS_proposed(data_locations,:) = x_OTFS_data_proposed;
        s_OTFS_proposed(pilots_locations,:) = repmat(pilots,1,nSym);

        %% Time domain 
        s_OTFS_TF_raviteja = W_Spred' * x_OTFS_data_raviteja;
        s_OTFS_TF_proposed = W_Spred' * s_OTFS_proposed;
        % Appending cylic prefix
        CP_Coded_raviteja = s_OTFS_TF_raviteja((nFFT - K_cp +1):nFFT,:);
        s_OTFS_TF_CP_raviteja = [CP_Coded_raviteja; s_OTFS_TF_raviteja];       
        CP_Coded_proposed = s_OTFS_TF_proposed((nFFT - K_cp +1):nFFT,:);
        s_OTFS_TF_CP_proposed = [CP_Coded_proposed; s_OTFS_TF_proposed];       
        % Proposed: Appending preamble symbol 
        s_OTFS_TF_CP_Preamble_proposed = [ preamble_80211p s_OTFS_TF_CP_proposed];

        % PAPR linear case
        PAPR_raviteja(n_ch,:) = max(abs(s_OTFS_TF_CP_raviteja).^2)./mean(abs(s_OTFS_TF_CP_raviteja).^2);
        PAPR_proposed(n_ch,:) = max(abs(s_OTFS_TF_CP_Preamble_proposed).^2)./mean(abs(s_OTFS_TF_CP_Preamble_proposed).^2);
     end
end
%% CCDF
% Linear case
[h1_raviteja,t1_raviteja] = hist(PAPR_raviteja(:),length(PAPR_raviteja(:)));
h1_raviteja = h1_raviteja/length(PAPR_raviteja(:));
tdB_Raviteja = 10*log10(t1_raviteja);

CCDF_raviteja = 1 - cumsum(h1_raviteja); 

[h1_proposed,t1_proposed] = hist(PAPR_proposed(:),length(PAPR_proposed(:)));
h1_proposed = h1_proposed/length(PAPR_proposed(:));
tdB_proposed = 10*log10(t1_proposed);

CCDF_proposed = 1 - cumsum(h1_proposed); 

figure(1)
colorOrder = get(gca, 'ColorOrder');
semilogy(tdB_Raviteja,CCDF_raviteja ,'--','LineWidth',2,'color',colorOrder(2,:))
hold on
semilogy(tdB_proposed,CCDF_proposed,'--','LineWidth',2,'color',colorOrder(4,:))
grid on
set(gca, 'FontSize',18)
hAxes.TickLabelInterpreter = 'latex';
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
legend('TCE','LS-LSTM-NN','Location','best','Interpreter','latex');
xlabel('$\lambda$ [dB]')
ylabel('$\mathrm{CCDF}= \bf{P}(\textrm{PAPR}> \lambda)$')
axis([0 20 10^-2 10^0])

% Adjusts to the PAPR to show the probability 1
tdB_Raviteja = [0:0.5:min(tdB_Raviteja) tdB_Raviteja];
tdB_proposed = [0:0.5:min(tdB_proposed) tdB_proposed];

rest_raviteja = length(tdB_Raviteja) - length(CCDF_raviteja);
rest_proposed = length(tdB_proposed) - length(CCDF_proposed);

CCDF_Linear_raviteja_comp = [ones(rest_raviteja,1)' CCDF_raviteja];
CCDF_Linear_proposed_comp = [ones(rest_proposed,1)' CCDF_proposed];

figure(2)
colorOrder = get(gca, 'ColorOrder');
semilogy(tdB_Raviteja,CCDF_Linear_raviteja_comp ,'--s','LineWidth',2,'MarkerSize',10,'MarkerIndices',[1:10:rest_raviteja 1:1000:length(CCDF_raviteja)],'MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:))
hold on
semilogy(tdB_proposed,CCDF_Linear_proposed_comp,'--h','LineWidth',2,'MarkerSize',10,'MarkerIndices',[1:10:rest_proposed 1:1000:length(CCDF_proposed)],'MarkerFaceColor',colorOrder(4,:),'color',colorOrder(4,:))
grid on
set(gca, 'FontSize',18)
hAxes.TickLabelInterpreter = 'latex';
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
legend('TCE','LS-LSTM-NN','Location','best','Interpreter','latex');
xlabel('$\lambda$ [dB]')
ylabel('$\mathrm{CCDF}= \bf{P}(\textrm{PAPR}> \lambda)$')
axis([0 20 10^-2 10^0])