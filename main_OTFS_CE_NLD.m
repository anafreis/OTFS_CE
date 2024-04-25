clc
clearvars
close all
ch_func = Channel_functions();
addpath('OTFS_functions');

mod = '16QAM';
ChType = 'VehicularA';
v = 300;            % Moving speed of user in km/h
IBO = 4;

nSym                    = 14;    % Number of symbols within one frame
N_CH                    = [2000; 2000; 2000; 2000; 2000; 2000; 2000; 2000; 10000]; % Number of channel realizations
EbN0dB                  = 30;    % bit to noise ratio

pathdata = [num2str(nSym) 'Sym_' mod '_VehA_' num2str(v) 'kmh_IBO' num2str(IBO)'];

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
Tsignal                = Tgi+Tfft;       % Total duration of symbol = Guard time + FFT period = 8us
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

%% Vehicular Channel Model Parameters
L_Tabs                    = 12;
fs                        = nFFT*deltaF;            % Sampling frequency in Hz, 
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
c                         = 3e8;                    % Speed of Light in m/s
fD                        = (v/3.6)/c*fc;           % Doppler freq in Hz
plotFlag                  = 0;                      % 1 to display the channel frequency response
rchan_OTFS    = ch_func.GenFadingChannel(ChType, fD, fs);
init_seed = 22;
numtx = 1;            % Number of transmit antennas
numrx = 1;            % Number of receive antennas
D = 1/(nSym*nFFT*Tsignal); % Doppler resolution
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

%% ---------HPA variables: 3GPP_POLY ---------
fig = 0;         % Plot curves if fig == 1
[s_AMAM_3GPP, s_AMPM_3GPP, val_m1dB_hpa] = function_HPA_WONG5_Poly(fig);

%% Simulation Parameters 
N_SNR                   = length(EbN0dB); % SNR length

% Initiating vectors
NMSE_raviteja        = zeros(N_SNR,1);
Ber_Ideal_TF         = zeros(N_SNR,1);
Ber_Ideal_DD         = zeros(N_SNR,1);
Ber_est_raviteja_DD  = zeros(N_SNR,1);
BLER_est_raviteja_DD  = zeros(N_SNR,1);

% average channel power E(|hf|^2)
Phf_H_Total             = zeros(N_SNR,1);

%% OTFS Precoder
hPrecoder = OtfsPrecoder( ...
    'NumDelayBins', nFFT, ...
    'NumDopplerBins', nSym, ...
    'NumTransmitAntennas', numtx, ...
    'NumReceiveAntennas', numrx);

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
%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;    
    TX_Bits_Stream_Structure                = zeros(nDSC * nSym  * nBitPerSym * rate, N_CH(n_snr));
    Received_Symbols_FFT_Structure          = zeros(Kon,nSym, N_CH(n_snr));
    True_Channels_Structure                 = zeros(Kon, nSym + preamble_size, N_CH(n_snr));
    LS_Interpolation_Structure              = zeros(Kon,  nSym + preamble_size, N_CH(n_snr));

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

        % Applying NLD model
        [output_HPA_proposed] = NLD(s_AMAM_3GPP,s_AMPM_3GPP,val_m1dB_hpa,IBO,s_OTFS_TF_CP_Preamble_proposed);
        % Only for the data in the benchmark
        [output_HPA_raviteja] = NLD(s_AMAM_3GPP,s_AMPM_3GPP,val_m1dB_hpa,IBO,s_OTFS_TF_CP_raviteja((pilotGuardSize+K_cp+1):end,:));
        output_HPA_raviteja_final = [s_OTFS_TF_CP_raviteja(1:(pilotGuardSize+K_cp),:); output_HPA_raviteja];

        %% ----------- Channel -------------
        %% OTFS 
        % Channel realization
        [ h_proposed, ~, pathgains_proposed] = ch_func.ApplyChannel_TF(rchan_OTFS, output_HPA_proposed, K_cp);
        [ h_raviteja, ~, pathgains_raviteja] = ch_func.ApplyChannel_TF(rchan_OTFS, output_HPA_raviteja_final, K_cp);
        release(rchan_OTFS);
        rchan_OTFS.Seed = rchan_OTFS.Seed+1;

        % Received signal
        H_proposed = ch_func.GetConvolutionMatrix(rchan_OTFS,pathgains_proposed,numtx,numrx);   
        r_OTFS_noNoise_CP_proposed          = cell2mat(H_proposed) * output_HPA_proposed(:);
        r_OTFS_noNoise_CP_proposed          = reshape(r_OTFS_noNoise_CP_proposed,(nFFT+K_cp),nSym+size(preamble_80211p,2));
        r_OTFS_noNoise_proposed_preamble    = r_OTFS_noNoise_CP_proposed((K_cp+1):end,1:preamble_size);
        r_OTFS_noNoise_proposed_Symbols     = r_OTFS_noNoise_CP_proposed((K_cp+1):end,preamble_size+1:end);

        H_raviteja = ch_func.GetConvolutionMatrix(rchan_OTFS,pathgains_raviteja,numtx,numrx);   
        r_OTFS_noNoise_CP_raviteja = cell2mat(H_raviteja) * output_HPA_raviteja_final(:);
        r_OTFS_noNoise_CP_raviteja = reshape(r_OTFS_noNoise_CP_raviteja,(nFFT+K_cp),nSym);
        r_OTFS_noNoise_raviteja    = r_OTFS_noNoise_CP_raviteja((K_cp+1):end,:);
        % Benchmark: Channel in the DD domain
        H1_raviteja = GRX_OTFS*cell2mat(H_raviteja)*GTX_OTFS;

        % add noise
        noise_preamble           = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(r_OTFS_noNoise_proposed_preamble,2)], 1);
        r_OTFS_p_proposed        = r_OTFS_noNoise_proposed_preamble + noise_preamble;
        noise                    = sqrt(K*N0(n_snr))*ch_func.GenRandomNoise([K,size(r_OTFS_noNoise_proposed_Symbols,2)], 1);
        r_OTFS_symbols_proposed  = r_OTFS_noNoise_proposed_Symbols + noise;
        r_OTFS_raviteja          = r_OTFS_noNoise_raviteja + noise;

       % Proposed: Calculate One Tap Channels in the TF domain
       hf_TAPS_proposed  = full( reshape(sum((GRX_OFDM_proposed*cell2mat(H_proposed)).*GTX_OFDM_proposed.',2), nFFT, [] )); 
       Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + norm(hf_TAPS_proposed(Kset))^2;

       % Cyclic prefix loss
       cpl     = 1 - (1/K_cp);
       Pn      = 1/cpl * 1/log2(M) * 10^(-EbN0dB(n_snr)/10); 

       % MMSE Equalizer for OTFS in the TF domain
       Scaling_OFDM     =  repmat(1./(mean(1 ./( 1 + Pn./abs( hf_TAPS_proposed(data_locations,:) ).^2 ),1)), nDSC,1);
       e_OFDM     = Scaling_OFDM     .*   conj(hf_TAPS_proposed(data_locations,:))    ./( abs(hf_TAPS_proposed(data_locations,:)).^2     + Pn );

       %% ----------- Receiver -------------
       %% Frequency domain
       % OFDM Demodulation FFT
       y_OTFS_p_proposed =  W_Spred * r_OTFS_p_proposed;                           % Preamble
       y_OTFS_proposed   =  W_Spred * r_OTFS_symbols_proposed;                     % Symbols
       % MMSE equalization in the TF domain
       Equalized_Symbols_Ideal_OTFS = y_OTFS_proposed(data_locations,:)  .* e_OFDM(:,preamble_size+1:end); % Ideal OTFS
      
       %% Proposed channel estimation: LS Estimate at Preambles
       he_LS_Preamble_OTFS = (sum(y_OTFS_p_proposed(Kset,:),2)./(preamble_size.*predefined_preamble(Kset).'));
       H_LS_OTFS_preamble = repmat(he_LS_Preamble_OTFS,1,preamble_size);

       % Interpolation with the pilots
       H_interpolation_OTFS = zeros(nFFT,nSym);
       H_interpolation_OTFS(pilots_locations,:) = y_OTFS_proposed(pilots_locations,:) ./ s_OTFS_proposed(pilots_locations,:);

       H_LS_interpolation_OTFS = zeros(Kon,preamble_size+nSym); 
       H_LS_interpolation_OTFS(:,1:preamble_size)       = H_LS_OTFS_preamble;
       H_LS_interpolation_OTFS(:,preamble_size+1:end)   = H_interpolation_OTFS(Kset,:);

       %% Delay-Doppler Domain
       % OTFS decodification SFFT
       y_OTFS_raviteja     = hPrecoder.decode(W_Spred * r_OTFS_raviteja);
       x_est_Ideal_OTFS_TF = hPrecoder.decode(Equalized_Symbols_Ideal_OTFS);

       %% Benchmark channel estimation
       y_OTFS2 = y_OTFS_raviteja;
       y_OTFS2(pilotGuardSize+1:end,:) = 0;
       y_OTFS2 = y_OTFS2/pilotPower;           
       y_OTFS2 = reshape(y_OTFS2,nFFT*nSym,1);
        
       threshold = 0.01;
       y_OTFS2(abs(y_OTFS2) < threshold) = 1e-17+1e-17i;
       H1est = circulant(y_OTFS2,1);   

       est_error = abs(H1_raviteja-H1est).^2;
       NMSE_raviteja(n_snr) = NMSE_raviteja(n_snr) + sum(est_error(:))/numel(H1est);

       % MMSE Equalizer in the delay-Doppler domain
       e_OTFS_ideal =      (H1_raviteja' / (H1_raviteja*H1_raviteja' + Pn*eye(nFFT*nSym)));
       e_OTFS_est   =      (H1est' / (H1est*H1est' + Pn*eye(nFFT*nSym))); 

       % MMSE equalization in the DD domain
       x_equalized_ideal            = e_OTFS_ideal * y_OTFS_raviteja(:);
       x_equalized_ideal_reshape    = reshape(x_equalized_ideal,nFFT,nSym);
       x_equalized_raviteja         = e_OTFS_est * y_OTFS_raviteja(:);
       x_equalized_raviteja_reshape = reshape(x_equalized_raviteja,nFFT,nSym);

       % QAM - DeMapping
       De_Mapped_Ideal_TF     = qamdemod(sqrt(Pow) * x_est_Ideal_OTFS_TF,M);
       De_Mapped_Ideal_DD     = qamdemod(sqrt(Pow) * x_equalized_ideal_reshape(pilotGuardSize+1:end,:),M);
       De_Mapped_raviteja_DD  = qamdemod(sqrt(Pow) * x_equalized_raviteja_reshape(pilotGuardSize+1:end,:),M);
        
       % Bits Extraction
       Bits_Ideal_TF         = zeros(nDSC,nSym,log2(M));
       Bits_ext_ideal_DD     = zeros(nDSC,nSym,log2(M));
       Bits_ext_raviteja_DD  = zeros(nDSC,nSym,log2(M));

       for b = 1:nSym
           Bits_Ideal_TF(:,b,:)         = de2bi(De_Mapped_Ideal_TF(:,b),nBitPerSym);
           Bits_ext_ideal_DD(:,b,:)     = de2bi(De_Mapped_Ideal_DD(:,b),nBitPerSym);
           Bits_ext_raviteja_DD(:,b,:)  = de2bi(De_Mapped_raviteja_DD(:,b),nBitPerSym);    
       end
 
       % Viterbi decoder
       Decoded_Bits_Ideal_TF     = vitdec(Bits_Ideal_TF(:),trellis,tbl,'trunc','hard');
       Decoded_Bits_ideal_DD     = vitdec(Bits_ext_ideal_DD(:),trellis,tbl,'trunc','hard');
       Decoded_Bits_raviteja_DD  = vitdec(Bits_ext_raviteja_DD(:),trellis,tbl,'trunc','hard');
        
       % De-scrambler Data
       deScramble_Ideal_Final_TF    = wlanScramble(Decoded_Bits_Ideal_TF,scramInit);
       deScramble_Bits_ideal_DD     = wlanScramble(Decoded_Bits_ideal_DD,scramInit);
       deScramble_Bits_raviteja_DD  = wlanScramble(Decoded_Bits_raviteja_DD,scramInit);

       % BER Calculation
       ber_Ideal_TF         = biterr(deScramble_Ideal_Final_TF,txBits);
       ber_Ideal_DD         = biterr(deScramble_Bits_ideal_DD, txBits);
       ber_est_raviteja_DD  = biterr(deScramble_Bits_raviteja_DD,txBits);

       Ber_Ideal_TF (n_snr)             = Ber_Ideal_TF (n_snr) + ber_Ideal_TF;
       Ber_Ideal_DD (n_snr)             = Ber_Ideal_DD (n_snr) + ber_Ideal_DD;
       Ber_est_raviteja_DD (n_snr)      = Ber_est_raviteja_DD (n_snr) + ber_est_raviteja_DD;


       TX_Bits_Stream_Structure(:, n_ch) = txBits;
       Received_Symbols_FFT_Structure(:,:,n_ch) = y_OTFS_proposed(Kset,:);
       True_Channels_Structure(:,:,n_ch) = hf_TAPS_proposed(Kset,:);
       LS_Interpolation_Structure(:,:,n_ch)  = H_LS_interpolation_OTFS; 
    end
    save(['data_' pathdata '\Simulation_' num2str(n_snr)],...
           'TX_Bits_Stream_Structure',...
           'Received_Symbols_FFT_Structure',...
           'True_Channels_Structure',...
           'LS_Interpolation_Structure');
end
%% Bit Error Rate (BER)
BER_Ideal_TF_NLD            = Ber_Ideal_TF ./(N_CH .* nSym * nDSC * nBitPerSym);
BER_Ideal_DD_NLD            = Ber_Ideal_DD ./(N_CH .* nSym * nDSC * nBitPerSym);
BER_Raviteja_Est_NLD        = Ber_est_raviteja_DD ./(N_CH .* nSym * nDSC * nBitPerSym);

ERR_raviteja = NMSE_raviteja ./ N_CH;

figure
colorOrder = get(gca, 'ColorOrder');
semilogy(EbN0dB, BER_Ideal_TF_NLD,'*-','MarkerFaceColor',colorOrder(1,:),'color',colorOrder(1,:),'MarkerSize',8,'LineWidth',2);
hold on
semilogy(EbN0dB, BER_Ideal_DD_NLD,'-s','MarkerFaceColor',colorOrder(2,:),'color',colorOrder(2,:),'MarkerSize',8,'LineWidth',2);
semilogy(EbN0dB, BER_Raviteja_Est_NLD,'-^','MarkerFaceColor',colorOrder(3,:),'color',colorOrder(3,:),'MarkerSize',8,'LineWidth',2);
grid on
set(0,'defaulttextinterpreter','latex')
set(0,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')
axis([min(EbN0dB) max(EbN0dB) 10^-6 10^0])
xticks(min(EbN0dB):5:max(EbN0dB))
xlabel('SNR ($\xi$) [dB]');
ylabel('BER');
legend('OTFS CSI TF','OTFS CSI DD','Raviteja','FontSize',14,'Interpreter','latex');
set(gca, 'FontSize',16)

save(['data_' pathdata '\Simulation_variables'],'mod','Kset','fD','ChType');
save(['data_' pathdata '\Results_OTFS_NLD'],'BER_Ideal_TF_NLD','BER_Ideal_DD_NLD','BER_Raviteja_Est_NLD','ERR_raviteja');
