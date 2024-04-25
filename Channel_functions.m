function func = Channel_functions()
%% --------------- Memeber functions Declaration --------------------------
func.GenRandomNoise = @GenRandomNoise;
func.GenFadingChannel = @GenFadingChannel;
func.GenFastFadingChannel = @GenFastFadingChannel;
func.ApplyChannel_TF = @ApplyChannel_TF;
func.ApplyChannel_FD = @ApplyChannel_FD;
func.UpdateSeed   = @UpdateSeed;
func.GetSeed      = @GetSeed;
func.SetSeed      = @SetSeed;
func.GetConvolutionMatrix = @GetConvolutionMatrix;
%% --------------- Implementation -----------------------------------------
function v = GenRandomNoise(siz, N0)
v = sqrt(N0/2) * (randn(siz)+1j*randn(siz));
end

function s = UpdateSeed(rchan)
release(rchan);
rchan.Seed = rchan.Seed + 1;
s = rchan.Seed;
end

function ChannelSeed = GetSeed(rchan)

    ChannelSeed = rchan.Seed;
end

function SetSeed(rchan,seed)
 release(rchan);
    rchan.Seed = seed;
end

function [ He, Y, pathgains] = ApplyChannel_TF( rchan, X, Ncp)
[Ns, NB] = size(X);
D = zeros(Ns,NB);
% Estimate the channel appling a impulse
D(Ncp+1,:) = 1;
He = zeros(size(D));
for nb=1:NB
    He(:,nb) = step(rchan, D(:,nb));
end
% reset the channel to first state which correspond to the estimation
reset(rchan);
[y,pathgains] = rchan(X(:));
Y = reshape(y,Ns, NB);
end

function ConvolutionMatrix = GetConvolutionMatrix(rchan,ImpulseResponse,txAntennas,rxAntennas)
    % returns the time-variant convolution matrix
    IndexDelayTaps = find(rchan.PathDelays)';
    % Mapping Convolution Matrix
    NrOnes = sum(size(ImpulseResponse,1):-1:(size(ImpulseResponse,1)-length(rchan.PathDelays)+1));
    MappingConvolutionMatrix = nan(NrOnes,2);
    for i_NrTaps = 1:length(rchan.PathDelays)
        MappingConvolutionMatrix(sum(size(ImpulseResponse,1):-1:size(ImpulseResponse,1)-i_NrTaps+2)+(1:size(ImpulseResponse,1)-i_NrTaps+1),:) = [-i_NrTaps+1+(i_NrTaps:size(ImpulseResponse,1)).' (i_NrTaps:size(ImpulseResponse,1)).'];
    end
    % Faster mapping because zeros are not mapped
    MappingConvolutionMatrixFast = MappingConvolutionMatrix;
    for i_NrTaps = find(rchan.PathDelays==0)
        MappingConvolutionMatrixFast(sum(size(ImpulseResponse,1):-1:size(ImpulseResponse,1)-i_NrTaps+2)+(1:size(ImpulseResponse,1)-i_NrTaps+1),:) = -1;
    end  
    MappingConvolutionMatrixFast(MappingConvolutionMatrixFast(:,1)==-1,:)=[];

    CancelElementsConvolutionMatrix = ones(size(ImpulseResponse,1),length(rchan.PathDelays))==1;
    for i_NrTaps = 1:length(rchan.PathDelays)-1
        CancelElementsConvolutionMatrix(i_NrTaps,(end-length(rchan.PathDelays)+1+i_NrTaps):end) = false;
    end
    CancelElementsConvolutionMatrixFast = CancelElementsConvolutionMatrix(:,IndexDelayTaps);

    ConvolutionMatrix = cell( rxAntennas, txAntennas );
    MaximumDopplerShift = rchan.MaximumDopplerShift;
    if MaximumDopplerShift>0
        for iTx = txAntennas
            for iRx = rxAntennas
                ImpulseResponseTemp = ImpulseResponse(:,IndexDelayTaps,iRx,iTx);
                ConvolutionMatrix{iRx,iTx} = sparse(MappingConvolutionMatrixFast(:,2),MappingConvolutionMatrixFast(:,1),ImpulseResponseTemp(CancelElementsConvolutionMatrixFast),size(ImpulseResponse,1),size(ImpulseResponse,1));
            end
        end
    else
        for iTx = txAntennas
            for iRx = 1:obj.Nr.rxAntennas
                ImpulseResponseTemp = obj.ImpulseResponse(ones(size(ImpulseResponse,1)),IndexDelayTaps,iRx,iTx);
                ConvolutionMatrix{iRx,iTx} = sparse(MappingConvolutionMatrixFast(:,2),MappingConvolutionMatrixFast(:,1),ImpulseResponseTemp(CancelElementsConvolutionMatrixFast),size(ImpulseResponse,1),size(ImpulseResponse,1));
            end
        end
    end
end


function rchan = GenFadingChannel( ChType, fD, fs)
switch ChType
    case 'PedestrianA'
        PathDelays = 1e-9.*[0 110 190 410];
        avgPathGains = [0 -9.7 -19.2 -22.8];
    case 'PedestrianB'
        PathDelays = 1e-9.*[0 200 800 1200 2300 3700];
        avgPathGains = [0 -0.9 -4.9 -8.0 -7.8 -23.9];  
    case 'VehicularA'
        PathDelays = 1e-9.*[0 310 710 1090 1730 2510];
        avgPathGains = [0 -1 -9 -10 -15 -20];
    case 'ExtendedVehicularA'
        PathDelays = 1e-9.*[0 30 150 310 370 710 1090 1730 2510];
        avgPathGains = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
    case 'ExtendedPedestrianA'
        PathDelays = 1e-9.*[0 30 70 90 110 190 410];
        avgPathGains = [0 -1 -2 -3 -8 -17.2 -20.8];
    case 'VehicularB'
        PathDelays = 1e-9.*[0 300 8900 12900 17100 20000];
        avgPathGains = [-2.5 0 -12.8 -10.0 -25.2 -16];
    case 'VTV_UC'
    PathDelays = 1e-9.*[0, 1, 100, 101, 102, 200, 201, 202, 300, 301, 400, 401];
    avgPathGains = [0, 0, -10, -10, -10, -17.8, -17.8, -17.8, -21.1, -21.1, -26.3,-26.3];
    %-------Channel Models Used in Vehicular communications----------------
    otherwise
        error('Channel model unknown');
end
rchan = comm.RayleighChannel('SampleRate',fs, ...
    'PathDelays',PathDelays, ...
    'AveragePathGains',avgPathGains, ...
    'MaximumDopplerShift',fD,...
    'DopplerSpectrum',{doppler('Jakes')},...,
    'PathGainsOutputPort',1,...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22);
end

    function rchan = GenFastFadingChannel( ChType, fD, fs,N,CP,nSym)
    switch ChType
        case 'PedestrianA'
            PathDelays = 1e-9.*[0 110 190 410];
            avgPathGains = [0 -9.7 -19.2 -22.8];
        case 'PedestrianB'
            PathDelays = 1e-9.*[0 200 800 1200 2300 3700];
            avgPathGains = [0 -0.9 -4.9 -8.0 -7.8 -23.9];  
        case 'VehicularA'
            PathDelays = 1e-9.*[0 310 710 1090 1730 2510];
            avgPathGains = [0 -1 -9 -10 -15 -20];
        case 'ExtendedVehicularA'
            PathDelays = 1e-9.*[0 30 150 310 370 710 1090 1730 2510];
            avgPathGains = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
        case 'ExtendedPedestrianA'
            PathDelays = 1e-9.*[0 30 70 90 110 190 410];
            avgPathGains = [0 -1 -2 -3 -8 -17.2 -20.8];
        case 'VehicularB'
            PathDelays = 1e-9.*[0 300 8900 12900 17100 20000];
            avgPathGains = [-2.5 0 -12.8 -10.0 -25.2 -16];

        %-------Channel Models Used in Vehicular communications----------------
        otherwise
            error('Channel model unknown');
    end

    rchan = Channel.FastFading(...
    fs,...                                                              % Sampling rate (Samples/s)
    PathDelays,...                                                      % Power delay profile, either string or vector: 'Flat', 'AWGN', 'PedestrianA', 'PedestrianB', 'VehicularA', 'VehicularB', 'ExtendedPedestrianA', 'ExtendedPedestrianB', or 'TDL-A_xxns','TDL-B_xxns','TDL-C_xxns' (with xx the RMS delay spread in ns, e.g. 'TDL-A_30ns'), or [1 0 0.2] (Self-defined power delay profile which depends on the sampling rate) 
    (N+CP)*nSym,...                                                               % Number of total samples
    fD,...                                                               % Maximum Doppler shift
    'Jakes',...                                                         % Which Doppler model: 'Jakes', 'Uniform', 'Discrete-Jakes', 'Discrete-Uniform'.                                       
    1, ...                                                              % Number of paths for the WSSUS process. Only relevant for a 'Jakes' and 'Uniform' Doppler spectrum                                                 
    1,...                                                               % Number of transmit antennas
    1,...                                                               % Number of receive antennas
    true ...                                                            % Gives a warning if the predefined delay taps of the channel do not fit the sampling rate. This is usually not much of a problem if they are approximatly the same.
    );
end
%% --------------- END of Implementation ----------------------------------
end
