function [ rmsmed,integral,freqmedian,psdpeak,linenveli,rmsi,freqmediani,signal, temp ] = emg_labiocom(dat,freq,graficos,windowlength,overlap)
% emg_analisys
% Realiza a análise de EMG de 1 músculos
% Author: Prof. Dr. Paulo Roberto Pereira Santiago - EEFERP/USP
% 04/11/2013
%%
if nargin==1
    freq = 2000;
    windowlength = 3;
    overlap = 1.5;
    graficos = 1;
end
if nargin==2
    graficos = 1;
    windowlength = 0.125;
    overlap = 0.0625;
end
if nargin==3
    windowlength = 0.125;
    overlap = windowlength/2;
end

if nargin==4
    overlap = 0.0625;
end

windowlength = windowlength * freq;
overlap = overlap * freq;

% datemgvoltraw = importemg_delsys(dat);
datemgvoltraw = dat;
datemgvoltfilt = filtbutter(datemgvoltraw,[15 500],freq,'bandpass');
%

datemg = datemgvoltfilt(:,2)*1000000; % passando de volts para micro volts
% datemg = datemgvoltfilt(:,2);

% obl_ext_L = datemg(:,2);

%% tempo
[nl1,nc1] = size(datemg);
temp1 = [(0:nl1-1)/freq]'; % vetor tempo
%%
close all

%% GRAFICO 1 - RAW EMG

    figure(1)
%     subplot(2,2,1)
    plot(temp1,dat)
    % plot(rec_abd_L)
    ylabel ('Amplitude (V)');
    title('RAW EMG')
    axis tight
    grid on

%%

disp('Deseja selecionar uma intervalo de tempo para as análises?')
disp('Para "NÃO" digite 0')
disp('Para "SIM" digite quanto deseja retirar do início e fim em seg. i.e. [10 20]')
disp('para uma análise do 10 ao 20 segundo')

tanalise = input('Digite sua opção:  ');

if size(tanalise,2) == 1 && tanalise(1,1) == 0
    tini = 1;
    tfim = nl1;
end
if size(tanalise,2) == 2 && tanalise(1,1) == 0
        tini = 1;
        tfim = tanalise(1,2)*freq;
end

if size(tanalise,2) == 2 && tanalise(1,1) ~= 0
    tini = tanalise(1,1)*freq;
    tfim = tanalise(1,2)*freq;
end

signal = datemg(tini:tfim,:);

%% tempo
[nl,nc] = size(signal);
temp = [(0:nl-1)/freq]'; % vetor tempo
%%

%% GRAFICO 1 - RAW EMG
if graficos == 1
    figure(1)
    clf

    % subplot(2,2,1)
    plot(temp,signal)
    hold on
    % plot(rec_abd_L)
    ylabel ('Amplitude (µV)');
    title('EMG SELECT AND FILTER')
    grid on
    axis tight
end

%% SIGNAL RECTIFIED
signalD = detrend(signal);
signalDabs = abs(signalD);

%% GRAFICO 2 - FULL-WAVE RECTIFIED EMG
if graficos == 1
    figure(2)
    % subplot(2,2,2)
    plot(temp,signalDabs)
    hold on
    % plot(rec_abd_R_abs)
    ylabel ('Amplitude (µV)');
    title('FULL-WAVE EMG')
    axis tight
end

%% LINEAR ENVELOPE
signalenv = abs(filtbutter(signalDabs,50,freq,'low'));
signal_integ = trapz(temp,signalenv(:,2));



%% GRAFICO 3 - LINEAR ENVELOPE
if graficos == 1
    figure(2)
    % subplot(2,2,1)
    plot(temp,signalenv(:,2),'r','linewidth',2)
    % plot(rec_abd_L_linenvel(:,2))
    ylabel ('Amplitude (µV)');
    % title(['LEFT RECTUS ABDOMINIS (LINEAR ENVELOPE EMG = ',num2str(rec_abd_L_linenvel_integ),')'])
    title(['FULL-WAVE & LINEAR ENVELOPE = ',num2str(roundn(signal_integ,-1)),' µV.s)'])
    axis tight
end
%% RMS do sinal - window length e overlap
signalrms2 = rms2(signal,windowlength,overlap,'NO')'; 


%% 
length_rms2 = length(signalrms2);
temp_rms = [linspace(0,temp(end),length_rms2)]'; % vetor tempo rms
%%
%% RMS de todo o sinal
signalrms = rms(signal);

%% GRAFICO 4 - RMS do EMG
if graficos == 1
figure(3)
% subplot(2,2,1)
plot(temp_rms,signalrms2)
% plot(rec_abd_L_rms2)
hold on
line([temp_rms(1) temp_rms(end)],[signalrms , signalrms],'LineWidth',2,'LineStyle','--','Color','r')
% line([1 , length_rms2],[rec_abd_L_rms , rec_abd_L_rms],'LineWidth',2,'Line','--','Color','r')
ylabel ('RMS (µV)');
title(['EMG RMS = ',num2str(roundn(signalrms,-1)),' µV'])
xlabel('time (s)')
axis tight
end

%% Analise no domínio da frequência

[MPF_signal,PEAK_signal,F50_signal,F95_signal,F_signal,P_signal] = psd2(signal,freq);




%% GRAFICO 5 - PSD
if graficos == 1
    figure(4)
    % subplot(2,2,2)
    plot(F_signal,P_signal)
    ylabel ('Power (dB)');
    title('EMG - PSD')
    xlabel('Frequency (Hz)')
    axis tight
end

%% Frequencia mediana no dominio no temporal

Fs = freq;
t = temp;
nfft = 2^nextpow2(windowlength);
window = hanning(round(windowlength));
overlap = round(overlap);

% [signal_S,signal_F,signal_T,signal_P] = spectrogram(signal,window,overlap,window,Fs);
[signal_S,signal_F,signal_T,signal_P] = spectrogram(signal,window,overlap,nfft,Fs);
% [signal_P1,signal_F1] = pwelch(signal,window,overlap,nfft,Fs);

  

  %%
  medianfreqs_signal = ones(size(signal_P,2),1);

  
  %%
  
 for nn = 1:size(signal_P,2)
       % 
        signal_P_normcumsumpsd = cumsum(signal_P(:,nn))./sum(signal_P(:,nn));
        Ind1 = find(signal_P_normcumsumpsd <=0.5,1,'last');
        sizeInd1 = size(Ind1,1);
        
%         psdest = psd(spectrum.periodogram,signal,'Fs',2000,'NFFT',length(signal)); 
%    normcumsumpsd = cumsum(psdest.Data)./sum(psdest.Data);
%    Ind = find(normcumsumpsd <=0.5,1,'last');
%    fprintf('Median frequency is %2.3f Hz\n',psdest.Frequencies(Ind));
   
        
        if sizeInd1 == 0
            medianfreqs_signal(nn) = 3;
        elseif sizeInd1 ~= 0
            medianfreqs_signal(nn) = signal_F(Ind1);
        end
        
end

%    zerosfm1 = find(medianfreqs_signal == 0);
%    medianfreqs_signal(zerosfm1) = F50_signal; 
%    medianfreqs_signal1 = filtbutter(medianfreqs_signal,freq/10,freq,'low');
%   medianfreqs_signal1 = filtbutter(medianfreqs_signal,freq/10,freq,'low');
  medianfreqs_signal1 = smooth(medianfreqs_signal,0.5);
% medianfreqs_signalf = medianfreqs_signalf;
   medianfreqs_signalf = medianfreqs_signal1;
   freqmediani = medianfreqs_signal;
       
   tam_mf = length(medianfreqs_signalf);
   t_mft = [linspace(0,temp(end),tam_mf)]'; % vetor tempo freq median t
   %% GRAFICOS 6 - Frequencia Mediana na série temporal
 if graficos == 1
      figure(6)
    %   subplot(2,2,1)
      plot(t_mft,medianfreqs_signal);
      hold on
      plot(t_mft,medianfreqs_signalf,'color','r','linewidth',1);
      xlabel('Time (s)'); 
      ylabel('Hz');
      title('Median Frequency')
      axis tight
 end
%%
integral = signal_integ;
rmsmed = signalrms;
freqmedian = F50_signal;
psdpeak = PEAK_signal;
linenveli = signalenv;
rmsi = signalrms2;

end

%%

function [datf] = filtbutter(dat,fc,freq,ftype)
if nargin == 2
    freq = 100;
    ftype = 'low'; 
end

n=4; %ordem do filtro

wn=fc/(freq/2);   %frequencia de corte de

[b,a] = butter(n,wn,ftype); %definindo os parametros para o filtro de Butterworth

if size(dat,2) == 1
    dat = [[1:size(dat,1)]',dat];
end

[nlin,ncol] = size(dat);
 
datf = NaN(nlin,ncol);
for i = 2:ncol
datf(:,i) = filtfilt(b,a,dat(:,i));
end

datf = [[1:size(dat,1)]',datf(:,2:end)];

end
%%

function [y] = rms2(signal, windowlength, overlap, zeropad)
%% DECLARATIONS AND INITIALIZATIONS
% Calculates windowed (over- and non-overlapping) RMS of a signal using the specified windowlength
% y = rms(signal, windowlength, overlap, zeropad)
% signal is a 1-D vector
% windowlength is an integer length of the RMS window in samples
% overlap is the number of samples to overlap adjacent windows (enter 0 to use non-overlapping windows)
% zeropad is a flag for zero padding the end of your data...(0 for NO, 1 for YES)
% ex. y=rms(mysignal, 30, 10, 1).  Calculate RMS with window of length 30 samples, overlapped by 10 samples each, and zeropad the last window if necessary
% ex. y=rms(mysignal, 30, 0, 0).  Calculate RMS with window of length 30 samples, no overlapping samples, and do not zeropad the last window
%
% Author: A. Bolu Ajiboye


delta = windowlength - overlap;

%% CALCULATE RMS

indices = 1:delta:length(signal);
% Zeropad signal
if length(signal) - indices(end) + 1 < windowlength
    if zeropad
        signal(end+1:indices(end)+windowlength-1) = 0;
    else
        indices = indices(1:find(indices+windowlength-1 <= length(signal), 1, 'last'));
    end
end

y = zeros(1, length(indices));
% Square the samples
signal = signal.^2;

index = 0;
for i = indices
	index = index+1;
	% Average and take the square root of each window
	y(index) = sqrt(mean(signal(i:i+windowlength-1)));
end
end
%%
function varargout = psd2(varargin)
%PSD2 Power Spectral Density and frequency characteristics.
%  PSD2 estimates the power spectral density (PSD), mean power frequency (MPF),
%  peak frequency (PEAK), and limit frequency (F95) that contains up to 95%
%  of the PSD using Welch's averaged periodogram method (PSD Matlab function).
%  [MPF,PEAK,F50,F95,F,P]=psd2(X,Fs,NFFT,WINDOW,NOVERLAP,DFLAG)
%  [MPF,PEAK,F50,F95,F,P]=psd2(X,Fs) uses default values
%  Inputs:
%    X: data vector
%    Fs: sampling frequency
%    NFFT,WINDOW,NOVERLAP,DFLAG: see PSD matlab function for help
%      default values: NFFT=1024, WINDOW=256, NOVERLAP= WINDOW/2, DFLAG='mean'
%  Outputs:
%    MPF: mean power frequency
%    PEAK: peak frequency (mode)
%    F50: median frequency of the power spectral density
%    F95: frequency limit that contains up to 95% of the power spectral density
%    F: frequency vector	
%    P: PSD estimated vector
%
%   See also PSD
%  Marcos Duarte  mduarte@usp.br 11oct1998 

if nargin == 2
    x=varargin{1};
    fs=varargin{2};
    if length(x)<1000
        nfft=512;
        window=256;
    else
        nfft=1024;
        window=512;
    end
    noverlap=window/2;
    dflag='mean';
elseif nargin==6
    x=varargin{1};
    fs=varargin{2};
    nfft=varargin{3};
    window=varargin{4};
    noverlap=varargin{5};
    dflag=varargin{6};
else
    error('Incorrect number of inputs')
    return
end
%power spectral density:
% [p,f]=psd(x,nfft,fs,window,noverlap,dflag); % OBSOLETE in Matlab 2018
[p,f]=pwelch(x, hanning(nfft), noverlap, 1024, fs);
%mpf:
mpf=trapz(f,f.*p)/trapz(f,p);
%peak:
[m,peak]=max(p);
peak=f(peak);
%50 and 95% of PSD:
area=cumtrapz(f,p);
f50=find(area >= .50*area(end));
if ~isempty(f50)
    f50=f(f50(1));
else
    f50=0;
end
f95=find(area >= .95*area(end));
if ~isempty(f95)
    f95=f(f95(1));
else
    f95=0;
end
varargout{1}=mpf;
varargout{2}=peak;
varargout{3}=f50;
varargout{4}=f95;
varargout{5}=f;
varargout{6}=p;
end