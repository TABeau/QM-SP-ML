% Code for generating the Time-Frequency-Like representation. See comments
% to create directories and set the path to it. Also uncommnent the
% Time-Frequency Representation you want to compute

clear
[data,text] = xlsread('CES_QM9_BohrRadii.xlsx'); % Load the Coulomb Eigen Spectrum of the molecules in QM9 dataset
[N,M] = size(data);
X = data(:,:); X = normalize(X); % Normalized them.

% create a directory where to same your time-frequency representation and
% set the path with fname. different directory for each time-frequency
% transform
fname = 'D:\Matlab\QM_ML\SignalProcessingQM\Python_CNN_Regre_Spect\QM9_Original\QM9_EXCEL\QM9_EXCEL_bis\Spectrogram_QM9_WVD_EI_bis';

%% Time-Frequency Transform
for n = 1:N
    %Select one Time-Frequency Transform
    
    %cwt(X(n,:),'amor');  % Continous wavelet transform with (CWT)
    %spectrogram(X(n,:)); % Spectrogram (STFT)
    wvd(X(n,:));         % Wigner-Ville Distributon
    
    colorbar('off');
    set(gca,'visible','off');
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'LooseInset',get(gca,'TightInset'));
    F = getframe(figure(1)); img = F.cdata;
    imwrite(img, fullfile(fname,strcat(strcat('gdb_',int2str(n)),'.png')));
end