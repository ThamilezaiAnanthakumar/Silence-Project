%% processing.mlx
% Created by   : Thamilezai Ananthakumar
% Date Created : 2025-12-26
% Last Modified: 2026-01-14
% Version      : v2.0
%
% Description:
% Two neuro muscular signal channels are processed using a band-pass filter to remove motion artifacts and unwanted noise, while the audio signal is kept unfiltered to preserve its original characteristics.
% The neuro muscular signal and audio recordings are temporally synchronized by calculating the time difference between the start of the audio file and the EMG acquisition, allowing both signals to be aligned on a common time axis. 
% After synchronization, the neuro muscular signal and audio signals are segmented into word-level windows based on predefined time intervals. 
% For each word segment, a structured MATLAB dataset is created that contains the audio waveform and its spectrogram, along with the neuro muscular signal waveforms, neuro muscular signal frequency-domain representation using FFT, neuro muscular signal spectrogram, and neuro muscular signal continuous wavelet transform, enabling comprehensive time–frequency and time–scale analysis for each word.

%%
clc;
close all;
%%
folder_path = "F:\JHU\Audio-project\New folder\4_04" 
%%
[numChannels, data, time, sampleRate, unpackedFile, fileList, timestamp] = loadFiles(folder_path)
%%
cellfun(@class, sampleRate, 'UniformOutput', false);
%%
for i = 1:numel(sampleRate)
    sampleRate{i} = sampleRate{i} / 2;
end
%%
for i = 1:length(time)

    time{i} = time{i}(1:end-6);     % remove last 6 elements in the time axis last 6 are zeroes so that zeros removed and also last 6 values of data file also removed
end
%%
for i = 1:length(data)
    data{i} = data{i}(:, 1:end-6);  % remove last 6 columns 
end

%%
sampleRate 
%%
low_cutoff = 0.1;
high_cutoff = 1000; 
%%
[filteredSignal] = FilteredSignals(fileList, data, numChannels, sampleRate, low_cutoff, high_cutoff) % filtering emg signal
%%
windowlength = 30;
%%
plotSignals_Windowed(fileList, time, filteredSignal, numChannels, windowlength)% adjustable windowed scrollable emg signal
%%
[data_Audio, sampleRate_Audio, ~ , fileList_Audio, recTimeAll] = loadFilesAudio(folder_path) % Loading audio files
%%
time_emg = time{1}
Audiotimestamp = recTimeAll{1}
timestamp_emg = timestamp{1}
sampleRate_emg = sampleRate{1}
emg_data = filteredSignal{1}
Audio_data = data_Audio{1}               
sampleRate_Audio   = sampleRate_Audio{1}
%%
scrollableAudioViewer(Audio_data, sampleRate_Audio, 10); % scrollable audio signal to see the time interval of each words
%%
[emg_data_aligned, Audio_data_aligned, time_EMG, time_Audio] = alignSignalsTrimEarlier(emg_data, sampleRate_emg, timestamp_emg, time_emg, Audio_data, sampleRate_Audio, Audiotimestamp)%synchronizing the emg and audio signal
%%
scrollableAudioViewer(Audio_data_aligned, sampleRate_Audio, 10);%scrollable audio signal to see the time interval of each words after synchronization
%%
figure
plot(time_EMG(1,:), emg_data_aligned(1,:))
title('Aligned EMG')
xlabel('Time (s)')
%%
numSamples = size(emg_data_aligned, 2);   % samples from neuro muscular signalS data
sampleIdx  = 1:numSamples;               % sample indices

figure
plot(time_EMG(1, :), sampleIdx)
title('Time vs Sample Index (from EMG)')
xlabel('Time (s)')
ylabel('Sample Index')
grid on
%%
dt = diff(time_EMG(1,:));

figure
plot(dt)
title('Inter-sample Time Differences')
xlabel('Sample')
ylabel('\Delta t (s)')
grid on
%%
words =  { ...
    'father','father','father','father','father', ...
    'cat','cat','cat','cat','cat', ...
    'appel','appel','appel','appel','appel', ...
    'car','car','car','car','car', ...
    'map','map','map','map','map', ...
    'last','last','last','last','last', ...
    'back','back','back','back','back', ...
    'stop','stop','stop','stop','stop', ...
    'fast','fast','fast','fast','fast', ...
    'glass','glass','glass','glass','glass', ...
    'bad','bad','bad','bad','bad', ...
    'at','at','at','at','at', ...
    'path','path','path','path','path', ...
    'grass','grass','grass','grass','grass', ...
    'ant','ant','ant','ant','ant', ...
    'flat','flat','flat','flat','flat', ...
    'lack','lack','lack','lack','lack', ...
    'drag','drag','drag','drag','drag', ...
    'ask','ask','ask','ask','ask' ...
};
%%
timeinterval = [ ...
 1 2.5 4 5.5 7 9 ...
 10.5 12 13 14.5 16 ...
 17.5 19 20 21.5 23 ...
 25 26 27.5 28.5 30 ...
 31.5 33 34 35 37 ...
 38.4 39.5 40.5 41.5 43 ...
 45 46 47.5 48.5 50 ...
 52 53 54 55.5 57 ...
 58.5 59.5 60.8 62 63.5 ...
 65.3 66.5 67.5 68.5 70 ...
 72 73 74 75 76 ...
 77.8 79 80 81 82 ...
 83.5 84.5 85.5 87 88 ...
 89.5 90.7 92 93 94 ...
 95.5 97 98 99 100.5 ...
 101.8 103 104 105 106.5 ...
 108.5 109.5 110.5 111.5 113 ...
 114.5 116 117 118 119.5 ...
 121 122 123 124.5 125.5 ...
 127 128 129 130.3 132 ...
];
%%
Dataset = buildWordAlignedAudioEMGDataset( words, timeinterval, Audio_data_aligned, time_Audio, sampleRate_Audio, emg_data_aligned, time_EMG, sampleRate_emg)% building the struct containing all needed analysis
%%
fieldnames(Dataset)
%%
plotAudioEMGFromDataset_SeparateFigures(Dataset, 1, [1 2])% visualizing the spectrogram, CWT, FFT, wave form of audio and neuro muscular signal for both channel for specific word , second input variable is word indices in word array Eg : here its see
%%
save('4_04.mat', 'Dataset', '-v7.3'); % save as .mat files