function Dataset = buildWordAlignedAudioEMGDataset( ...
                    words, timeinterval, ...
                    Audio_data_aligned, time_Audio, sampleRate_Audio, ...
                    emg_data_aligned, time_EMG, sampleRate_emg)
% BUILDWORDALIGNEDAUDIOEMGDATASET

% Builds a word-wise dataset containing aligned Audio and neuro muscular signal
% segments with time-frequency analyses.
%
% INPUTS:
%   words               - Cell array of spoken words (1×N)
%   timeinterval        - Word boundary times in seconds (1×(N+1))
%
%   Audio_data_aligned  - Aligned audio signal (samples × channels)
%   time_Audio          - Audio time vector (seconds)
%   sampleRate_Audio    - Audio sampling rate (Hz)
%
%   emg_data_aligned    - Aligned EMG signal (channels × samples)
%   time_EMG            - neuro muscular signal time vector (seconds)
%   sampleRate_emg      - neuro muscular signal sampling rate (Hz)
%
% OUTPUT:
%   Dataset             - Struct array with word-wise analysis:
%                         Audio waveform + spectrogram
%                         neuro muscular signal waveform + spectrogram + FFT + CWT
%
% EXAMPLE:
%   Dataset = buildWordAlignedAudioEMGDataset( ...
%              words, timeinterval, ...
%              Audio_data_aligned, time_Audio, FsAudio, ...
%              neuro muscular signal_data_aligned, time_neuro muscular signal, Fsneuro muscular signal);
%


numWords = numel(words);
Dataset = struct();

for k = 1:numWords

    %%  TIME BOUNDARIES 
    tStart = timeinterval(k);
    tEnd   = timeinterval(k+1);

    %% AUDIO SEGMENT 
    idxAudio = time_Audio >= tStart & time_Audio < tEnd;
    audioSeg = Audio_data_aligned(idxAudio, :);
    audioTime = time_Audio(idxAudio) - tStart;

    %% EMG SEGMENT 
    idxEMG = time_EMG >= tStart & time_EMG < tEnd;
    emgSeg = emg_data_aligned(:, idxEMG);
    emgTime = time_EMG(idxEMG) - tStart;

    %% AUDIO ANALYSIS 
    winA     = round(0.020 * sampleRate_Audio);   
    overlapA = round(0.010 * sampleRate_Audio);   
    nfftA    = 2^nextpow2(winA);

    [Sa, Fa, Ta] = spectrogram(audioSeg(:,1), ...
                              winA, overlapA, nfftA, sampleRate_Audio);

    %% STORE AUDIO 
    Dataset(k).word   = words{k};
    Dataset(k).tStart = tStart;
    Dataset(k).tEnd   = tEnd;

    Dataset(k).Audio.time   = audioTime;
    Dataset(k).Audio.signal = audioSeg;

    Dataset(k).Audio.spectrogram.S = Sa;
    Dataset(k).Audio.spectrogram.F = Fa;
    Dataset(k).Audio.spectrogram.T = Ta;

    %% EMG ANALYSIS 
    numch = size(emgSeg,1);

    Dataset(k).EMG.time   = emgTime;
    Dataset(k).EMG.signal = emgSeg;

    for ch = 1:numch

        emgCh = emgSeg(ch,:);

        % EMG Spectrogram 
        winE     = round(0.050 * sampleRate_emg);   
        overlapE = round(0.025 * sampleRate_emg);
        nfftE    = 2^nextpow2(winE);

        [Se, Fe, Te] = spectrogram(emgCh, ...
                                   winE, overlapE, nfftE, sampleRate_emg);

        % EMG FFT 
        N = length(emgCh);
        EMG_FFT = abs(fft(emgCh));
        fFFT = (0:N-1) * sampleRate_emg / N;

        % EMG CWT 
        [cwtCoeff, fCWT] = cwt(emgCh, sampleRate_emg);

        % STORE PER CHANNEL 
        Dataset(k).EMG.channel(ch).spectrogram.S = Se;
        Dataset(k).EMG.channel(ch).spectrogram.F = Fe;
        Dataset(k).EMG.channel(ch).spectrogram.T = Te;

        Dataset(k).EMG.channel(ch).FFT.magnitude = EMG_FFT;
        Dataset(k).EMG.channel(ch).FFT.frequency = fFFT;

        Dataset(k).EMG.channel(ch).CWT.coeff = cwtCoeff;
        Dataset(k).EMG.channel(ch).CWT.frequency = fCWT;

    end
end
end
