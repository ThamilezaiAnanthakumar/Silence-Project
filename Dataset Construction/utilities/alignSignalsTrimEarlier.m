function [emg_data_aligned, Audio_data_aligned, time_EMG, time_Audio] = alignSignalsTrimEarlier(emg_data, sampleRate_emg, timestamp_emg,time_emg, Audio_data, sampleRate_Audio, recTimeAll)
% ALIGNSIGNALSTRIMEARLIER Align EMG and Audio signals by trimming the earlier-starting signal
%
% Inputs:
%   emg_data         - EMG signal (channels x samples)
%   sampleRate_emg   - EMG sampling rate (Hz)
%   fileTime         - EMG timestamp (datetime)
%   Audio_data       - Audio signal (channels x samples)
%   sampleRate_Audio - Audio sampling rate (Hz)
%   recTimeAll       - Audio timestamp (datetime)
%
% Outputs:
%   emg_data_aligned   - Aligned EMG data
%   Audio_data_aligned - Aligned Audio data
%   time_EMG           - Time axis for EMG (s)
%   time_Audio         - Time axis for Audio (s)

%% Extract hours, minutes, seconds
hAudio = hour(recTimeAll); mAudio = minute(recTimeAll); sAudio = second(recTimeAll);
hEMG   = hour(timestamp_emg);   mEMG   = minute(timestamp_emg);   sEMG   = second(timestamp_emg);

%% Handle 12-hour vs 24-hour ambiguity
if hAudio < 12 && hEMG >= 12
    hAudio = hAudio + 12;   % assume both PM
elseif hAudio >= 12 && hEMG < 12
    hEMG = hEMG + 12;
end

%% Convert to seconds since midnight
secAudio = hAudio*3600 + mAudio*60 + sAudio;
secEMG   = hEMG*3600 + mEMG*60 + sEMG;

%% Compute offset in seconds
offsetSeconds = secEMG - secAudio;   % positive = EMG starts later
disp(['Offset in seconds: ', num2str(offsetSeconds)])

%% Determine which signal is earlier
if offsetSeconds > 0
    earlierSignal = 'Audio';
elseif offsetSeconds < 0
    earlierSignal = 'EMG';
    offsetSeconds = abs(offsetSeconds);  % make positive for trimming
else
    earlierSignal = 'Same';
end

disp(['Earlier signal: ', earlierSignal])

%% Trim the earlier-starting signal
if strcmp(earlierSignal, 'Audio')
    numTrim = round(offsetSeconds * sampleRate_Audio);
    fprintf('numTrim : %d\n', numTrim);
    Audio_data_aligned = Audio_data(numTrim+1:end, :);
    emg_data_aligned   = emg_data;  % EMG intact
    time_EMG = time_emg;
    
elseif strcmp(earlierSignal, 'EMG')
    numTrim = round(offsetSeconds * sampleRate_emg);
    fprintf('numTrim : %d\n', numTrim);
    emg_data_aligned   = emg_data(:, numTrim+1:end);
    time_EMG = time_emg(:,numTrim+1:end);
    Audio_data_aligned = Audio_data;  % Audio intact

else
    % Both start at same time
    emg_data_aligned   = emg_data;
    Audio_data_aligned = Audio_data;
end

%% Generate time axes for aligned signals
%time_EMG   = (0:size(emg_data_aligned,2)-1) / sampleRate_emg;
time_Audio = (0:size(Audio_data_aligned,1)-1) / sampleRate_Audio;

end
