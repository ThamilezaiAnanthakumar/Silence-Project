function [filteredSignal] = FilteredSignals(fileList, data, numChannels, sampleRate, low_cutoff, high_cutoff)
%FilteredSignals
%  Applies a band-pass Butterworth filter for each channel.
%
% Inputs:
%   fileList     - struct from dir with file names
%   data         - cell array of channel data
%   numChannels  - cell array with number of channels
%   sampleRate   - cell array of sampling rates
%   low_cutoff   - lower cutoff frequency in Hz 
%   high_cutoff  - upper cutoff frequency in Hz 
%
% Outputs:
%   filteredSignal - cell array containing filtered signals

    filteredSignal = cell(length(fileList),1);

    for file = 1:length(fileList)
  
        fs = sampleRate{file};  % Sampling rate for this file

        % Normalized cutoff for Nyquist frequency
        Wn = [low_cutoff high_cutoff] / (fs/2);

        % Design 2nd-order Butterworth band-pass filter
        [b, a] = butter(2, Wn, 'bandpass');

        for ch = 1:numChannels{file}
            % Apply zero-phase filtering
            filteredSignal{file}(ch,:) = filtfilt(b, a, data{file}(ch,:));
       
        end
    end
end
