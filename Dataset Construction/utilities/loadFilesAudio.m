function [data, sampleRate, time, fileList, recTimeAll] = loadFilesAudio(folderPath)
% loadFilesAudio - Load .m4a audio files and return data, sample rate, and time vector
%
% Inputs:
%   folderPath - folder containing .m4a files
%
% Outputs:
%   data        - cell array of audio data
%   sampleRate  - cell array of sampling rates
%   time        - cell array of time vectors (seconds)
%   fileList    - list of files
%   recTimeAll  - cell array of recording start datetime

    % Check folder exists
    if ~isfolder(folderPath)
        error('Folder does not exist: %s', folderPath);
    end

    % Get .m4a files
    fileList = dir(fullfile(folderPath, '*.m4a'));
    numFiles = length(fileList);

    % Preallocate
    data = cell(numFiles,1);
    sampleRate = cell(numFiles,1);
    time = cell(numFiles,1);
    recTimeAll = cell(numFiles,1);

    for i = 1:numFiles
        filePath = fullfile(folderPath, fileList(i).name);
        fprintf('Processing file %d of %d: %s\n', i, numFiles, fileList(i).name);

        % 1. Get recording start time
        recTime = getM4ARecordingTime_(filePath);
        recTimeAll{i} = recTime;

        % 2. Read audio
        [audioData, fs] = audioread(filePath);

        % 3. Store outputs
        data{i} = audioData;
        sampleRate{i} = fs;

        % 4. Time vector in seconds (starting from 0)
        nSamples = length(audioData);
        time{i} = (0:nSamples-1)/fs;
    end
end
