function [numChannels, data, time, sampleRate, unpackedFile, fileList, timestamp] = loadFiles(folderPath)
% loadFiles
%   Loads and unpacks .bin files from a folder using fn_BionodeBinOpen_1.
%
% Inputs:
%   folderPath - Path to the folder containing .bin files
%
% Outputs:
%   data         - Cell array of channel data for each file
%   time         - Cell array of time vectors for each file
%   sampleRate   - Cell array of sampling rates for each file
%   unpackedFile - Cell array of unpacked file structs

    % Check if folder exists
    if ~isfolder(folderPath)
        error('Folder does not exist: %s', folderPath);
    end

    % Get a list of all .bin files
    fileList = dir(fullfile(folderPath, '*.bin'));
    numFiles = length(fileList);

    % Preallocate cell arrays
    data = cell(numFiles,1);
    time = cell(numFiles,1);
    sampleRate = cell(numFiles,1);
    unpackedFile = cell(numFiles,1);
    numChannels=cell(numFiles,1);
    timestamp =cell(numFiles,1);

    % Loop over each file
    for i = 1:numFiles
        filePath = fullfile(folderPath, fileList(i).name);

        % Display progress
        fprintf('Processing file %d of %d: %s\n', i, numFiles, fileList(i).name);

        % Call unpack function
        unpackedFile{i} = fn_BionodeBinOpen_1(filePath, 12);  % ADC Resolution=12

        % Extract data
        data{i} = unpackedFile{i}.channelsData;
        time{i} = unpackedFile{i}.time;
        sampleRate{i} = unpackedFile{i}.sampleRate;
        numChannels{i} = unpackedFile{i}.numChannels;
        timestamp{i}  = unpackedFile{i}.Date;
    end
    
end