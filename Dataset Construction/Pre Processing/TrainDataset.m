clc; clear; close all;
%%
%% PATH 
folder_path = "D:\JHU\Audio-project\trainv2"
fileList = dir(fullfile(folder_path,'*.mat'));
%%
fprintf('Total MAT files found: %d\n', numel(fileList));
%%
%%  PHONEME GROUPS 
LabelGroups.fricatives   = {'f','s','z','v'};
LabelGroups.Vowels       = {'a','o','u','i','e'};
LabelGroups.plosives     = {'p','b','t','d','k','g'};
LabelGroups.nasals       = {'m','n'};
LabelGroups.liquids      = {'l','r'};
LabelGroups.misc         = {'c','h','j','q','w','x','y'};
%%
groupNames = fieldnames(LabelGroups);
disp('Phoneme Groups Loaded:');
disp(groupNames');
%%
%% FINAL TABLE 
FinalTable = table();
%%
rowCounter = 0;   % global row counter
%%
%%  MAIN LOOP 
for f = 1:numel(fileList)

    matpath = fullfile(folder_path, fileList(f).name);
    fprintf('\nLoading file: %s\n', fileList(f).name);

    try
        S = load(matpath);
        Dataset = S.Dataset;
    catch
        warning('Skipping file (cannot load Dataset): %s', fileList(f).name);
        continue;
    end

    numWords = numel(Dataset);
    fprintf('  Words in file: %d\n', numWords);

    for k = 1:numWords

        rowCounter = rowCounter + 1;
        row = struct();

        %% WORD 
        wordStr = char(Dataset(k).word);
        row.word = string(wordStr);

        %%  SPLIT LETTERS (CRITICAL FIX) 
        letters = cellstr(lower(wordStr)');   % {'c','a','r'}

        %% EMG FEATURES 
        numCh = 2;   % fixed channels

        for ch = 1:numCh
            row.(['EMG_ch' num2str(ch) '_signal'])        = {Dataset(k).EMG.signal(ch,:)};
            row.(['EMG_ch' num2str(ch) '_FFT'])           = {Dataset(k).EMG.channel(ch).FFT.magnitude};
            row.(['EMG_ch' num2str(ch) '_CWT'])           = {Dataset(k).EMG.channel(ch).CWT.coeff};
            row.(['EMG_ch' num2str(ch) '_Spectrogram'])   = {Dataset(k).EMG.channel(ch).spectrogram.S};
        end

        %%  PHONEME GROUP LABELING
        row.fricatives  = getGroupLabel(letters, LabelGroups.fricatives);
        row.Vowels      = getGroupLabel(letters, LabelGroups.Vowels);
        row.plosives    = getGroupLabel(letters, LabelGroups.plosives);
        row.nasals      = getGroupLabel(letters, LabelGroups.nasals);
        row.liquids     = getGroupLabel(letters, LabelGroups.liquids);
        row.misc        = getGroupLabel(letters, LabelGroups.misc);

        %% APPEND 
        FinalTable = [FinalTable; struct2table(row)];

        %% DEBUG OUTPUT (FIRST 5 ROWS ONLY)
        if rowCounter <= 5
            fprintf('\n--- DEBUG ROW %d ---\n', rowCounter);
            disp(FinalTable(end, {'word','fricatives','Vowels','plosives','nasals','liquids','misc'}));

            fprintf('Signal sizes:\n');
            fprintf('  EMG ch1 signal: %s\n', mat2str(size(row.EMG_ch1_signal{1})));
            fprintf('  EMG ch1 FFT   : %s\n', mat2str(size(row.EMG_ch1_FFT{1})));
            fprintf('  EMG ch1 CWT   : %s\n', mat2str(size(row.EMG_ch1_CWT{1})));
            fprintf('  EMG ch1 Spec  : %s\n', mat2str(size(row.EMG_ch1_Spectrogram{1})));
        end

    end
end
%%
%% SAVE 
save('trainv2.mat','FinalTable', '-v7.3');
%%
disp('Dataset creation completed successfully');
disp('FinalTable size:');
disp(size(FinalTable));
%%
openvar('FinalTable');