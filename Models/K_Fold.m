%% INITIAL SETUP AND DATA LOADING
clc; clear; close all;
%%
rng(42, 'twister');
fprintf('========================================\n');
fprintf('K-FOLD CROSS-VALIDATION FOR VOWEL A\n');
fprintf('========================================\n\n');
%%
% Configuration
FREQ_MIN = 0.1;
FREQ_MAX = 450;
SAMPLE_RATE = 12500;
K_FOLDS = 5;  % Number of folds
%%
% Load training data only
fprintf('=== Loading Training Data ===\n');
tic;
trainData = load('trainv2.mat');
FinalTableTrain = trainData.FinalTable;
clear trainData;
fprintf(' Loaded in %.1f sec\n', toc);
fprintf('  Total rows: %d\n\n', height(FinalTableTrain));
%% PROCESS LABELS
fprintf('=== Processing Labels ===\n');

% Normalize labels
v = upper(strtrim(FinalTableTrain.Vowels));
v(v == "" | v == "N" | v == "NONE") = "";
FinalTableTrain.Vowels = v;

% Add multi-hot encoding
FinalTableTrain = addVowelsMultiHot(FinalTableTrain);

% Extract Vowel A labels
Ytrain_A = double(FinalTableTrain.isA);

fprintf('Label distribution:\n');
fprintf('  Present (1): %d (%.1f%%)\n', sum(Ytrain_A==1), 100*sum(Ytrain_A==1)/length(Ytrain_A));
fprintf('  Absent (0):  %d (%.1f%%)\n\n', sum(Ytrain_A==0), 100*sum(Ytrain_A==0)/length(Ytrain_A));
%% FILTER WORDS TO EXACTLY 5 REPETITIONS PER WORD
fprintf('=== Filtering to 5 Repetitions per Word ===\n');

% Check if 'Word' column exists
if ~ismember('word', FinalTableTrain.Properties.VariableNames)
    error('Column "Word" not found in table. Available columns: %s', ...
          strjoin(FinalTableTrain.Properties.VariableNames, ', '));
end

% Get unique words and their counts
uniqueWords = unique(FinalTableTrain.word);
fprintf('Total unique words: %d\n', length(uniqueWords));

% Identify rows to keep (5 per word)
rowsToKeep = [];

for i = 1:length(uniqueWords)
    word = uniqueWords{i};
    wordIdx = find(strcmp(FinalTableTrain.word, word));
    
    numReps = length(wordIdx);
    
    if numReps >= 5
        % Keep first 5 occurrences
        rowsToKeep = [rowsToKeep; wordIdx(1:5)];
    end
end

% Sort to maintain order
rowsToKeep = sort(rowsToKeep);

% Filter the table
FinalTableTrain = FinalTableTrain(rowsToKeep, :);
Ytrain_A = Ytrain_A(rowsToKeep);

fprintf('\n Filtered dataset:\n');
fprintf('  Rows after filtering: %d\n', height(FinalTableTrain));
fprintf('  Expected rows (words × 5): %d\n\n', length(uniqueWords) * 5);
%% EXTRACT FEATURES FOR ALL DATA
fprintf('=== Extracting Features ===\n');
fprintf('This will take several minutes...\n\n');

N = height(FinalTableTrain);
X_all = cell(N, 1);

tic;
for i = 1:N
    X_all{i} = extractLSTMFeatures(FinalTableTrain, i, FREQ_MIN, FREQ_MAX, SAMPLE_RATE);
    
    if mod(i, 100) == 0 || i == N
        elapsed = toc;
        pct = 100 * i / N;
        eta = elapsed * (N - i) / i;
        fprintf('  [%6d/%6d] %.1f%% | Elapsed: %.1fm | ETA: %.1fm\n', ...
            i, N, pct, elapsed/60, eta/60);
    end
end

totalTime = toc;
fprintf('\n✓ Feature extraction complete (%.1f min)\n', totalTime/60);

[numFeatures, seqLength] = size(X_all{1});
fprintf('  Feature shape: %d features × %d timesteps\n\n', numFeatures, seqLength);
%% CREATE WORD-BASED K-FOLD SPLITS (NO DATA LEAKAGE)
fprintf('=== Creating %d-Fold Word-Based Splits ===\n', K_FOLDS);

uniqueWords = unique(FinalTableTrain.word, 'stable');
numWords = length(uniqueWords);

fprintf('Total words: %d\n', numWords);
fprintf('Samples per word: 5\n');
fprintf('Words per fold: ~%d\n\n', floor(numWords / K_FOLDS));

% Shuffle words randomly
shuffledWords = uniqueWords(randperm(numWords));

% Assign words to folds
wordsPerFold = floor(numWords / K_FOLDS);
foldAssignments = cell(K_FOLDS, 1);

for k = 1:K_FOLDS
    if k < K_FOLDS
        foldAssignments{k} = shuffledWords((k-1)*wordsPerFold + 1 : k*wordsPerFold);
    else
        % Last fold gets remaining words
        foldAssignments{k} = shuffledWords((k-1)*wordsPerFold + 1 : end);
    end
    fprintf('Fold %d: %d words assigned\n', k, length(foldAssignments{k}));
end

% Create index mapping for each fold
foldIndices = cell(K_FOLDS, 1);

for k = 1:K_FOLDS
    foldWords = foldAssignments{k};
    foldIdx = [];
    
    for w = 1:length(foldWords)
        wordRows = find(strcmp(FinalTableTrain.word, foldWords{w}));
        foldIdx = [foldIdx; wordRows];
    end
    
    foldIndices{k} = sort(foldIdx);
    fprintf('Fold %d: %d samples (rows)\n', k, length(foldIndices{k}));
end

fprintf('\n K-Fold splits created (word-based, no leakage)\n\n');
%% INITIALIZE STORAGE FOR RESULTS
fprintf('=== Initializing Result Storage ===\n');

% Metrics storage
metrics = struct();
metrics.train_acc = zeros(K_FOLDS, 1);
metrics.train_prec = zeros(K_FOLDS, 1);
metrics.train_rec = zeros(K_FOLDS, 1);
metrics.train_f1 = zeros(K_FOLDS, 1);
metrics.train_spec = zeros(K_FOLDS, 1);
metrics.train_sens = zeros(K_FOLDS, 1);

metrics.val_acc = zeros(K_FOLDS, 1);
metrics.val_prec = zeros(K_FOLDS, 1);
metrics.val_rec = zeros(K_FOLDS, 1);
metrics.val_f1 = zeros(K_FOLDS, 1);
metrics.val_spec = zeros(K_FOLDS, 1);
metrics.val_sens = zeros(K_FOLDS, 1);

% ROC storage (for aggregation)
roc_data = struct();
roc_data.train_scores = cell(K_FOLDS, 1);
roc_data.train_labels = cell(K_FOLDS, 1);
roc_data.val_scores = cell(K_FOLDS, 1);
roc_data.val_labels = cell(K_FOLDS, 1);

% Confusion matrix storage
val_confusion_total = zeros(2, 2);  % Aggregate across folds

fprintf(' Storage initialized for %d folds\n\n', K_FOLDS);
%% SETUP VISUALIZATION
% Create a single figure to hold all 5 confusion matrices
hFigCM = figure('Name', 'Confusion Matrices by Fold', 'Position', [100, 100, 1200, 700], 'Color', 'w');
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, 'Confusion Matrices per Fold (Validation Set)', 'FontSize', 14, 'FontWeight', 'bold');
%%
%% K-FOLD TRAINING LOOP
fprintf('========================================\n');
fprintf('STARTING %d-FOLD CROSS-VALIDATION\n', K_FOLDS);
fprintf('========================================\n\n');

for fold = 1:K_FOLDS
    fprintf('\n');
    fprintf('╔══════════════════════════════════════╗\n');
    fprintf('║         FOLD %d / %d                   ║\n', fold, K_FOLDS);
    fprintf('╚══════════════════════════════════════╝\n\n');
    
    %% SPLIT DATA
    fprintf('--- Splitting Data ---\n');
    
    % Validation indices for this fold
    valIdx = foldIndices{fold};
    
    % Training indices (all other folds)
    trainIdx = [];
    for k = 1:K_FOLDS
        if k ~= fold
            trainIdx = [trainIdx; foldIndices{k}];
        end
    end
    
    % Split features and labels
    Xtrain = X_all(trainIdx);
    Ytrain = Ytrain_A(trainIdx);
    
    Xval = X_all(valIdx);
    Yval = Ytrain_A(valIdx);
    
    fprintf('  Training samples: %d\n', length(Ytrain));
    fprintf('  Validation samples: %d\n', length(Yval));
    
    % Check class distribution
    fprintf('  Train - Absent: %d, Present: %d\n', sum(Ytrain==0), sum(Ytrain==1));
    fprintf('  Val   - Absent: %d, Present: %d\n', sum(Yval==0), sum(Yval==1));
    
    %% DATA AUGMENTATION
    fprintf('\n--- Augmenting Training Data ---\n');
    
    idx0 = find(Ytrain == 0);
    idx1 = find(Ytrain == 1);
    
    if length(idx0) < length(idx1)
        minority_idx = idx0;
        minority_label = 0;
        targetN = length(idx1);
    else
        minority_idx = idx1;
        minority_label = 1;
        targetN = length(idx0);
    end
    
    numToGenerate = targetN - length(minority_idx);
    fprintf('  Generating %d augmented samples...\n', numToGenerate);
    
    noise_std = 0.05;
    augmented_features = cell(numToGenerate, 1);
    augmented_labels = zeros(numToGenerate, 1);
    
    for i = 1:numToGenerate
        source_idx = minority_idx(randi(length(minority_idx)));
        original = Xtrain{source_idx};
        noise = noise_std * randn(size(original), 'single');
        augmented_features{i} = original + noise;
        augmented_labels(i) = minority_label;
    end
    
    % Combine original + augmented
    XtrainAug = [Xtrain; augmented_features];
    YtrainAug_numeric = [Ytrain; augmented_labels];
    
    fprintf('  Balanced - Absent: %d, Present: %d\n', ...
        sum(YtrainAug_numeric==0), sum(YtrainAug_numeric==1));
    
    %% NORMALIZATION (Standard Scaler using TRAINING data only)
    fprintf('\n--- Normalizing Features ---\n');
    
    allTrainData = horzcat(XtrainAug{:});
    mu = mean(allTrainData, 2);
    sigma = std(allTrainData, 0, 2) + eps;
    
    % Apply to training
    for i = 1:length(XtrainAug)
        XtrainAug{i} = (XtrainAug{i} - mu) ./ sigma;
    end
    
    % Apply SAME normalization to validation
    for i = 1:length(Xval)
        Xval{i} = (Xval{i} - mu) ./ sigma;
    end
    
    fprintf(' Normalized using training statistics\n');
    
    % Convert to categorical
    YtrainAug_cat = categorical(YtrainAug_numeric, [0, 1], {'Absent', 'Present'});
    Yval_cat = categorical(Yval, [0, 1], {'Absent', 'Present'});
    
    %% BUILD MODEL
    fprintf('\n--- Building LSTM Model ---\n');
    
    layers = [
        sequenceInputLayer(numFeatures, 'Name', 'input')
        
        bilstmLayer(64, 'OutputMode', 'last', 'Name', 'bilstm1')
        batchNormalizationLayer('Name', 'bn1')
        dropoutLayer(0.5, 'Name', 'dropout1')
        
        fullyConnectedLayer(2, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 8, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {Xval, Yval_cat}, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', 15, ...
        'ExecutionEnvironment', 'auto', ...
        'Plots', 'none', ...
        'Verbose', false, ...
        'GradientThreshold', 1, ...
        'SequenceLength', 'longest', ...
        'SequencePaddingDirection', 'right');
    
    fprintf('   Model architecture ready\n');
    
    %% TRAIN
    fprintf('\n--- Training (Fold %d) ---\n', fold);
    tic;
    net = trainNetwork(XtrainAug, YtrainAug_cat, layers, options);
    trainTime = toc;
    fprintf('   Training complete (%.1f min)\n', trainTime/60);
    
    %% EVALUATE TRAINING SET
    fprintf('\n--- Evaluating Training Set ---\n');
    
    [YPred_train, scores_train] = classify(net, XtrainAug);
    
    Ytrue_train = double(YtrainAug_cat == "Present");
    YPred_train_num = double(YPred_train == "Present");
    
    TP_tr = sum((YPred_train_num == 1) & (Ytrue_train == 1));
    FP_tr = sum((YPred_train_num == 1) & (Ytrue_train == 0));
    FN_tr = sum((YPred_train_num == 0) & (Ytrue_train == 1));
    TN_tr = sum((YPred_train_num == 0) & (Ytrue_train == 0));
    
    metrics.train_acc(fold) = (TP_tr + TN_tr) / length(Ytrue_train);
    metrics.train_prec(fold) = TP_tr / (TP_tr + FP_tr + eps);
    metrics.train_rec(fold) = TP_tr / (TP_tr + FN_tr + eps);
    metrics.train_sens(fold) = metrics.train_rec(fold);  % Sensitivity = Recall
    metrics.train_spec(fold) = TN_tr / (TN_tr + FP_tr + eps);
    metrics.train_f1(fold) = 2 * metrics.train_prec(fold) * metrics.train_rec(fold) / ...
                             (metrics.train_prec(fold) + metrics.train_rec(fold) + eps);
    
    fprintf('  Accuracy:    %.2f%%\n', metrics.train_acc(fold)*100);
    fprintf('  Precision:   %.2f%%\n', metrics.train_prec(fold)*100);
    fprintf('  Recall:      %.2f%%\n', metrics.train_rec(fold)*100);
    fprintf('  Sensitivity: %.2f%%\n', metrics.train_sens(fold)*100);
    fprintf('  Specificity: %.2f%%\n', metrics.train_spec(fold)*100);
    fprintf('  F1-Score:    %.2f%%\n', metrics.train_f1(fold)*100);
    
    % Store for ROC
    roc_data.train_scores{fold} = scores_train(:, 2);  % "Present" class probability
    roc_data.train_labels{fold} = YtrainAug_cat;
    
    %% EVALUATE VALIDATION SET
    fprintf('\n--- Evaluating Validation Set ---\n');
    
    [YPred_val, scores_val] = classify(net, Xval);
    
    Ytrue_val = double(Yval_cat == "Present");
    YPred_val_num = double(YPred_val == "Present");
    
    TP_val = sum((YPred_val_num == 1) & (Ytrue_val == 1));
    FP_val = sum((YPred_val_num == 1) & (Ytrue_val == 0));
    FN_val = sum((YPred_val_num == 0) & (Ytrue_val == 1));
    TN_val = sum((YPred_val_num == 0) & (Ytrue_val == 0));
    
    metrics.val_acc(fold) = (TP_val + TN_val) / length(Ytrue_val);
    metrics.val_prec(fold) = TP_val / (TP_val + FP_val + eps);
    metrics.val_rec(fold) = TP_val / (TP_val + FN_val + eps);
    metrics.val_sens(fold) = metrics.val_rec(fold);
    metrics.val_spec(fold) = TN_val / (TN_val + FP_val + eps);
    metrics.val_f1(fold) = 2 * metrics.val_prec(fold) * metrics.val_rec(fold) / ...
                           (metrics.val_prec(fold) + metrics.val_rec(fold) + eps);
    
    nexttile(tlo); % Select the next slot in the window
    
    % Create the chart for this specific fold
    cm_fold = confusionchart(Yval_cat, YPred_val);
    
    % Add detailed title with metrics
    title(sprintf('Fold %d\nAcc: %.1f%% | F1: %.1f%%', ...
        fold, metrics.val_acc(fold)*100, metrics.val_f1(fold)*100));
        
    % Sort classes to keep "Absent" first
    sortClasses(cm_fold, {'Absent', 'Present'});
    
    % Remove axis labels to save space
    cm_fold.XLabel = '';
    cm_fold.YLabel = '';
    
    fprintf('  Accuracy:    %.2f%%\n', metrics.val_acc(fold)*100);
    fprintf('  Precision:   %.2f%%\n', metrics.val_prec(fold)*100);
    fprintf('  Recall:      %.2f%%\n', metrics.val_rec(fold)*100);
    fprintf('  Sensitivity: %.2f%%\n', metrics.val_sens(fold)*100);
    fprintf('  Specificity: %.2f%%\n', metrics.val_spec(fold)*100);
    fprintf('  F1-Score:    %.2f%%\n', metrics.val_f1(fold)*100);
    
    % Accumulate confusion matrix
    val_confusion_total(1,1) = val_confusion_total(1,1) + TN_val;  % TN
    val_confusion_total(1,2) = val_confusion_total(1,2) + FP_val;  % FP
    val_confusion_total(2,1) = val_confusion_total(2,1) + FN_val;  % FN
    val_confusion_total(2,2) = val_confusion_total(2,2) + TP_val;  % TP
    
    % Store for ROC
    roc_data.val_scores{fold} = scores_val(:, 2);
    roc_data.val_labels{fold} = Yval_cat;
    
    fprintf('\n Fold %d complete\n', fold);
end

fprintf('\n');
fprintf('╔══════════════════════════════════════╗\n');
fprintf('║   ALL %d FOLDS COMPLETED             ║\n', K_FOLDS);
fprintf('╚══════════════════════════════════════╝\n\n');
%%
% Add shared labels to the layout
xlabel(tlo, 'Predicted Class', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(tlo, 'True Class', 'FontSize', 12, 'FontWeight', 'bold');

%% DISPLAY FOLD-BY-FOLD RESULTS
fprintf('========================================\n');
fprintf('FOLD-BY-FOLD RESULTS\n');
fprintf('========================================\n\n');

fprintf('╔════════════════════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Fold  │     TRAINING METRICS      │      VALIDATION METRICS       │\n');
fprintf('╠════════════════════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║        │  Acc   Prec   Rec    F1   │   Acc   Prec   Rec    F1      │\n');
fprintf('╠════════════════════════════════════════════════════════════════════════════════════════════════╣\n');

for fold = 1:K_FOLDS
    fprintf('║   %d    │ %.2f  %.2f  %.2f  %.2f │  %.2f  %.2f  %.2f  %.2f     │\n', ...
        fold, ...
        metrics.train_acc(fold)*100, metrics.train_prec(fold)*100, ...
        metrics.train_rec(fold)*100, metrics.train_f1(fold)*100, ...
        metrics.val_acc(fold)*100, metrics.val_prec(fold)*100, ...
        metrics.val_rec(fold)*100, metrics.val_f1(fold)*100);
end

fprintf('╚════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n');

%% CALCULATE AND DISPLAY MEAN ± STD
fprintf('========================================\n');
fprintf('AGGREGATED RESULTS (Mean ± Std)\n');
fprintf('========================================\n\n');

fprintf('TRAINING SET:\n');
fprintf('  Accuracy:    %.2f%% ± %.2f%%\n', mean(metrics.train_acc)*100, std(metrics.train_acc)*100);
fprintf('  Precision:   %.2f%% ± %.2f%%\n', mean(metrics.train_prec)*100, std(metrics.train_prec)*100);
fprintf('  Recall:      %.2f%% ± %.2f%%\n', mean(metrics.train_rec)*100, std(metrics.train_rec)*100);
fprintf('  Sensitivity: %.2f%% ± %.2f%%\n', mean(metrics.train_sens)*100, std(metrics.train_sens)*100);
fprintf('  Specificity: %.2f%% ± %.2f%%\n', mean(metrics.train_spec)*100, std(metrics.train_spec)*100);
fprintf('  F1-Score:    %.2f%% ± %.2f%%\n\n', mean(metrics.train_f1)*100, std(metrics.train_f1)*100);

fprintf('VALIDATION SET:\n');
fprintf('  Accuracy:    %.2f%% ± %.2f%%\n', mean(metrics.val_acc)*100, std(metrics.val_acc)*100);
fprintf('  Precision:   %.2f%% ± %.2f%%\n', mean(metrics.val_prec)*100, std(metrics.val_prec)*100);
fprintf('  Recall:      %.2f%% ± %.2f%%\n', mean(metrics.val_rec)*100, std(metrics.val_rec)*100);
fprintf('  Sensitivity: %.2f%% ± %.2f%%\n', mean(metrics.val_sens)*100, std(metrics.val_sens)*100);
fprintf('  Specificity: %.2f%% ± %.2f%%\n', mean(metrics.val_spec)*100, std(metrics.val_spec)*100);
fprintf('  F1-Score:    %.2f%% ± %.2f%%\n\n', mean(metrics.val_f1)*100, std(metrics.val_f1)*100);

% Model stability assessment
fprintf('MODEL STABILITY ASSESSMENT:\n');
val_f1_std = std(metrics.val_f1)*100;
if val_f1_std < 3.0
    fprintf('  ✓ EXCELLENT stability (F1 std: %.2f%%)\n', val_f1_std);
elseif val_f1_std < 5.0
    fprintf('  ✓ GOOD stability (F1 std: %.2f%%)\n', val_f1_std);
elseif val_f1_std < 8.0
    fprintf('  ⚠ MODERATE stability (F1 std: %.2f%%)\n', val_f1_std);
else
    fprintf('  ✗ POOR stability (F1 std: %.2f%%) - Consider more data or regularization\n', val_f1_std);
end
fprintf('\n');

%% AGGREGATED VALIDATION CONFUSION MATRIX
fprintf('========================================\n');
fprintf('AGGREGATED VALIDATION CONFUSION MATRIX\n');
fprintf('========================================\n\n');

TN_total = val_confusion_total(1,1);
FP_total = val_confusion_total(1,2);
FN_total = val_confusion_total(2,1);
TP_total = val_confusion_total(2,2);

% Calculate overall metrics from aggregated confusion matrix
acc_overall = (TP_total + TN_total) / sum(val_confusion_total(:));
prec_overall = TP_total / (TP_total + FP_total + eps);
rec_overall = TP_total / (TP_total + FN_total + eps);
sens_overall = rec_overall;
spec_overall = TN_total / (TN_total + FP_total + eps);
f1_overall = 2 * prec_overall * rec_overall / (prec_overall + rec_overall + eps);

fprintf('Confusion Matrix (All Folds Combined):\n');
fprintf('                 Predicted\n');
fprintf('                Absent  Present\n');
fprintf('   Actual  Absent   %4d     %4d\n', TN_total, FP_total);
fprintf('          Present   %4d     %4d\n\n', FN_total, TP_total);

fprintf('Metrics from Aggregated Confusion Matrix:\n');
fprintf('  Accuracy:    %.2f%%\n', acc_overall*100);
fprintf('  Precision:   %.2f%%\n', prec_overall*100);
fprintf('  Recall:      %.2f%%\n', rec_overall*100);
fprintf('  Sensitivity: %.2f%%\n', sens_overall*100);
fprintf('  Specificity: %.2f%%\n', spec_overall*100);
fprintf('  F1-Score:    %.2f%%\n\n', f1_overall*100);

% Visualize 
figure('Name', 'Validation Confusion Matrix (Aggregated)', 'Position', [100 100 500 450]);

total_samples = TN_total + FP_total + FN_total + TP_total;

actual_labels = categorical([...
    repmat({'Absent'}, TN_total, 1); ...
    repmat({'Absent'}, FP_total, 1); ...
    repmat({'Present'}, FN_total, 1); ...
    repmat({'Present'}, TP_total, 1)], ...
    {'Absent', 'Present'});

predicted_labels = categorical([...
    repmat({'Absent'}, TN_total, 1); ...
    repmat({'Present'}, FP_total, 1); ...
    repmat({'Absent'}, FN_total, 1); ...
    repmat({'Present'}, TP_total, 1)], ...
    {'Absent', 'Present'});

cm = confusionchart(actual_labels, predicted_labels);
cm.Title = sprintf('Validation Confusion Matrix\n(Aggregated across %d folds)', K_FOLDS);
%cm.RowSummary = 'row-normalized';
%cm.ColumnSummary = 'column-normalized';

fprintf(' Confusion matrix visualization created\n\n');
%% ROC AND AUC CURVES
fprintf('========================================\n');
fprintf('ROC CURVES AND AUC\n');
fprintf('========================================\n\n');

% Aggregate all training data
all_train_scores = vertcat(roc_data.train_scores{:});
all_train_labels = vertcat(roc_data.train_labels{:});

% Aggregate all validation data
all_val_scores = vertcat(roc_data.val_scores{:});
all_val_labels = vertcat(roc_data.val_labels{:});

% Calculate ROC curves
[Xroc_train, Yroc_train, ~, AUC_train] = perfcurve(all_train_labels, all_train_scores, 'Present');
[Xroc_val, Yroc_val, ~, AUC_val] = perfcurve(all_val_labels, all_val_scores, 'Present');

fprintf('AUC Results (Aggregated):\n');
fprintf('  Training AUC:   %.4f\n', AUC_train);
fprintf('  Validation AUC: %.4f\n\n', AUC_val);

% Plot ROC Curves
figure('Name', 'ROC Curves - K-Fold Cross-Validation', 'Position', [150 150 700 550], 'Color', 'w');
hold on;

% Plot curves
plot(Xroc_train, Yroc_train, 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410], ...
     'DisplayName', sprintf('Training (AUC = %.3f)', AUC_train));
plot(Xroc_val, Yroc_val, 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980], ...
     'LineStyle', '--', 'DisplayName', sprintf('Validation (AUC = %.3f)', AUC_val));

% Random chance line
plot([0 1], [0 1], 'k:', 'LineWidth', 1.5, 'DisplayName', 'Random Chance');

% Formatting
xlabel('False Positive Rate (1 - Specificity)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('True Positive Rate (Sensitivity)', 'FontWeight', 'bold', 'FontSize', 14);
title(sprintf('ROC Curves - Vowel A Detection\n(%d-Fold Cross-Validation)', K_FOLDS), ...
      'FontSize', 16, 'FontWeight', 'bold');
%legend('Location', 'SouthEast', 'FontSize', 12);

lgd = legend('Location', 'SouthEast');
lgd.FontSize = 12;         % Legend font size
lgd.FontWeight = 'bold';   % Legend bold

grid on;
axis square;
xlim([0 1]);
ylim([0 1]);
hold off;

fprintf(' ROC curves plotted\n\n');
%% HELPER FUNCTION
function features = extractLSTMFeatures(table, idx, freqMin, freqMax, sampleRate)
    % Extract spectrogram features
    spec1 = abs(single(table.EMG_ch1_Spectrogram{idx}));
    spec2 = abs(single(table.EMG_ch2_Spectrogram{idx}));
    
    % Filter frequency range
    freqsSpec = linspace(0, sampleRate/2, size(spec1,1));
    freqIdxSpec = (freqsSpec >= freqMin) & (freqsSpec <= freqMax);
    spec1_filtered = spec1(freqIdxSpec, :);
    spec2_filtered = spec2(freqIdxSpec, :);
    
    currentWidth = size(spec1_filtered, 2);
    
    % Resize to standard size
    spec1_resized = imresize(spec1_filtered, [450, currentWidth]);
    spec2_resized = imresize(spec2_filtered, [450, currentWidth]);
    
    % Stack features
    features = [spec1_resized; spec2_resized];
    features = single(features);
end