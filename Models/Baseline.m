%% COMPLETE LOGISTIC REGRESSION VOWEL A CLASSIFIER
clc; clear; close all;

rng(42, 'twister');
fprintf('Random seed set to 42\n\n');
%% CONFIGURATION

FREQ_MIN = 0.1;
FREQ_MAX = 450;
SAMPLE_RATE = 12500;
%% LOAD DATA

fprintf('=== Loading Data ===\n');
tic;

trainData = load('trainv2.mat');
FinalTableTrain = trainData.FinalTable;
clear trainData

testData = load('testv2.mat');
FinalTableTest = testData.FinalTable;
clear testData

fprintf(' Loaded in %.1f sec\n', toc);
fprintf('  Train: %d rows\n', height(FinalTableTrain));
fprintf('  Test: %d rows\n\n', height(FinalTableTest));
%% PROCESS LABELS

fprintf('=== Processing Labels ===\n');

v = upper(strtrim(FinalTableTrain.Vowels));
v(v == "" | v == "N" | v == "NONE") = "";
FinalTableTrain.Vowels = v;

v_test = upper(strtrim(FinalTableTest.Vowels));
v_test(v_test == "" | v_test == "N" | v_test == "NONE") = "";
FinalTableTest.Vowels = v_test;

FinalTableTrain = addVowelsMultiHot(FinalTableTrain);
FinalTableTest  = addVowelsMultiHot(FinalTableTest);

Ytrain_A = double(FinalTableTrain.isA);
Ytest_A  = double(FinalTableTest.isA);

fprintf('Training distribution:\n');
fprintf('  Present (1): %d (%.1f%%)\n', sum(Ytrain_A==1), 100*sum(Ytrain_A==1)/length(Ytrain_A));
fprintf('  Absent  (0): %d (%.1f%%)\n', sum(Ytrain_A==0), 100*sum(Ytrain_A==0)/length(Ytrain_A));

fprintf('\nTest distribution:\n');
fprintf('  Present (1): %d (%.1f%%)\n', sum(Ytest_A==1), 100*sum(Ytest_A==1)/length(Ytest_A));
fprintf('  Absent  (0): %d (%.1f%%)\n\n', sum(Ytest_A==0), 100*sum(Ytest_A==0)/length(Ytest_A));
%% EXTRACT TRAINING FEATURES

fprintf('=== Extracting Training Features ===\n');
fprintf('This will take several minutes...\n\n');

N = height(FinalTableTrain);
Xtrain_cell = cell(N, 1);

tic;
for i = 1:N
    Xtrain_cell{i} = extractLRFeatures(FinalTableTrain, i, FREQ_MIN, FREQ_MAX, SAMPLE_RATE);

    if mod(i, 100) == 0 || i == N
        elapsed = toc;
        pct = 100 * i / N;
        eta = elapsed * (N - i) / i;
        fprintf('  [%6d/%6d] %.1f%% | Elapsed: %.1fm | ETA: %.1fm\n', ...
            i, N, pct, elapsed/60, eta/60);
    end
end

fprintf('\n Training features ready (%.1f min)\n', toc/60);
fprintf('  Feature vector length: %d\n\n', length(Xtrain_cell{1}));

clear FinalTableTrain
%% EXTRACT TEST FEATURES

fprintf('=== Extracting Test Features ===\n');

M = height(FinalTableTest);
Xtest_cell = cell(M, 1);

tic;
for i = 1:M
    Xtest_cell{i} = extractLRFeatures(FinalTableTest, i, FREQ_MIN, FREQ_MAX, SAMPLE_RATE);

    if mod(i, 50) == 0 || i == M
        elapsed = toc;
        pct = 100 * i / M;
        eta = elapsed * (M - i) / i;
        fprintf('  [%4d/%4d] %.1f%% | ETA: %.1fm\n', i, M, pct, eta/60);
    end
end

fprintf('\n Test features ready (%.1f min)\n\n', toc/60);
clear FinalTableTest
%% TRAIN/VALIDATION SPLIT

fprintf('=== Creating Train/Val Split ===\n');

valRatio   = 0.2;
numSamples = length(Ytrain_A);
shuffleIdx = randperm(numSamples);
valCount   = round(numSamples * valRatio);

trainIdx = shuffleIdx(valCount+1:end);
valIdx   = shuffleIdx(1:valCount);

Xtrain = Xtrain_cell(trainIdx);
Ytrain = Ytrain_A(trainIdx);

Xval = Xtrain_cell(valIdx);
Yval = Ytrain_A(valIdx);

fprintf('  Training:   %d samples\n', length(trainIdx));
fprintf('  Validation: %d samples\n\n', length(valIdx));

original_counts = [sum(Ytrain==0), sum(Ytrain==1)];
fprintf('Training distribution (before augmentation):\n');
fprintf('  Absent (0):  %d\n', original_counts(1));
fprintf('  Present (1): %d\n\n', original_counts(2));
%% DATA AUGMENTATION

fprintf('=== Augmenting Training Data ===\n');

idx0 = find(Ytrain == 0);
idx1 = find(Ytrain == 1);

if length(idx0) < length(idx1)
    minority_idx   = idx0; minority_label = 0;
    majority_idx   = idx1;
    fprintf('  Minority class: Absent (0)\n');
else
    minority_idx   = idx1; minority_label = 1;
    majority_idx   = idx0;
    fprintf('  Minority class: Present (1)\n');
end

targetN        = length(majority_idx);
numToGenerate  = targetN - length(minority_idx);
noise_std      = 0.05;

fprintf('Generating %d augmented samples...\n', numToGenerate);

augmented_features = cell(numToGenerate, 1);
augmented_labels   = zeros(numToGenerate, 1);

tic;
for i = 1:numToGenerate
    src = minority_idx(randi(length(minority_idx)));
    augmented_features{i} = Xtrain{src} + noise_std * randn(size(Xtrain{src}), 'single');
    augmented_labels(i)   = minority_label;

    if mod(i, 100) == 0 || i == numToGenerate
        fprintf('  Generated: %d/%d\n', i, numToGenerate);
    end
end

fprintf(' Augmentation done (%.1f sec)\n\n', toc);

XtrainFinal        = [Xtrain; augmented_features];
YtrainFinal_numeric= [Ytrain; augmented_labels];

balanced_counts = [sum(YtrainFinal_numeric==0), sum(YtrainFinal_numeric==1)];
fprintf('Balanced training set:\n');
fprintf('  Absent (0):  %d (%.1f%%)\n', balanced_counts(1), 100*balanced_counts(1)/sum(balanced_counts));
fprintf('  Present (1): %d (%.1f%%)\n\n', balanced_counts(2), 100*balanced_counts(2)/sum(balanced_counts));

clear Xtrain augmented_features Xtrain_cell
%% DATA NORMALIZATION (Standard Scaler)

fprintf('=== Normalizing Features ===\n');

% Convert cell arrays to matrices (samples × features)
XtrainMat = double(cell2mat(XtrainFinal')');   % [N_train × D]
XvalMat   = double(cell2mat(Xval')');          % [N_val   × D]
XtestMat  = double(cell2mat(Xtest_cell')');    % [N_test  × D]

mu    = mean(XtrainMat, 1);
sigma = std(XtrainMat, 0, 1) + eps;

XtrainMat = (XtrainMat - mu) ./ sigma;
XvalMat   = (XvalMat   - mu) ./ sigma;
XtestMat  = (XtestMat  - mu) ./ sigma;

fprintf('  Normalized %d features.\n\n', size(XtrainMat, 2));

clear XtrainFinal Xval Xtest_cell
%% TRAIN LOGISTIC REGRESSION MODEL

fprintf('=== Training Logistic Regression ===\n');

YtrainFinal_cat = categorical(YtrainFinal_numeric, [0,1], {'Absent','Present'});
Yval_cat        = categorical(Yval,                [0,1], {'Absent','Present'});
Ytest_cat       = categorical(Ytest_A,             [0,1], {'Absent','Present'});

tic;
model_LR = fitclinear(XtrainMat, YtrainFinal_cat, ...
    'Learner',      'logistic', ...
    'Regularization','ridge', ...
    'Lambda',        1e-4, ...
    'Solver',        'lbfgs', ...
    'Verbose',       1);

trainTime = toc;
fprintf('\n Training done in %.1f minutes\n\n', trainTime/60);  
%% EVALUATE ALL SETS

fprintf('=== Evaluating Training Set ===\n');
[YPred_train, scores_train] = predict(model_LR, XtrainMat);
printMetrics(YtrainFinal_cat, YPred_train, 'Training');

fprintf('=== Evaluating Validation Set ===\n');
[YPred_val, scores_val] = predict(model_LR, XvalMat);
printMetrics(Yval_cat, YPred_val, 'Validation');

fprintf('=== Evaluating Test Set ===\n');
[YPred_test, scores_test] = predict(model_LR, XtestMat);
printMetrics(Ytest_cat, YPred_test, 'Test');
%% COMPARATIVE SUMMARY
% Recalculate for table

calc = @(Ytrue, YPred) deal( ...
    sum(double(YPred=="Present")==double(Ytrue=="Present"))/length(Ytrue), ...
    sum((double(YPred=="Present")==1)&(double(Ytrue=="Present")==1)) / ...
        (sum((double(YPred=="Present")==1)&(double(Ytrue=="Present")==1)) + ...
         sum((double(YPred=="Present")==1)&(double(Ytrue=="Present")==0)) + eps), ...
    sum((double(YPred=="Present")==1)&(double(Ytrue=="Present")==1)) / ...
        (sum((double(YPred=="Present")==1)&(double(Ytrue=="Present")==1)) + ...
         sum((double(YPred=="Present")==0)&(double(Ytrue=="Present")==1)) + eps) );

[acc_tr, pr_tr, re_tr] = calc(YtrainFinal_cat, YPred_train);
[acc_v,  pr_v,  re_v]  = calc(Yval_cat,        YPred_val);
[acc_te, pr_te, re_te]  = calc(Ytest_cat,       YPred_test);

F1_tr = 2*pr_tr*re_tr/(pr_tr+re_tr+eps);
F1_v  = 2*pr_v *re_v /(pr_v +re_v +eps);
F1_te = 2*pr_te*re_te/(pr_te+re_te+eps);

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                   COMPARATIVE SUMMARY                             ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Metric      │   Training  │  Validation │    Test     │  T-T Gap ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Accuracy    │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', acc_tr*100, acc_v*100, acc_te*100, (acc_tr-acc_te)*100);
fprintf('║  Precision   │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', pr_tr*100,  pr_v*100,  pr_te*100,  (pr_tr-pr_te)*100);
fprintf('║  Recall      │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', re_tr*100,  re_v*100,  re_te*100,  (re_tr-re_te)*100);
fprintf('║  F1-Score    │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', F1_tr*100,  F1_v*100,  F1_te*100,  (F1_tr-F1_te)*100);
fprintf('╚═══════════════════════════════════════════════════════════════════╝\n\n');
%% VISUALIZATIONS

fprintf('=== Creating Visualizations ===\n');

% Confusion Matrices
figure('Position', [50 50 1500 500], 'Name', 'Confusion Matrices');
subplot(1,3,1); confusionchart(YtrainFinal_cat, YPred_train); title(sprintf('Training (Acc: %.1f%%)', acc_tr*100));
subplot(1,3,2); confusionchart(Yval_cat,        YPred_val);   title(sprintf('Validation (Acc: %.1f%%)', acc_v*100));
subplot(1,3,3); confusionchart(Ytest_cat,        YPred_test);  title(sprintf('Test (Acc: %.1f%%)', acc_te*100));

% Class Distribution
figure('Position', [100 100 1000 500], 'Name', 'Class Distribution');
subplot(1,2,1);
b1 = bar(categorical({'Absent','Present'}), original_counts);
title('Before Augmentation'); ylabel('Count');
b1.FaceColor = 'flat'; b1.CData(1,:) = [0.8 0.3 0.3]; b1.CData(2,:) = [0.3 0.8 0.3];
subplot(1,2,2);
b2 = bar(categorical({'Absent','Present'}), balanced_counts);
title('After Augmentation'); ylabel('Count');
b2.FaceColor = 'flat'; b2.CData(1,:) = [0.8 0.3 0.3]; b2.CData(2,:) = [0.3 0.8 0.3];
%% ROC & AUC

fprintf('=== Calculating ROC and AUC ===\n');

% scores column 2 = P(Present)
[Xroc_tr, Yroc_tr, ~, AUC_tr] = perfcurve(YtrainFinal_cat, scores_train(:,2), 'Present');
[Xroc_v,  Yroc_v,  ~, AUC_v]  = perfcurve(Yval_cat,        scores_val(:,2),   'Present');
[Xroc_te, Yroc_te, ~, AUC_te] = perfcurve(Ytest_cat,        scores_test(:,2),  'Present');

fprintf('  Training AUC:   %.4f\n', AUC_tr);
fprintf('  Validation AUC: %.4f\n', AUC_v);
fprintf('  Test AUC:       %.4f\n', AUC_te);

figure('Name', 'ROC Curves', 'Color', 'w', 'Position', [100 100 600 500]);
hold on;
plot(Xroc_tr, Yroc_tr, 'LineWidth', 2, 'Color', 'b',           'DisplayName', sprintf('Train (AUC=%.2f)', AUC_tr));
plot(Xroc_v,  Yroc_v,  'LineWidth', 2, 'Color', 'g', 'LineStyle', '--', 'DisplayName', sprintf('Val (AUC=%.2f)', AUC_v));
plot(Xroc_te, Yroc_te, 'LineWidth', 2, 'Color', 'r', 'LineStyle', '-.', 'DisplayName', sprintf('Test (AUC=%.2f)', AUC_te));
plot([0 1], [0 1], 'k:', 'LineWidth', 1, 'DisplayName', 'Random Chance');
xlabel('False Positive Rate', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('True Positive Rate',  'FontWeight', 'bold', 'FontSize', 14);
title('ROC Curves - Vowel A Detection (Logistic Regression)', 'FontSize', 16);
ax = gca; ax.FontSize = 14; ax.FontWeight = 'bold';
lgd = legend('Location', 'SouthEast'); lgd.FontSize = 12; lgd.FontWeight = 'bold';
grid on; axis square; hold off;
%% FINAL SUMMARY

fprintf('╔════════════════════════════════════════════╗\n');
fprintf('║  TRAINING COMPLETE (Logistic Regression)   ║\n');
fprintf('╠════════════════════════════════════════════╣\n');
fprintf('║  Total time: %.1f minutes                   ║\n', trainTime/60);
fprintf('║  Test accuracy: %.2f%%                     ║\n', acc_te*100);
fprintf('║  Test F1-Score: %.2f%%                     ║\n', F1_te*100);
fprintf('╚════════════════════════════════════════════╝\n');
fprintf('\n All done!\n');
%% HELPER FUNCTIONS

function features = extractLRFeatures(table, idx, freqMin, freqMax, sampleRate)
    spec1 = abs(single(table.EMG_ch1_Spectrogram{idx}));
    spec2 = abs(single(table.EMG_ch2_Spectrogram{idx}));

    freqsSpec    = linspace(0, sampleRate/2, size(spec1,1));
    freqIdxSpec  = (freqsSpec >= freqMin) & (freqsSpec <= freqMax);
    spec1_filtered = spec1(freqIdxSpec, :);
    spec2_filtered = spec2(freqIdxSpec, :);

    currentWidth = size(spec1_filtered, 2);
    spec1_resized = imresize(spec1_filtered, [450, currentWidth]);
    spec2_resized = imresize(spec2_filtered, [450, currentWidth]);

    % Flatten 2D → 1D vector for Logistic Regression
    stacked  = [spec1_resized; spec2_resized];   
    features = mean(stacked, 2);                     
    features = single(features);
end
%%
function printMetrics(Ytrue_cat, YPred_cat, setName)
    Ytrue_num = double(Ytrue_cat == "Present");
    YPred_num = double(YPred_cat == "Present");

    TP = sum((YPred_num==1) & (Ytrue_num==1));
    FP = sum((YPred_num==1) & (Ytrue_num==0));
    FN = sum((YPred_num==0) & (Ytrue_num==1));
    TN = sum((YPred_num==0) & (Ytrue_num==0));

    acc  = sum(YPred_num == Ytrue_num) / length(Ytrue_num);
    prec = TP / (TP + FP + eps);
    rec  = TP / (TP + FN + eps);
    F1   = 2 * prec * rec / (prec + rec + eps);

    fprintf('%s Results:\n', setName);
    fprintf('  Accuracy:  %.2f%%\n', acc*100);
    fprintf('  Precision: %.2f%%\n', prec*100);
    fprintf('  Recall:    %.2f%%\n', rec*100);
    fprintf('  F1-Score:  %.2f%%\n', F1*100);
    fprintf('  TN=%d  FP=%d\n', TN, FP);
    fprintf('  FN=%d  TP=%d\n\n', FN, TP);
end