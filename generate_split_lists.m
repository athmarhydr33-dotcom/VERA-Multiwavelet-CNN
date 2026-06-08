% =========================================================
% generate_split_lists.m
% -----------------------
% Generates and saves train/validation split lists
% for all six evaluation scenarios.
%
% Requested by Reviewer 1:
%   "publicly release train/validation split lists"
%
% OUTPUT: split_lists/
%   split_palm_train.mat   — training image indices/names (palm)
%   split_palm_val.mat     — validation image indices/names (palm)
%   split_finger_train.mat
%   split_finger_val.mat
%   split_fusion_train.mat
%   split_fusion_val.mat
%
% Random seed: 42 (fixed for reproducibility)
% =========================================================

clear; clc;

rng(42);

if ~exist('split_lists','dir'), mkdir('split_lists'); end

scenarios = {'palm','finger','fusion'};

for sc = 1:length(scenarios)
    scenario = scenarios{sc};

    if strcmp(scenario,'fusion')
        palmData   = load('VERA_features_DMWT_palm.mat');
        fingerData = load('VERA_features_DMWT_finger.mat');
        labels_person   = [palmData.labels_person,   fingerData.labels_person];
        labels_filename = [palmData.labels_filename, fingerData.labels_filename];
        labels_session  = [palmData.labels_session,  fingerData.labels_session];
    else
        data = load('VERA_features_DMWT.mat');
        labels_person   = data.labels_person;
        labels_filename = data.labels_filename;
        labels_session  = data.labels_session;
    end

    uniquePersons = unique(labels_person);
    numClasses    = length(uniquePersons);
    N             = length(labels_person);
    Y             = zeros(N,1);

    for p = 1:numClasses
        idx = strcmp(labels_person, uniquePersons{p});
        Y(idx) = p;
    end
    Y = categorical(Y);

    % Stratified 80/20 split
    trainIdx = [];  valIdx = [];
    for c = 1:numClasses
        classIdx = find(Y == categorical(c));
        classIdx = classIdx(randperm(length(classIdx)));
        n        = length(classIdx);
        nTrain   = round(0.80 * n);
        trainIdx = [trainIdx; classIdx(1:nTrain)];
        valIdx   = [valIdx;   classIdx(nTrain+1:end)];
    end

    % Build split tables
    train_persons   = labels_person(trainIdx)';
    train_sessions  = labels_session(trainIdx)';
    train_filenames = labels_filename(trainIdx)';
    train_classIDs  = double(Y(trainIdx));

    val_persons     = labels_person(valIdx)';
    val_sessions    = labels_session(valIdx)';
    val_filenames   = labels_filename(valIdx)';
    val_classIDs    = double(Y(valIdx));

    save(sprintf('split_lists/split_%s_train.mat', scenario), ...
        'train_persons','train_sessions','train_filenames','train_classIDs');

    save(sprintf('split_lists/split_%s_val.mat', scenario), ...
        'val_persons','val_sessions','val_filenames','val_classIDs');

    fprintf('✅ %s: %d train / %d val (seed=42)\n', ...
        scenario, length(trainIdx), length(valIdx));
end

fprintf('\nAll split lists saved to split_lists/\n');
