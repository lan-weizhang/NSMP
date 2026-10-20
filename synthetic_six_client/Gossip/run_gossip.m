function varargout = run_gossip()
% Paper six-client Gossip baseline; original numerical routines retained.
close all;
clc;

cfg.seed = 0;
cfg.numClients = 6;
cfg.dimension = 11;
cfg.samples = 100;
cfg.noiseStd = 10;
cfg.numRounds = 500;
cfg.gossipBatchSize = cfg.samples;
cfg.gossipStepSafety = 0.9;
cfg.convergenceTolerance = 1e-15;
cfg.plotFloor = 1e-16;

[data, blocks, edges] = generate_six_client_problem(cfg);
[xStar, fStar] = centralized_solution(data, blocks, cfg.dimension);

localH = cell(cfg.numClients, 1);
localB = cell(cfg.numClients, 1);
maxLocalLipschitz = 0;
for i = 1:cfg.numClients
    Xi = data.X{i};
    yi = data.y{i};
    localH{i} = (Xi' * Xi) / size(Xi, 1);
    localB{i} = (Xi' * yi) / size(Xi, 1);
    maxLocalLipschitz = max(maxLocalLipschitz, max(eig(symmetrize(localH{i}))));
end

gossipStep = cfg.gossipStepSafety / maxLocalLipschitz;

gossip = run_gossip_dsgd(data, blocks, edges, localH, localB, ...
    xStar, fStar, gossipStep, cfg);


[gossipRound, gossipScalars] = first_round_to_tolerance( ...
    gossip.relativeObjectiveGap, gossip.communicatedScalars, cfg.convergenceTolerance);
numEdges = size(edges, 1);
method = {'Gossip'};
roundsToTolerance = gossipRound;
directedMessages = 2*numEdges*gossipRound;
communicatedScalars = gossipScalars;
communicatedBytesFloat64 = 8*communicatedScalars;
communicationSummary = table(method, roundsToTolerance, directedMessages, ...
    communicatedScalars, communicatedBytesFloat64);
outputDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
writetable(communicationSummary, fullfile(outputDir, 'communication_to_1e-15.csv'));
roundIndex = (0:cfg.numRounds)';
convergenceTable = table(roundIndex, gossip.relativeObjectiveGap, ...
    gossip.relativeParameterError, gossip.communicatedScalars, ...
    'VariableNames', {'round','gossip_relative_objective_gap', ...
    'gossip_relative_parameter_error','gossip_communicated_scalars'});
writetable(convergenceTable, fullfile(outputDir, 'gossip_convergence.csv'));
results.config = cfg;
results.blocks = blocks;
results.edges = edges;
results.centralized.solution = xStar;
results.centralized.objective = fStar;
results.gossip = gossip;
results.communicationSummary = communicationSummary;
save(fullfile(outputDir, 'gossip_convergence.mat'), 'results');
fprintf('\n===== Six-client Gossip =====\n');
fprintf('%-32s : relative objective gap <= %.0e\n', ...
    'Checkpoint criterion', cfg.convergenceTolerance);
fprintf('%-32s : %d\n', 'Iteration budget', cfg.numRounds);
fprintf('\n%-12s %10s %12s %12s %14s\n', ...
    'Method', 'Rounds', 'Messages', 'Scalars', 'Bytes (F64)');
fprintf('%s\n', repmat('-', 1, 64));
fprintf('%-12s %10g %12g %12g %14g\n', ...
    'Gossip', gossipRound, 2 * numEdges * gossipRound, ...
    gossipScalars, 8 * gossipScalars);
fprintf('\n');
if isfinite(gossipRound)
    checkpoint = gossipRound + 1; % History includes the initial round zero.
    fprintf('%-32s : %.3e\n', 'Gossip objective gap (stop)', ...
        gossip.relativeObjectiveGap(checkpoint));
    fprintf('%-32s : %.3e\n', 'Gossip parameter error (stop)', ...
        gossip.relativeParameterError(checkpoint));
else
    fprintf('%-32s : not reached within budget\n', 'Checkpoint criterion');
end
fprintf('%-32s : %.3e\n', 'Gossip objective gap (final)', ...
    gossip.relativeObjectiveGap(end));
fprintf('%-32s : %.3e\n', 'Gossip parameter error (final)', ...
    gossip.relativeParameterError(end));
fprintf('%-32s : %s\n', 'Output directory', outputDir);
if nargout > 0
    varargout{1} = results;
end
end

function [data, blocks, edges] = generate_six_client_problem(cfg)
% Reproduce the random-number order and data construction in main_6_users.m.
rng(cfg.seed, 'twister');
N = cfg.samples;
sigma = cfg.noiseStd;

X = cell(cfg.numClients, 1);
y = cell(cfg.numClients, 1);

X{1} = randn(N, 2);
a1 = randn(2, 1);
y{1} = X{1} * a1 + sigma * randn(N, 1);

X{2} = randn(N, 2);
a2 = randn(2, 1);
y{2} = X{2} * a2 + sigma * randn(N, 1);

X{3} = randn(N, 4);
a3 = randn(4, 1);
a3(1) = a1(2);
a3(2) = a2(2);
y{3} = X{3} * a3 + sigma * randn(N, 1);

X{4} = randn(N, 2);
a4 = randn(2, 1);
y{4} = X{4} * a4 + sigma * randn(N, 1);

X{5} = randn(N, 3);
a5 = randn(3, 1);
a5(1) = a4(2);
y{5} = X{5} * a5 + sigma * randn(N, 1);

X{6} = randn(N, 3);
a6 = randn(3, 1);
a6(1) = a3(4);
a6(2) = a5(3);
y{6} = X{6} * a6 + sigma * randn(N, 1);

blocks = {
    [1, 2]
    [3, 4]
    [2, 4, 5, 6]
    [7, 8]
    [8, 9, 10]
    [6, 10, 11]
    };

% Columns are [client_i, client_j, shared_global_coordinate].
edges = [
    1, 3, 2
    2, 3, 4
    3, 6, 6
    4, 5, 8
    5, 6, 10
    ];

data.X = X;
data.y = y;
end


function [xStar, fStar] = centralized_solution(data, blocks, dimension)
numClients = numel(blocks);
numRows = 0;
for i = 1:numClients
    numRows = numRows + size(data.X{i}, 1);
end

A = zeros(numRows, dimension);
yAll = zeros(numRows, 1);
firstRow = 1;
for i = 1:numClients
    numLocalRows = size(data.X{i}, 1);
    rows = firstRow:(firstRow + numLocalRows - 1);
    A(rows, blocks{i}) = data.X{i};
    yAll(rows) = data.y{i};
    firstRow = firstRow + numLocalRows;
end

xStar = A \ yAll;
fStar = joint_objective(xStar, data, blocks);
end


function out = run_gossip_dsgd(data, blocks, edges, localH, localB, ...
    xStar, fStar, baseStep, cfg)

localModels = initialize_local_models(blocks);
T = cfg.numRounds;
out.relativeObjectiveGap = zeros(T + 1, 1);
out.relativeParameterError = zeros(T + 1, 1);
out.communicatedScalars = zeros(T + 1, 1);

[out.relativeObjectiveGap(1), out.relativeParameterError(1)] = ...
    evaluate_models(localModels, data, blocks, xStar, fStar, cfg.plotFloor);

useFullBatch = cfg.gossipBatchSize >= cfg.samples;
for t = 1:T
    tentative = localModels;
    if useFullBatch
        step = baseStep;
    else
        step = baseStep / sqrt(t);
    end

    for i = 1:numel(blocks)
        if useFullBatch
            gradient = localH{i} * localModels{i} - localB{i};
        else
            batchSize = min(cfg.gossipBatchSize, size(data.X{i}, 1));
            indices = randperm(size(data.X{i}, 1), batchSize);
            Xi = data.X{i}(indices, :);
            yi = data.y{i}(indices);
            gradient = (Xi' * (Xi * localModels{i} - yi)) / batchSize;
        end
        tentative{i} = localModels{i} - step * gradient;
    end

    % One synchronous pairwise gossip operation on every tree edge.
    for e = 1:size(edges, 1)
        clientI = edges(e, 1);
        clientJ = edges(e, 2);
        coordinate = edges(e, 3);
        indexI = find(blocks{clientI} == coordinate, 1);
        indexJ = find(blocks{clientJ} == coordinate, 1);
        averageValue = 0.5 * (tentative{clientI}(indexI) + ...
            tentative{clientJ}(indexJ));
        tentative{clientI}(indexI) = averageValue;
        tentative{clientJ}(indexJ) = averageValue;
    end
    localModels = tentative;

    [out.relativeObjectiveGap(t + 1), out.relativeParameterError(t + 1)] = ...
        evaluate_models(localModels, data, blocks, xStar, fStar, cfg.plotFloor);
    % Each endpoint sends one scalar to the other endpoint per edge.
    out.communicatedScalars(t + 1) = 2 * size(edges, 1) * t;
end

out.localModels = localModels;
out.solution = assemble_global_model(localModels, blocks, numel(xStar));
out.stepSize = baseStep;
out.batchSize = cfg.gossipBatchSize;
end


function localModels = initialize_local_models(blocks)
localModels = cell(numel(blocks), 1);
for i = 1:numel(blocks)
    localModels{i} = zeros(numel(blocks{i}), 1);
end
end


function [relativeGap, relativeError] = evaluate_models(localModels, data, ...
    blocks, xStar, fStar, plotFloor)
x = assemble_global_model(localModels, blocks, numel(xStar));
objectiveValue = joint_objective(x, data, blocks);
relativeGap = max((objectiveValue - fStar) / max(abs(fStar), eps), plotFloor);
relativeError = max(norm(x - xStar) / max(norm(xStar), eps), plotFloor);
end


function x = assemble_global_model(localModels, blocks, dimension)
% Averaging duplicate copies is a safeguard; recorded algorithm iterates
% already agree on every overlapping coordinate.
x = zeros(dimension, 1);
counts = zeros(dimension, 1);
for i = 1:numel(blocks)
    x(blocks{i}) = x(blocks{i}) + localModels{i};
    counts(blocks{i}) = counts(blocks{i}) + 1;
end
x = x ./ counts;
end


function value = joint_objective(x, data, blocks)
value = 0;
for i = 1:numel(blocks)
    residual = data.X{i} * x(blocks{i}) - data.y{i};
    value = value + 0.5 * (residual' * residual) / numel(residual);
end
end


function [roundNumber, scalarCount] = first_round_to_tolerance( ...
    relativeGap, communicatedScalars, threshold)
firstIndex = find(relativeGap <= threshold, 1);
if isempty(firstIndex)
    roundNumber = NaN;
    scalarCount = NaN;
else
    roundNumber = firstIndex - 1;
    scalarCount = communicatedScalars(firstIndex);
end
end


function A = symmetrize(A)
A = 0.5 * (A + A');
end
