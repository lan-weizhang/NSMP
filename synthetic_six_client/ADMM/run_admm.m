function varargout = run_admm()
%RUN_ADMM Run edge-consensus ADMM on the paper six-client problem.
% Centralized least squares is used only as an accuracy reference.

close all;
clc;

cfg.seed = 0;
cfg.numClients = 6;
cfg.dimension = 11;
cfg.samples = 100;
cfg.noiseStd = 10;
cfg.admmRho = 100; % Equivalent to rho=1 for losses normalized by N=100.
cfg.targetRelativeError = 1e-8;
cfg.admmAbsoluteTolerance = 1e-8;
cfg.admmRelativeTolerance = 1e-8;
cfg.maxIterations = 10000;
cfg.plotFloor = 1e-16;

[data, blocks, edges, parent] = generate_six_client_problem(cfg);
[xStar, fStar] = centralized_solution(data, blocks, cfg.dimension);

localH = cell(cfg.numClients, 1);
localB = cell(cfg.numClients, 1);
for i = 1:cfg.numClients
    localH{i} = data.X{i}' * data.X{i};
    localB{i} = data.X{i}' * data.y{i};
end

admm = run_edge_consensus_admm(localH, localB, blocks, edges, ...
    data, xStar, fStar, cfg);

scriptDir = fileparts(mfilename('fullpath'));
outputDir = fullfile(scriptDir, 'results');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

method = {'ADMM'};
synchronousRoundsOrIterations = admm.iterations;
messages = admm.messages;
communicatedScalars = admm.communicatedScalars;
communicatedBytesFloat64 = 8 * communicatedScalars;
relativeObjectiveGap = admm.relativeObjectiveGap(end);
relativeParameterError = admm.relativeParameterError(end);
relativePrimalResidual = admm.relativePrimalResidual(end);
primalResidual = admm.primalResidual(end);
primalTolerance = admm.primalTolerance(end);
dualResidual = admm.dualResidual(end);
dualTolerance = admm.dualTolerance(end);

summary = table(method, synchronousRoundsOrIterations, messages, ...
    communicatedScalars, communicatedBytesFloat64, relativeObjectiveGap, ...
    relativeParameterError, relativePrimalResidual, primalResidual, ...
    primalTolerance, dualResidual, dualTolerance);
writetable(summary, fullfile(outputDir, 'communication_summary.csv'));

round = (0:admm.iterations)';
admmConvergence = table(round, admm.communicatedScalarsHistory, ...
    admm.relativeObjectiveGap, admm.relativeParameterError, ...
    admm.relativePrimalResidual, admm.primalResidual, admm.dualResidual, ...
    admm.primalTolerance, admm.dualTolerance, ...
    'VariableNames', {'round', 'communicated_scalars', ...
    'relative_objective_gap', 'relative_parameter_error', ...
    'relative_primal_residual', 'primal_residual', 'dual_residual', ...
    'primal_tolerance', 'dual_tolerance'});
writetable(admmConvergence, fullfile(outputDir, 'admm_convergence.csv'));

plot_communication(admm, cfg, outputDir);

results.config = cfg;
results.blocks = blocks;
results.edges = edges;
results.parent = parent;
results.centralized.solution = xStar;
results.centralized.objective = fStar;
results.admm = admm;
results.summary = summary;
save(fullfile(outputDir, 'admm_results.mat'), 'results');

fprintf('\n===== Six-client ADMM =====\n');
fprintf('%-32s : primal and dual residual tests\n', 'Stopping criterion');
fprintf('%-32s : abs=%.0e, rel=%.0e\n', 'Residual tolerances', ...
    cfg.admmAbsoluteTolerance, cfg.admmRelativeTolerance);
fprintf('%-32s : %d\n', 'Iteration budget', cfg.maxIterations);
fprintf('\n%-12s %10s %12s %12s %14s\n', ...
    'Method', 'Rounds', 'Messages', 'Scalars', 'Bytes (F64)');
fprintf('%s\n', repmat('-', 1, 64));
fprintf('%-12s %10d %12d %12d %14d\n', 'ADMM', ...
    admm.iterations, admm.messages, admm.communicatedScalars, ...
    8 * admm.communicatedScalars);
fprintf('\n');
fprintf('%-32s : %.3e\n', 'ADMM objective gap (stop)', ...
    admm.relativeObjectiveGap(end));
fprintf('%-32s : %.3e\n', 'ADMM parameter error (stop)', ...
    admm.relativeParameterError(end));
fprintf('%-32s : %.3e\n', 'ADMM relative primal residual', ...
    admm.relativePrimalResidual(end));
fprintf('%-32s : %.3e / %.3e\n', 'ADMM primal residual / tolerance', ...
    admm.primalResidual(end), admm.primalTolerance(end));
fprintf('%-32s : %.3e / %.3e\n', 'ADMM dual residual / tolerance', ...
    admm.dualResidual(end), admm.dualTolerance(end));
fprintf('%-32s : error <= %.0e, iteration %d, %d scalars\n', ...
    'ADMM accuracy checkpoint', ...
    cfg.targetRelativeError, ...
    admm.accuracyIteration, 2 * size(edges, 1) * admm.accuracyIteration);
fprintf('%-32s : %s\n', 'Output directory', outputDir);
if nargout > 0
    varargout{1} = results;
end
end


function [data, blocks, edges, parent] = generate_six_client_problem(cfg)
% Reproduce main_6_users.m, including its random-number order.
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

% One row per shared coordinate: [endpoint_i, endpoint_j, coordinate].
edges = [
    1, 3, 2
    2, 3, 4
    3, 6, 6
    4, 5, 8
    5, 6, 10
    ];

% Root the same tree used in the paper at client 6.
parent = [3; 3; 6; 5; 6; 0];

data.X = X;
data.y = y;
end


function [xStar, fStar] = centralized_solution(data, blocks, dimension)
numRows = sum(cellfun(@(x) size(x, 1), data.X));
A = zeros(numRows, dimension);
yAll = zeros(numRows, 1);
firstRow = 1;
for i = 1:numel(blocks)
    count = size(data.X{i}, 1);
    rows = firstRow:(firstRow + count - 1);
    A(rows, blocks{i}) = data.X{i};
    yAll(rows) = data.y{i};
    firstRow = firstRow + count;
end
xStar = A \ yAll;
fStar = joint_objective(xStar, data, blocks);
end


function out = run_edge_consensus_admm(localH, localB, blocks, edges, ...
    data, xStar, fStar, cfg)
numClients = numel(blocks);
numEdges = size(edges, 1);
rho = cfg.admmRho;

localModels = cellfun(@(block) zeros(numel(block), 1), blocks, ...
    'UniformOutput', false);
edgeConsensus = zeros(numEdges, 1);
dualAtI = zeros(numEdges, 1);
dualAtJ = zeros(numEdges, 1);

relativeObjectiveGap = zeros(cfg.maxIterations + 1, 1);
relativeParameterError = zeros(cfg.maxIterations + 1, 1);
relativePrimalResidual = zeros(cfg.maxIterations + 1, 1);
primalResidual = zeros(cfg.maxIterations + 1, 1);
dualResidual = zeros(cfg.maxIterations + 1, 1);
primalTolerance = zeros(cfg.maxIterations + 1, 1);
dualTolerance = zeros(cfg.maxIterations + 1, 1);
communicatedScalars = zeros(cfg.maxIterations + 1, 1);
accuracyIteration = NaN;

initial = assemble_feasible_model(localModels, edgeConsensus, ...
    blocks, edges, numel(xStar));
[relativeObjectiveGap(1), relativeParameterError(1)] = ...
    evaluate_solution(initial, data, blocks, xStar, fStar, cfg.plotFloor);

converged = false;
for t = 1:cfg.maxIterations
    for i = 1:numClients
        augmentedH = localH{i};
        augmentedB = localB{i};
        incidentEdges = find(edges(:, 1) == i | edges(:, 2) == i)';
        for e = incidentEdges
            coordinate = edges(e, 3);
            localIndex = find(blocks{i} == coordinate, 1);
            augmentedH(localIndex, localIndex) = ...
                augmentedH(localIndex, localIndex) + rho;
            if edges(e, 1) == i
                augmentedB(localIndex) = augmentedB(localIndex) + ...
                    rho * (edgeConsensus(e) - dualAtI(e));
            else
                augmentedB(localIndex) = augmentedB(localIndex) + ...
                    rho * (edgeConsensus(e) - dualAtJ(e));
            end
        end
        localModels{i} = augmentedH \ augmentedB;
    end

    previousConsensus = edgeConsensus;
    endpointValuesI = zeros(numEdges, 1);
    endpointValuesJ = zeros(numEdges, 1);
    for e = 1:numEdges
        i = edges(e, 1);
        j = edges(e, 2);
        coordinate = edges(e, 3);
        indexI = find(blocks{i} == coordinate, 1);
        indexJ = find(blocks{j} == coordinate, 1);
        endpointValuesI(e) = localModels{i}(indexI);
        endpointValuesJ(e) = localModels{j}(indexJ);

        % Each endpoint sends one scalar (x + u) to the other endpoint.
        % Both endpoints can then compute the same average locally.
        edgeConsensus(e) = 0.5 * (...
            endpointValuesI(e) + dualAtI(e) + ...
            endpointValuesJ(e) + dualAtJ(e));
    end

    residualI = endpointValuesI - edgeConsensus;
    residualJ = endpointValuesJ - edgeConsensus;
    dualAtI = dualAtI + residualI;
    dualAtJ = dualAtJ + residualJ;

    primalResidual(t + 1) = norm([residualI; residualJ]);
    dualResidual(t + 1) = rho * sqrt(2) * ...
        norm(edgeConsensus - previousConsensus);
    scale = max([1; norm(endpointValuesI); norm(endpointValuesJ); ...
        sqrt(2) * norm(edgeConsensus)]);
    relativePrimalResidual(t + 1) = primalResidual(t + 1) / scale;
    primalTolerance(t + 1) = sqrt(2 * numEdges) * ...
        cfg.admmAbsoluteTolerance + cfg.admmRelativeTolerance * ...
        max(norm([endpointValuesI; endpointValuesJ]), ...
        sqrt(2) * norm(edgeConsensus));
    dualTolerance(t + 1) = sqrt(2 * numEdges) * ...
        cfg.admmAbsoluteTolerance + cfg.admmRelativeTolerance * rho * ...
        norm([dualAtI; dualAtJ]);

    feasible = assemble_feasible_model(localModels, edgeConsensus, ...
        blocks, edges, numel(xStar));
    [relativeObjectiveGap(t + 1), relativeParameterError(t + 1)] = ...
        evaluate_solution(feasible, data, blocks, xStar, fStar, cfg.plotFloor);

    % Five scalar interfaces, with one scalar sent in each direction.
    communicatedScalars(t + 1) = 2 * numEdges * t;

    if isnan(accuracyIteration) && ...
            relativeParameterError(t + 1) <= cfg.targetRelativeError && ...
            relativePrimalResidual(t + 1) <= cfg.targetRelativeError
        accuracyIteration = t;
    end

    if primalResidual(t + 1) <= primalTolerance(t + 1) && ...
            dualResidual(t + 1) <= dualTolerance(t + 1)
        converged = true;
        break;
    end
end

if ~converged
    error('SixClientADMM:NoConvergence', ...
        'ADMM did not meet the target tolerances within %d iterations.', ...
        cfg.maxIterations);
end

used = 1:(t + 1);
out.iterations = t;
out.relativeObjectiveGap = relativeObjectiveGap(used);
out.relativeParameterError = relativeParameterError(used);
out.relativePrimalResidual = relativePrimalResidual(used);
out.primalResidual = primalResidual(used);
out.dualResidual = dualResidual(used);
out.primalTolerance = primalTolerance(used);
out.dualTolerance = dualTolerance(used);
out.communicatedScalarsHistory = communicatedScalars(used);
out.communicatedScalars = communicatedScalars(t + 1);
out.messages = 2 * numEdges * t;
out.solution = feasible;
out.edgeConsensus = edgeConsensus;
out.rho = rho;
out.accuracyIteration = accuracyIteration;
end


function x = assemble_feasible_model(localModels, edgeConsensus, blocks, ...
    edges, dimension)
x = zeros(dimension, 1);
counts = zeros(dimension, 1);
for i = 1:numel(blocks)
    x(blocks{i}) = x(blocks{i}) + localModels{i};
    counts(blocks{i}) = counts(blocks{i}) + 1;
end
x = x ./ max(counts, 1);
for e = 1:size(edges, 1)
    x(edges(e, 3)) = edgeConsensus(e);
end
end


function [relativeGap, relativeError] = evaluate_solution(x, data, blocks, ...
    xStar, fStar, floorValue)
relativeGap = max((joint_objective(x, data, blocks) - fStar) / ...
    max(abs(fStar), eps), floorValue);
relativeError = max(norm(x - xStar) / max(norm(xStar), eps), floorValue);
end


function value = joint_objective(x, data, blocks)
value = 0;
for i = 1:numel(blocks)
    residual = data.X{i} * x(blocks{i}) - data.y{i};
    value = value + 0.5 * (residual' * residual);
end
end


function indices = coordinate_indices(scope, coordinates)
[found, indices] = ismember(coordinates, scope);
if ~all(found)
    error('SixClientADMM:MissingCoordinate', ...
        'A factor coordinate is absent from the combined scope.');
end
end


function A = symmetrize(A)
A = 0.5 * (A + A');
end


function plot_communication(admm, cfg, outputDir)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 600, 430]);
semilogy(admm.communicatedScalarsHistory, admm.relativeParameterError, ...
    'LineWidth', 1.8, 'Color', [0.42, 0.22, 0.72]);
hold on;
yline(cfg.targetRelativeError, '--', 'Target accuracy', ...
    'Color', [0.35, 0.35, 0.35]);
grid on;
box on;
xlabel('Cumulative communicated scalars');
ylabel('Relative parameter error');
legend('Edge-consensus ADMM', 'Target accuracy', 'Location', 'northeast');
exportgraphics(fig, fullfile(outputDir, 'admm_convergence.png'), ...
    'Resolution', 300);
exportgraphics(fig, fullfile(outputDir, 'admm_convergence.pdf'), ...
    'ContentType', 'vector');
close(fig);
end
