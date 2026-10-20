function varargout = run_all_experiments()
%RUN_ALL_EXPERIMENTS Run three independent solvers and print one summary.
% Run directly, or use results = run_all_experiments() to retain details.

experimentDir = fileparts(mfilename('fullpath'));
previousDir = pwd;
restoreDir = onCleanup(@() cd(previousDir)); %#ok<NASGU>

% Capture individual reports; display one table and the evaluation rules.
cd(fullfile(experimentDir, 'NSMP'));
evalc('results.NSMP = run_nsmp_script();');
cd(fullfile(experimentDir, 'Gossip'));
evalc('results.Gossip = run_gossip();');
cd(fullfile(experimentDir, 'ADMM'));
evalc('results.ADMM = run_admm();');

method = {'NSMP'; 'Gossip'; 'ADMM'};
rounds = [results.NSMP.summary.rounds; ...
    results.Gossip.communicationSummary.roundsToTolerance; ...
    results.ADMM.summary.synchronousRoundsOrIterations];
messages = [results.NSMP.summary.messages; ...
    results.Gossip.communicationSummary.directedMessages; ...
    results.ADMM.summary.messages];
scalars = [results.NSMP.summary.scalars; ...
    results.Gossip.communicationSummary.communicatedScalars; ...
    results.ADMM.summary.communicatedScalars];
bytes = 8 * scalars;
results.summary = table(method, rounds, messages, scalars, bytes);
outputDir = fullfile(experimentDir, 'results');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
writetable(results.summary, fullfile(outputDir, 'communication_summary.csv'));

fprintf('\n%-12s %10s %12s %12s %14s\n', ...
    'Method', 'Rounds', 'Messages', 'Scalars', 'Bytes (F64)');
fprintf('%s\n', repmat('-', 1, 64));
for i = 1:numel(method)
    fprintf('%-12s %10g %12g %12g %14g\n', ...
        method{i}, rounds(i), messages(i), scalars(i), bytes(i));
end
fprintf('\nADMM: primal residual <= primal tolerance AND dual residual <= dual tolerance.\n');
fprintf('      abs_tol=%.0e, rel_tol=%.0e; maximum %d iterations.\n', ...
    results.ADMM.config.admmAbsoluteTolerance, ...
    results.ADMM.config.admmRelativeTolerance, ...
    results.ADMM.config.maxIterations);
fprintf('Gossip: relative objective gap <= %.0e (table reports the first qualifying iteration).\n', ...
    results.Gossip.config.convergenceTolerance);
fprintf('        Full convergence history runs for %d iterations.\n', ...
    results.Gossip.config.numRounds);
if nargout > 0
    varargout{1} = results;
end
end

function output = run_nsmp_script()
% Isolate the original script's clear command from the launcher workspace.
main_6_users;
output = simulationResults;
end
