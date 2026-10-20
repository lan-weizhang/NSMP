clear
clc
close all
rng(0);
M = 6;
samples = 100;
sigma = 10;

X_1 = randn(samples,2);
a_1 = randn(2,1);
y_1 = X_1*a_1 + sigma*randn(samples,1);

X_2 = randn(samples,2);
a_2 = randn(2,1);
y_2 = X_2*a_2 + sigma*randn(samples,1);

X_3 = randn(samples,4);
a_3 = randn(4,1);
a_3(1) = a_1(2);
a_3(2) = a_2(2);
y_3 = X_3*a_3 + sigma*randn(samples,1);


X_4 = randn(samples,2);
a_4 = randn(2,1);
y_4 = X_4*a_4 + sigma*randn(samples,1);


X_5 = randn(samples,3);
a_5 = randn(3,1);
a_5(1) = a_4(2);
y_5 = X_5*a_5 + sigma*randn(samples,1);


X_6 = randn(samples,3);
a_6 = randn(3,1);
a_6(1) = a_3(4);
a_6(2) = a_5(3);
y_6 = X_6*a_6 + sigma*randn(samples,1);


%% accurate calculation
% left a_13
a_1_est_local = (X_1'*X_1)^-1*X_1'*y_1;
P_1_temp = X_1(:,1);
P_1 = eye(samples)-X_1(:,1)*(X_1(:,1)'*X_1(:,1))^-1*X_1(:,1)';
R_1 = P_1*y_1;
X_12 = P_1*X_1(:,2);

% left a_23
a_2_est_local = (X_2'*X_2)^-1*X_2'*y_2;
P_2_temp = X_2(:,1);
P_2 = eye(samples)-P_2_temp*(P_2_temp'*P_2_temp)^-1*P_2_temp'  ;
R_2 = P_2*y_2;
X_22 = P_2*X_2(:,2);


%% left a_36
a_3_est_local = (X_3'*X_3)^-1*X_3'*y_3;
% enlarge matrix
ZM = zeros(samples,1);
P_3_temp =[X_3(:, 1:3); X_12, ZM, ZM;ZM, X_22,ZM];
P_3 = eye(length(P_3_temp))-P_3_temp*(P_3_temp'*P_3_temp)^-1*P_3_temp';

R_3 = P_3(:,1:samples)*y_3 +  P_3(:,samples+1:2*samples)*R_1 +P_3(:,2*samples+1:3*samples)*R_2  ;
X_33 = P_3(:,1:samples)*X_3(:,4);


%% left a_45
a_4_est_local = (X_4'*X_4)^-1*X_4'*y_4;
P_4_temp = X_4(:,1);
P_4 = eye(samples)-X_4(:,1)*(X_4(:,1)'*X_4(:,1))^-1*X_4(:,1)';
R_4 = P_4*y_4;
X_44 = P_4*X_4(:,2);

%% left a_56
a_5_est_local = (X_5'*X_5)^-1*X_5'*y_5;
% enlarge matrix
ZM = zeros(samples,1);
P_5_temp =[X_5(:, 1:2); X_44, ZM];
P_5 = eye(length(P_5_temp))-P_5_temp*(P_5_temp'*P_5_temp)^-1*P_5_temp';

R_5 = P_5(:,1:samples)*y_5 +  P_5(:,samples+1:2*samples)*R_4 ;
X_55 = P_5(:,1:samples)*X_5(:,3);

%% root node
a_6_est_local = (X_6'*X_6)^-1*X_6'*y_6;
% enlarge matrix
ZM_3 = zeros(3*samples,1);
ZM_2 = zeros(2*samples,1);
P_6_O =[X_6(:, 1:3); X_33, ZM_3, ZM_3;ZM_2, X_55,ZM_2];
R_6 = [y_6 ; R_3 ;R_5]  ;

a_6_est_global = (P_6_O'*P_6_O)^-1*P_6_O'*R_6 ;

norm(a_6_est_global-a_6);
norm(a_6_est_local-a_6);

%% node 5
a_5_est_global = zeros(3,1);
a_5_est_global(3) = a_6_est_global(2); 
a_5_est_global(1:2) = (P_5_temp'*P_5_temp)^-1*P_5_temp'*[y_5-X_5(:, 3)*a_5_est_global(3);R_4];

norm(a_5_est_global-a_5);
norm(a_5_est_local-a_5);

%% node 3
a_3_est_global = zeros(4,1);
a_3_est_global(4) = a_6_est_global(1); 
a_3_est_global(1:3) = (P_3_temp'*P_3_temp)^-1*P_3_temp'*[y_3-X_3(:, 4)*a_3_est_global(4);R_1;R_2];

norm(a_3_est_global-a_3);
norm(a_3_est_local-a_3);

%% node 2
a_2_est_global = zeros(2,1);
a_2_est_global(2) = a_3_est_global(2); 
a_2_est_global(1) = (P_2_temp'*P_2_temp)^-1*P_2_temp'*[y_2-X_2(:, 2)*a_2_est_global(2)];

norm(a_2_est_global-a_2);
norm(a_2_est_local-a_2);


%% node 1
a_1_est_global = zeros(2,1);
a_1_est_global(2) = a_3_est_global(1); 
a_1_est_global(1) = (P_1_temp'*P_1_temp)^-1*P_1_temp'*[y_1-X_1(:, 2)*a_1_est_global(2)];

norm(a_1_est_global-a_1);
norm(a_1_est_local-a_1);


%% node 4
a_4_est_global = zeros(2,1);
a_4_est_global(2) = a_5_est_global(1); 
a_4_est_global(1) = (P_4_temp'*P_4_temp)^-1*P_4_temp'*[y_4-X_4(:, 2)*a_4_est_global(2)];

norm(a_4_est_global-a_4);
norm(a_4_est_local-a_4);


%% union

vector_global = [a_1_est_global;a_2_est_global;a_3_est_global(3:4);a_4_est_global;a_5_est_global(2:3);a_6_est_global(3)];

%% center optimization
y_global = [y_1;y_2;y_3;y_4;y_5;y_6];

A_global = zeros(length(y_global),11);
A_global(1:samples,1:2) = X_1;
A_global(samples+1:2*samples,3:4) = X_2;
A_global(2*samples+1:3*samples,[2,4,5,6]) = X_3;
A_global(3*samples+1:4*samples,[7,8]) = X_4;
A_global(4*samples+1:5*samples,[8,9,10]) = X_5;
A_global(5*samples+1:6*samples,[6,10,11]) = X_6;

vector_center = (A_global'*A_global)^-1*A_global'*y_global;

%% quantitative comparison
rel_err = norm(vector_global - vector_center) / max(norm(vector_center), eps);
obj_global = 0.5 * norm(y_global - A_global * vector_global)^2;
obj_center = 0.5 * norm(y_global - A_global * vector_center)^2;
obj_gap = obj_global - obj_center;

fprintf('\n===== Two-pass vs centralized =====\n');
fprintf('||x_mp - x_center||_2      = %.4e\n', norm(vector_global - vector_center));
fprintf('relative error             = %.4e\n', rel_err);
fprintf('objective(message)         = %.6e\n', obj_global);
fprintf('objective(centralized)     = %.6e\n', obj_center);
fprintf('objective gap              = %.4e\n', obj_gap);

%% figure for paper
% Styling only: keep the simulation and both solution vectors unchanged.
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, 'results');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fig = figure('Color','w','Visible','off', ...
    'Units','inches','Position',[1,1,3.35,3.20], ...
    'Renderer','painters');
ax = axes(fig, 'Units','normalized','Position',[0.19,0.18,0.77,0.78]);
hold(ax, 'on');

vmin = min([vector_center; vector_global]);
vmax = max([vector_center; vector_global]);
padding = 0.015 * (vmax - vmin);
limits = [vmin-padding, vmax+padding];

% Draw the equality reference first, so it cannot obscure the observations.
plot(ax, limits, limits, '--', 'Color',[0.35,0.38,0.42], ...
    'LineWidth',0.9);
points = plot(ax, vector_center, vector_global, 'o', ...
    'LineStyle','none','MarkerSize',5.2,'LineWidth',1.05, ...
    'MarkerEdgeColor',[0.12,0.35,0.56],'MarkerFaceColor','w');

set(ax, 'FontName','Times New Roman','FontSize',10, ...
    'LineWidth',0.75,'XColor',[0.15,0.15,0.15], ...
    'YColor',[0.15,0.15,0.15],'TickDir','in', ...
    'TickLength',[0.012,0.012],'Box','on', ...
    'XGrid','on','YGrid','on','GridColor',[0.4,0.4,0.4], ...
    'GridAlpha',0.14,'GridLineStyle','-','Layer','bottom');
axis(ax, 'equal');
xlim(ax, limits);
ylim(ax, limits);
ticks = ceil(vmin*2)/2 : 0.5 : floor(vmax*2)/2;
xticks(ax, ticks);
yticks(ax, ticks);
xlabel(ax, 'Centralized solution', 'FontName','Times New Roman', ...
    'FontSize',11);
ylabel(ax, 'NSMP solution', 'FontName','Times New Roman', ...
    'FontSize',11);

% Preserve the original reported values; only change typography.
rel_parts = strsplit(sprintf('%.2e', rel_err), 'e');
rel_label = sprintf('RelErr = %s \\times 10^{%d}', ...
    rel_parts{1}, str2double(rel_parts{2}));
text(ax, 0.035,0.96, rel_label, 'Units','normalized', ...
    'FontName','Times New Roman','FontSize',9.5, ...
    'Interpreter','tex','VerticalAlignment','top', ...
    'Color',[0.15,0.15,0.15], ...
    'BackgroundColor','w','EdgeColor',[0.78,0.78,0.78], ...
    'LineWidth',0.6,'Margin',3);
text(ax, 0.035,0.80, sprintf('ObjGap = %.2g', obj_gap), ...
    'Units','normalized','FontName','Times New Roman','FontSize',9.5, ...
    'Interpreter','tex','VerticalAlignment','top', ...
    'Color',[0.15,0.15,0.15], ...
    'BackgroundColor','w','EdgeColor',[0.78,0.78,0.78], ...
    'LineWidth',0.6,'Margin',3);

% Verify that the plotted coordinates are the unmodified computed values.
assert(isequal(points.XData(:), vector_center(:)));
assert(isequal(points.YData(:), vector_global(:)));
assert(numel(points.XData) == 11);
save(fullfile(out_dir, 'effectiveness_comparison_data.mat'), ...
    'vector_center','vector_global','rel_err','obj_gap','obj_center','obj_global');
exportgraphics(fig, fullfile(out_dir, 'effectiveness_comparison.png'), ...
    'Resolution',450,'BackgroundColor','white');
exportgraphics(fig, fullfile(out_dir, 'effectiveness_comparison.pdf'), ...
    'ContentType','vector','BackgroundColor','white');
fprintf('Figure saved to:\n  %s\n', fullfile(out_dir, 'effectiveness_comparison.png'));
fprintf('  %s\n', fullfile(out_dir, 'effectiveness_comparison.pdf'));


%% NSMP communication for the paper's six-client tree (height 2).
numTreeEdges = M - 1;
treeHeight = 2;
% Each edge sends three quadratic coefficients upward and one scalar down.
method = {'NSMP'};
rounds = 2 * treeHeight;
messages = 2 * numTreeEdges;
scalars = 4 * numTreeEdges;
bytes = 8 * scalars;
summary = table(method, rounds, messages, scalars, bytes);
writetable(summary, fullfile(out_dir, 'communication_summary.csv'));
simulationResults.solution = vector_global;
simulationResults.relativeParameterError = rel_err;
simulationResults.objectiveGap = obj_gap;
simulationResults.summary = summary;
simulationResults.outputDir = out_dir;
