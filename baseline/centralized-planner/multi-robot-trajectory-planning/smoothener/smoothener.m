%% Problem Specification

% Need 10 inputs. Definitions
%   k = number of waypoints in discrete plan
%   N = number of agents
%   T = number of agent types

%INPUTS: 
% paths: [dim k N] float
%   paths(:,:,i) k waypoints for agent i

% types: [N] int
%   types(i) number in {1...T} for agent i 

% conf_cylinders: [T,T,3] float
%   defines conflict geometry for every pair in T types
%   (i,j,1) separating radius of type i and type j
%   (i,j,2) type i must be this far above type j
%   (i,j,3) type i must be this far below type j

% obs_cylinders: [T,3] float
%   defines conflict geometry for every pair of T types
%   (i,j,1) separating radius of type i and obstacles
%   (i,j,2) agent i must be this far above obstacle
%   (i,j,3) agent i must be this far below obstacle

% deg: [1] int
%   polynomial degree

% cont: [1] int
%   derivative continuity

% timescale: [1] float
%   seconds per timestep in discrete plan

% iters: [1] int
%   number of smoothing iterations

% pp_obs_sep_fun: [1] func handle
%   obstacle separating function

%% Input Specification
clear; close all;
% r1_path = [0, 0,  3;...
%            0, 2,  3; ...
%            0, 4,  3];
% 
% r2_path = [4, 0,  3;...
%            4, 2,  3;...
%            4, 4,  3];
% 
% r3_path = [-4, 0,  3;...
%            -4, 2,  3; ...
%            -4, 4,  3];
% 
% r4_path = [0, 0,  7;...
%            0, 2,  7;...
%            0, 4,  7];
%        
% r5_path = [0, 0,  0;...
%            0, 2,  0; ...
%            0, 4,  0];

% ~~~~~~ USER INPUT ~~~~~~
PLOT = false;
ANIMATE = false;
OBSTACLES = true;

%example = 'crossing2';
%example = 'multitype';
%example = 'warehouse';
%example = 'swapMulti';
example = 'ground';
% ~~~~~~ deg,cont,timescale,iters input ~~~~~~
deg = 7;
cont = 1;%4;
timescale = 1.0;
iters = 8;
Neval = 32; %number of samples on pps separation

% ~~~~~~ END USER INPUT ~~~~~~~~~~~~

% ~~~~~~ Env file for octomap ~~~~~~
folder = '..';
map = strcat(folder, '/examples/', example, '/map.bt');

% ~~~~~~ STL file for plotting ~~~~~~
stl_file = strcat(folder, '/examples/', example, '/output/map.stl');

% ~~~~~~ paths Input ~~~~~~
schedule_file = strcat(folder, '/examples/', example, '/output/discreteSchedule.yaml');
[paths,names,typeNames] = read_schedule(schedule_file);
[dim, k, N] = size(paths);
nsteps = size(paths,2)-1;

% plot3(paths(1,:,1),paths(2,:,1),paths(3,:,1))
% hold on
% plot3(paths(1,:,2),paths(2,:,2),paths(3,:,2))

% ~~~~~ read types ~~~~~~
types_file = strcat(folder, '/examples/', example, '/types.yaml');
typesStruct = yaml.ReadYaml(types_file);

ntypes = size(typesStruct.agentTypes);
ntypes = ntypes(2);
locomotion = ones(ntypes, 1) * 3;

%Right now it is [rx,ry,rz] for ellipsoids
obs_cylinders = ones(ntypes,3);

% fill obs_cylinders and locomotion
for i=1:ntypes
    shape = typesStruct.agentTypes{1,i}.shape;
    if strcmp(shape.type, "cylinder")
        obs_cylinders(i,:) = [shape.radius, shape.radius, shape.height / 2];
    end
    if strcmp(shape.type, "sphere")
        obs_cylinders(i,:) = [shape.radius, shape.radius, shape.radius];
    end
    if contains(typesStruct.agentTypes{1,i}.type, "ground")
        locomotion(i) = 2;
    end
end

%obs_cylinders = obs_cylinders * 0.5;

% fill conf_cylinders
conf_cylinders = zeros(ntypes,ntypes,3);
for i=1:ntypes
    for j=1:ntypes
        type_i = typesStruct.agentTypes{1,i}.type;
        type_j = typesStruct.agentTypes{1,j}.type;
        for k=1:length(typesStruct.agentInteractions)
            interaction=typesStruct.agentInteractions{k};
            if strcmp(interaction.typeA, type_i) && strcmp(interaction.typeB, type_j)
                conf_cylinders(i,j,:) = [interaction.radius, interaction.below, interaction.above];
            end
            if strcmp(interaction.typeA, type_j) && strcmp(interaction.typeB, type_i)
                conf_cylinders(i,j,:) = [interaction.radius, interaction.above, interaction.below];
            end
        end
    end
end

% fill types
types = zeros(N, 1);

for n = 1:N
   for i=1:ntypes
     type = typesStruct.agentTypes{1,i}.type;
     if strcmp(typeNames{n}, type)
        types(n) = i;
     end
   end
end

%hack for ellipsoid input to octomap separation function
obs_ellipsoids = zeros(N,3);
for n = 1:N
    obs_ellipsoids(n,:) = obs_cylinders(types(n),:);
end

% ~~~~~~ pps Output ~~~~~~
outcsv = strcat(folder, '/examples/', example, '/output/pps/');

% ~~~~~~ bbox Input ~~~~~~
bbox = read_octomap_bbox_mex(map);
% bbox = [-5.5, 2.0;...
%         -3.0, 3.5;...
%          0.0, 2.5];
     
% bbbuffer = 10;
% bbox = [min(min(paths(1,:,:))) - bbbuffer,max(max(paths(1,:,:)) + bbbuffer);...
%         min(min(paths(2,:,:))) - bbbuffer,max(max(paths(2,:,:)) + bbbuffer);...
%         min(min(paths(3,:,:))) - bbbuffer,max(max(paths(3,:,:)) + bbbuffer)];
    
% ~~~~~~ pp obstacle separation function ~~~~~~
if (OBSTACLES)
    pp_obs_sep_fun = @(poly,elip) pp_obs_sep_octomap(poly,elip,map);
else
    pp_obs_sep_fun = @pp_obs_sep_none;
end


%% Smoothener
[dim, k, N] = size(paths);

% for a reasonable problem, cost should converge after ~5 iterations.
assert(iters >= 1);
assert(iters <= 20);

% outputs
all_pps = cell(iters, N);
all_costs = zeros(iters, N);
all_corridors = cell(iters, N);

% piecewise linear (physically impossible) pps of path
% for input to pp-vs-octree obstacle hyperplane function
pps = path_linear_pps(paths, timescale, deg + 1);

for iter=1:iters
    fprintf('iteration %d of %d...\n', iter, iters);
    tic;
    if iter==1
        % first iteration: decompose by segments
        [A, b] = robot_hp_waypoints(paths, types, conf_cylinders);
    else
        % continuing iteration: decompose by pps
        [A, b] = robot_hp_pps(pps,types,conf_cylinders,Neval);
    end

    if iter > 1
        for irobot=1:N
            paths(:,:,irobot) = ppval(pps{irobot}, pps{irobot}.breaks);
        end
    end

    hs = pp_obs_sep_fun(pps, obs_ellipsoids);

    t_hyperplanes = toc;
    fprintf('hyperplanes: %f sec\n', t_hyperplanes);

    % solve the independent spline trajectory optimization problems.
    tic;
    pps = cell(1,N);
    iter_costs = zeros(1,N);
    
    % parfor
    parfor j=1:N
        fprintf(' agent %d of %d...\n', j, N);
        lb = bbox(:,1) + [obs_cylinders(types(j),1);obs_cylinders(types(j),1);obs_cylinders(types(j),3)];
        ub = bbox(:,2) - [obs_cylinders(types(j),1);obs_cylinders(types(j),1);obs_cylinders(types(j),2)];

        hs_slice = squeeze(hs(j,:));
        step_n_faces = cellfun(@(a) size(a, 1), hs_slice);
        assert(length(step_n_faces) == (k-1));
        max_n_faces = max(step_n_faces(:));

        Aobs = nan(dim, max_n_faces, k-1);
        bobs = nan(max_n_faces, k-1);
        Arobots = A(:,j,:,:); %squeeze(A(:,j,:,:));
        brobots = b(j,:,:);%squeeze(b(j,:,:));
        for i=1:(k-1)
            n_faces = step_n_faces(i);
            hs_slice_step = hs_slice{i};
            assert(size(hs_slice_step, 2) == 4);
            Aobs(:,1:n_faces,i) = hs_slice_step(:,1:3)';
            bobs(1:n_faces,i) = hs_slice_step(:,4);
        end
        [pps{j}, iter_costs(j)] = corridor_trajectory_optimize(...
            Arobots, brobots, ...
            Aobs, bobs, ...
            lb, ub,...
            paths(:,:,j), deg, cont, timescale, obs_cylinders(types(j),:),j,iter,locomotion(types(j)));%[0.2 0.2 0.4], [0.2 0.2 0.2]);%
        s = [];
        s.Arobots = Arobots;
        s.brobots = brobots;
        s.Aobs = Aobs;
        s.bobs = bobs;
        all_corridors{iter,j} = s;
    end
    
    if isnan(sum(iter_costs))
        pps = all_pps(iter-1,:);
        break;
    end
    
    t_splines = toc;
    fprintf('splines: %f sec\n', t_splines);
    fprintf('total: %f sec\n', t_hyperplanes + t_splines);
    fprintf('cost: %f\n', sum(iter_costs));
    all_costs(iter,:) = iter_costs;
    all_pps(iter,:) = pps;
end

%% Save

% ~~~~~~ pps Output ~~~~~~
for n = 1:N
    pp2csv(pps{n}, [outcsv,names{n},'.csv'])
end

%% Plot?
if (PLOT)
    close all;
    %sample trajectories
    duration = pps{1}.breaks(end);
    sr = 0.05;
    t = 0:sr:duration;
    trajplots = cell(N,1);
    robcolors = cell(N,1);
    for i=1:size(paths,3)
        trajplots{i} = ppval(pps{i}, t)';
    end

    %plot path + trajectory
    for i=1:size(paths,3)	% [14,16,17,31]		
        h = plot3n(paths(:,:,i));
        hold on;
        robcolor{i} = get(h, 'color');
        h = plot3n(trajplots{i}', 'color', robcolor{i}, 'LineWidth', 3);
    end

    %plot map
    fv = stlread(stl_file);
        patch(fv, ...
                'FaceColor', [0.4 0.4 0.4], 'EdgeColor', 'none', ...
                'SpecularStrength', 0.1, 'AmbientStrength', 0.5, 'facealpha',0.6);

    %set axis props
    ax = gca;
    xlabel('x')
    ylabel('y')
    zlabel('z')
    ax.Projection = 'perspective';
    ax.DataAspectRatioMode = 'manual';
    ax.DataAspectRatio = [1 1 1];
    axis vis3d;
end
% Animate? Using ellipsoid atm

if (ANIMATE)
    %in = input('Animate: ')

    %robot hull
    robots = cell(N,1); %{xyz,handle}
    for n=1:N
        [sx,sy,sz] = ellipsoid(paths(1,1,n),paths(2,1,n),paths(3,1,n),obs_cylinders(types(n),1),obs_cylinders(types(n),2),obs_cylinders(types(n),3));
        r_color = zeros(size(sz,1),size(sz,2),3);
        r_color(:,:,1) = repmat(linspace((robcolor{n}(1)/5),robcolor{n}(1),size(sx,1))',1,size(sx,2)); % red
        r_color(:,:,2) = repmat(linspace((robcolor{n}(2)/5),robcolor{n}(2),size(sy,1))',1,size(sy,2)); % green
        r_color(:,:,3) = repmat(linspace((robcolor{n}(3)/5),robcolor{n}(3),size(sz,1))',1,size(sz,2));
        robots{n} = surf(ax,sx,sy,sz,r_color,'EdgeColor', 'none');
    end

    for time = 2:numel(t)
        for n = 1:N
            dx = trajplots{n}(time,1) - trajplots{n}(time-1,1);
            dy = trajplots{n}(time,2) - trajplots{n}(time-1,2);
            dz = trajplots{n}(time,3) - trajplots{n}(time-1,3);
            robots{n}.XData = robots{n}.XData + dx;
            robots{n}.YData = robots{n}.YData + dy;
            robots{n}.ZData = robots{n}.ZData + dz;
        end
        pause(0.033);
    end
end

if (PLOT)
hold off;
end


