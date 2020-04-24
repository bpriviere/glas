clear; close all

%% Robots / Path def

% r1r2_cyl.above = 1;
% r1r2_cyl.below = 2;
% r1r2_cyl.radius = 1;
% r1r2_cyl.height = r1r2_cyl.below + r1r2_cyl.above;

% ~~~~~~ paths Input ~~~~~~
paths = read_schedule('./examples/swap4/discreteSchedule.json');
[dim, k, N] = size(paths);
nsteps = size(paths,2)-1;

% ~~~~~~ types Input ~~~~~~
%1 = small, 2 = large for swap4
ntypes = 2;
types = [1;1;2;2];

% types = [1;2];
% ~~~~~~ conf_cylinders Input ~~~~~~
conf_cylinders = zeros(ntypes,ntypes,3);
%cylinders(i,j,1) = radius type i must stay away from type j
%cylinders(i,j,2) = radius type i must stay above type j
%cylinders(i,j,3) = radius type i must stay below type j

conf_cylinders(1,2,:) = [0.20,0.30,0.60];
conf_cylinders(2,1,:) = [0.20,0.60,0.30];

conf_cylinders(1,1,:) = [0.15,0.30,0.30];
conf_cylinders(2,2,:) = [0.25,0.50,0.50];

conf_cylinders = conf_cylinders*0.1;



%% compute hyperplanes
%pepare data in expected format

%hp computation for each timestep;
[A,b] = robot_hp_waypoints(paths,types,conf_cylinders);

%% Plot

% ~~~~~~~ Paths ~~~~~~~
% plot3(r1_path(:,1),r1_path(:,2),r1_path(:,3),'-ko', ...
%     'LineWidth', 3);
% hold on;
% ax = gca;
% plot3(r2_path(:,1),r2_path(:,2),r2_path(:,3),'-ro', ...
%     'LineWidth', 3);
% plot3(r3_path(:,1),r3_path(:,2),r3_path(:,3),'-go', ...
%     'LineWidth', 3);
% plot3(r4_path(:,1),r4_path(:,2),r4_path(:,3),'-bo', ...
%     'LineWidth', 3);
% plot3(r5_path(:,1),r5_path(:,2),r5_path(:,3),'-co', ...
%     'LineWidth', 3);

for n = 1:N
    plot3(paths(1,:,n),paths(2,:,n),paths(3,:,n),'LineWidth',3);
    hold on;
end
ax = gca;
xlabel('x')
ylabel('y')
zlabel('z')
ax.Projection = 'perspective';
ax.DataAspectRatioMode = 'manual';
ax.DataAspectRatio = [1 1 1];
axis vis3d;

% ~~~~~~~ Hyperplanes "Animation" ~~~~~~~
%bounds for hp surf
xrange = [-5,5];
yrange = [-5,5];
zrange = [-1,3];
xyzstep = 2;
%timestep hyperplanes
for t = 1:nsteps
    fprintf('Step: %d\n',t);
    
    %for each robot plot all shp constraints
    for i = 1:size(paths,3)
        hphands = cell(size(paths,3)-1,2);
        c = 1;
        for j = 1:size(paths,3)
            if j ~= i
                Astep = A(:,i,j,t);
                bstep = b(i,j,t);
                [hpx,hpy,hpz] = hyperplane_surf(-Astep,bstep,xrange,yrange,zrange,xyzstep);
                u = -Astep(1)*ones(size(hpx,1),size(hpx,2));
                v = -Astep(2)*ones(size(hpx,1),size(hpx,2));
                w = -Astep(3)*ones(size(hpx,1),size(hpx,2));
                hphands{c,1} = quiver3(ax,hpx,hpy,hpz,u,v,w,0.5);
                hphands{c,2} = surf(ax,hpx,hpy,hpz,'FaceAlpha',0.5,'FaceColor',[0.1,0.5,0.3],'edgecolor','none');
                c = c + 1;
            end
        end
        k = input('next: ');
        for q = 1:size(hphands,1)
            hphands{q,1}.Visible = 'off';
            hphands{q,2}.Visible = 'off';
        end
    end
    
end

hold off;

%% Helpers
function [x,y,z] = robot_cylinder(rcyl,pos)
    
    %cylinder radius r, height 1 with bottom at origin
    [cx, cy, cz] = cylinder(rcyl(1));
    
    %translate xy, scale and translate z
    x = cx + pos(1);
    y = cy + pos(2);
    z = (cz * (rcyl(2)+rcyl(3))) + pos(3) - rcyl(3);

end