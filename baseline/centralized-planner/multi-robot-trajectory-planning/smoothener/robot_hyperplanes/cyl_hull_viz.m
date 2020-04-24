clear; close all

%% Robots / Path def

% r1r2_cyl.above = 1;
% r1r2_cyl.below = 2;
% r1r2_cyl.radius = 1;
% r1r2_cyl.height = r1r2_cyl.below + r1r2_cyl.above;

r1r2_cyl = [1,1,2]; %radius,above,below, seperate r1 from r2
r2r1_cyl = [1,2,1];

% Paths
% r1_path = [0, -1, 0;...
%            2,0,0; ...
%            2,2,2; ...
%            2,5,-1; ...
%            1,7,-3; ...
%            -2,8,1];
% 
% r2_path = [-5,0,0;...
%            -6,2,1;...
%            -7,3,2;...
%            -6,4,3;...
%            -4,6,2;...
%            -5,8,-1];

r1_path = [4, 1,  0;...
           3, 2,  1; ...
           2, 3,  2; ...
           1, 4,  3; ...
           0, 5,  4; ...
           -1, 6,  5];

r2_path = [-4, 1,  0;...
           -3, 2,  -1;...
           -2, 3,  -2;...
           -1, 4,  -3;...
           -0, 5,  -4;...
           1, 6,  -5];

r1_path = [0, 0,  3;...
           0, 2,  3; ...
           0, 4,  3];

r2_path = [3, 0,  3;...
           3, 2,  3;...
           3, 4,  3];

nsteps = size(r1_path,1) - 1;



%% compute hyperplanes
%pepare data in expected format
paths = zeros(3,size(r1_path,1),2);
paths(:,:,1) = r1_path';
paths(:,:,2) = r2_path';
types = [1,2];

cylinders = zeros(2,2,3);
dummy_cyl = [0.1,0.1,0.1];
cylinders(1,2,:) = r1r2_cyl;
cylinders(2,1,:) = r2r1_cyl;
cylinders(1,1,:) = dummy_cyl;
cylinders(2,2,:) = dummy_cyl;

%hp computation for each timestep;
[A,b] = robot_hp_waypoints(paths,types,cylinders);
% A = -A;
% b = -b;

%hp computations assuming waypoints are trajectory samples
[r1_verts] = swept_cyl_verts(r1r2_cyl,r1_path);
[r2_verts] = swept_cyl_verts(r2r1_cyl,r2_path);

%constraint for r1 (i)
r1_cloud = [r2_verts;r1_path];
r1_labels = [ones(size(r2_verts,1),1);-1*ones(size(r1_path,1),1)];
SVM = svmtrain(r1_labels,r1_cloud,'-q -t 0');
suppVecs = r1_cloud(SVM.sv_indices,:);
w = SVM.sv_coef' * suppVecs;
r1_A = -(w/norm(w));
r1_B = (SVM.rho)/norm(w);

%constraint for r2 (j)
r2_cloud = [r1_verts;r2_path];
r2_labels = [ones(size(r1_verts,1),1);-1*ones(size(r2_path,1),1)];
SVM = svmtrain(r2_labels,r2_cloud,'-q -t 0');
suppVecs = r2_cloud(SVM.sv_indices,:);
w = SVM.sv_coef' * suppVecs;
r2_A = -(w/norm(w));
r2_B = (SVM.rho)/norm(w);
%% Swept Volume hull vertices

% ~~~~~~~ All Verts ~~~~~~~
% hull vertices
scatter3(r1_verts(:,1),r1_verts(:,2),r1_verts(:,3),'go')
hold on;
scatter3(r2_verts(:,1),r2_verts(:,2),r2_verts(:,3),'ro')
ax = gca;

% ~~~~~~~ Hull at each step ~~~~~~~
hull_hands = cell(nsteps,2);
for t = 1:nsteps
    DT = delaunayTriangulation(r1_verts(((t-1)*16+1):(t*16),:));
    [K,v] = convexHull(DT);
    hull_hands{t,1} = trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),'FaceAlpha',0.2,'FaceColor',[0.2,0.5,0.5],'edgecolor','none','Visible','off');
    
    DT = delaunayTriangulation(r2_verts(((t-1)*16+1):(t*16),:));
    [K,v] = convexHull(DT);
    hull_hands{t,2} = trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),'FaceAlpha',0.2,'FaceColor',[0.5,0.2,0.5],'edgecolor','none','Visible','off');
end

% ~~~~~~~ Cylinders at waypoints ~~~~~~~
%   initial cylinder and color scheme
% [r1_c1x, r1_c1y, r1_c1z] = robot_cylinder(r1r2_cyl,r1_path(1,:));
% r1_color = zeros(size(r1_c1z),'like',r1_c1z);
% r1_color(:,:,1) = 0.1; % red
% r1_color(:,:,2) = 0.5; % green
% r1_color(:,:,3) = 0.2; % blue
% 
% surf(ax,r1_c1x, r1_c1y, r1_c1z,r1_color);
% % other cylinders
% for p = 2:size(r1_path,1)
%     [r1_c2x, r1_c2y, r1_c2z] = robot_cylinder(r1r2_cyl,r1_path(p,:));
%     surf(ax,r1_c2x, r1_c2y, r1_c2z,r1_color);
% end

% ~~~~~~~ Paths ~~~~~~~
plot3(r1_path(:,1),r1_path(:,2),r1_path(:,3),'-go', ...
    'LineWidth', 3);
plot3(r2_path(:,1),r2_path(:,2),r2_path(:,3),'-ro', ...
    'LineWidth', 3);


xlabel('x')
ylabel('y')
zlabel('z')
ax.Projection = 'perspective';
ax.DataAspectRatioMode = 'manual';
ax.DataAspectRatio = [1 1 1];
axis vis3d;

% ~~~~~~~ Hyperplanes "Animation" ~~~~~~~
%bounds for hp surf
xrange = [-6,6];
yrange = [-2,9];
zrange = [-5,5];
xyzstep = 1;
%timestep hyperplanes
for t = 1:nsteps
    fprintf('Step: %d\n',t);
    
    k = input(sprintf('   sep: %d, %d: ',1,2));
    hull_hands{t,1}.Visible = 'on';
    hull_hands{t,2}.Visible = 'on';
    
    if (t>1) %hide plots from last step
        hull_hands{t-1,1}.Visible = 'off';
        hull_hands{t-1,2}.Visible = 'off';
        q1last.Visible = 'off';
        s1last.Visible = 'off';
        q2last.Visible = 'off';
        s2last.Visible = 'off';
        p1last.Visible = 'off';
        p2last.Visible = 'off';
    end
    
    %highlight segment
    p1last = plot3(r1_path(t:(t+1),1),r1_path(t:(t+1),2),r1_path(t:(t+1),3),'-go', 'LineWidth', 8);
    p2last = plot3(r2_path(t:(t+1),1),r2_path(t:(t+1),2),r2_path(t:(t+1),3),'-ro', 'LineWidth', 8);
    
    
    %hull for robot 2 (j)
%     hull_hands{t,2}.Visible = 'on';
    %Visualize constraint for agent i (1) interacting with j (2)
    Astep = A(:,1,2,t)
    bstep = b(1,2,t)
    [hpx,hpy,hpz] = hyperplane_surf(Astep,bstep,xrange,yrange,zrange,xyzstep);
    u = Astep(1)*ones(size(hpx,1),size(hpx,2));
    v = Astep(2)*ones(size(hpx,1),size(hpx,2));
    w = Astep(3)*ones(size(hpx,1),size(hpx,2));
    q1last = quiver3(ax,hpx,hpy,hpz,u,v,w,0.5);
    s1last = surf(ax,hpx,hpy,hpz,'FaceAlpha',0.5,'FaceColor',[0.1,0.5,0.3],'edgecolor','none');
    
%     k = input(sprintf('   sep: %d, %d: ',2,1));
%     hull_hands{t,2}.Visible = 'off';
%     hull_hands{t,1}.Visible = 'on';
    %Visualize constraint for agent j interacting with i
    Astep = A(:,2,1,t)
    bstep = b(2,1,t)
    [hpx,hpy,hpz] = hyperplane_surf(Astep,bstep,xrange,yrange,zrange,xyzstep);
    u = Astep(1)*ones(size(hpx,1),size(hpx,2));
    v = Astep(2)*ones(size(hpx,1),size(hpx,2));
    w = Astep(3)*ones(size(hpx,1),size(hpx,2));
    q2last = quiver3(ax,hpx,hpy,hpz,u,v,w,0.5);
    s2last = surf(ax,hpx,hpy,hpz,'FaceAlpha',0.5,'FaceColor',[0.4,0.1,0.4],'edgecolor','none');

end
%full trajectory constraint
k = input('Full traj: ');
for t = 1:nsteps
    hull_hands{t,1}.Visible = 'on';
    hull_hands{t,2}.Visible = 'on';
end
q1last.Visible = 'off';
s1last.Visible = 'off';
q2last.Visible = 'off';
s2last.Visible = 'off';
p1last.Visible = 'off';
p2last.Visible = 'off';

%r1 (i) Constraint
[hpx,hpy,hpz] = hyperplane_surf(r1_A,r1_B,xrange,yrange,zrange,xyzstep);
u = r1_A(1)*ones(size(hpx,1),size(hpx,2));
v = r1_A(2)*ones(size(hpx,1),size(hpx,2));
w = r1_A(3)*ones(size(hpx,1),size(hpx,2));
quiver3(ax,hpx,hpy,hpz,u,v,w,0.5);
surf(ax,hpx,hpy,hpz,'FaceAlpha',0.5,'FaceColor',[0.1,0.5,0.3],'edgecolor','none');

%r2 (j) Constraint
[hpx,hpy,hpz] = hyperplane_surf(r2_A,r2_B,xrange,yrange,zrange,xyzstep);
u = r2_A(1)*ones(size(hpx,1),size(hpx,2));
v = r2_A(2)*ones(size(hpx,1),size(hpx,2));
w = r2_A(3)*ones(size(hpx,1),size(hpx,2));
quiver3(ax,hpx,hpy,hpz,u,v,w,0.5);
surf(ax,hpx,hpy,hpz,'FaceAlpha',0.5,'FaceColor',[0.4,0.1,0.4],'edgecolor','none');

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