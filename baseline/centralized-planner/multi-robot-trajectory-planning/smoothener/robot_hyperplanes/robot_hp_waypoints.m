function [ A, b ] = robot_hp_waypoints( paths, types,cylinders )
%ROBOT_HP_WAYPOINTS Calculates robot seperating hyperplanes
%INPUT:
%   paths: [3, #waypoints, #robots]
%       paths(:,:,n) are the waypoints for robot n
%   types: [n]
%       types(n) is the type of robot traversing paths(:,:,n).
%       Integer in range [1,#types]
%   cylinders: [#types,#types, 3]
%       cylinders(a,b,1) radius of cyl needed to separate type 'a' from 'b'
%       cylinders(a,b,2) height 'a' needs to be above 'b'
%       cylinders(a,b,3) height 'a' needs to be below 'b'
%OUTPUT:
%   A: [DIM x NROBOTS x NROBOTS x (NPTS - 1)] array of 
%      hyperplane normal vectors for each robot-robot interaction
%      at each segment.
%   b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b
%NOTES:
%   currently only works for 3d paths / ellipsoids

Nrob = size(paths,3); % #robots
Nsteps = size(paths,2) -1; % #time steps

%initialize output structures
A = nan(3,Nrob,Nrob,Nsteps);
b = nan(Nrob,Nrob,Nsteps);

hullRes = 16; %nsides per polytope approx.

%parfor each timestep
for step = 1:Nsteps
    %private renames for parfor
    stepA = nan(3,Nrob,Nrob);
    stepb = nan(Nrob,Nrob);
    
    %stepVerts = hullVerts(:,:,:,step);
    %for every pair of robots
    for i = 1:Nrob
        for j = (i+1):Nrob
            %SHP constraint for j
            %compute conflict hull from i's perspective
            %   j's path must stay out of this hull
            [hull] = swept_cyl_verts_2(cylinders(types(j),types(i),:),...
                                     [paths(:,step,i)';paths(:,step+1,i)'], hullRes);
                                 
            %vertex cloud for hull + waypoints for agent j
            pairCloud = [hull; paths(:,step,j)';paths(:,step+1,j)'];
            
            %labels for cloud, 1 for robot i, -1 for j
            labels = [ones(size(hull,1),1);-1;-1];
            
            %train svm to get hyperplane
            SVM = svmtrain(labels,pairCloud,'-c 900000 -q -t 0');
            %extract params
            suppVecs = pairCloud(SVM.sv_indices,:);
            w = SVM.sv_coef' * suppVecs;
            normw = norm(w);
            currA = w/normw;
            currb = (SVM.rho)/normw;
            
            stepA(:,j,i) = currA;
            stepb(j,i) = currb;
            
            %sanity check for inseperable case
            suppDists = suppVecs * currA' - currb;
            suppLab = labels(SVM.sv_indices,:);
            if any(suppLab~=sign(suppDists))
                warning(sprintf('Robots (%d,%d) paths conflict at step %d',j,i,step));
                plot_constraints([paths(:,step,i),paths(:,step+1,i)],...
                                [paths(:,step,j),paths(:,step+1,j)],...
                                hull,currA,currb);
                i_green = [paths(:,step,i)';paths(:,step+1,i)']
                j_red = [paths(:,step,j)';paths(:,step+1,j)']
                do = input('continue: ');
            end
            
            %SHP constraint for i
            %compute conflict hull from j's perspective
            %   i's path must stay out of this hull
            [hull] = swept_cyl_verts_2(cylinders(types(i),types(j),:),...
                                     [paths(:,step,j)';paths(:,step+1,j)'], hullRes);

            %vertex cloud for hull + waypoints for agent i
            pairCloud = [hull; paths(:,step,i)';paths(:,step+1,i)'];
            
            %labels for cloud, 1 for robot j, -1 for i
            labels = [ones(size(hull,1),1);-1;-1];

            %train svm to get hyperplane
            SVM = svmtrain(labels,pairCloud,'-c 900000 -q -t 0');

            %hyperplane params
            suppVecs = pairCloud(SVM.sv_indices,:);
            w = SVM.sv_coef' * suppVecs;
            normw = norm(w);
            currA = w/normw;
            currb = (SVM.rho)/normw;
            stepA(:,i,j) = currA;
            stepb(i,j) = currb;
            
            %sanity check for inseperable case
            suppDists = suppVecs * currA' - currb;
            suppLab = labels(SVM.sv_indices,:);
            if any(suppLab~=sign(suppDists))
                warning(sprintf('Robots (%d,%d) paths conflict at step %d',i,j,step));
                plot_constraints([paths(:,step,j),paths(:,step+1,j)],...
                                [paths(:,step,i),paths(:,step+1,i)],...
                                hull,currA,currb);
                i_red = [paths(:,step,i)';paths(:,step+1,i)']
                j_green = [paths(:,step,j)';paths(:,step+1,j)']
                do = input('continue: ');
            end
            
            
        end
    end
    
    A(:,:,:,step) = stepA;
    b(:,:,step) = stepb;
end

end