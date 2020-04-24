function [allVerts,hullVerts] = swept_cyl_hull(cyl,from,to)
%SWEPT_CYL_HULL Summary of this function goes here
%   Detailed explanation goes here
    % comparison tolerance
    smallNum = 0.00001;

    %Translation direction
    dir = to - from;
    %total cyl height
    radius = cyl(1);
    above = cyl(2);
    below = cyl(3);
    height = above + below;
    
    %dirxy is used to orient polytope hull vertices in direction of travel
    dirxy = path(2,1:2) - path(1,1:2);
    if (abs(dir(1)) > smallNum) || (abs(dir(2)) > smallNum)
        dirxy_norm = dirxy/norm(dirxy);
    else %special case for vertical only movement
        dirxy_norm = [1,0];
    end

    %rotation matrices
    %rot = @(ang) [cos(ang), -sin(ang); sin(ang), cos(ang)];
    rot45 = [0.707107, -0.707107; 0.707107, 0.707107];
    rot90 = [0.0,-1.0;1.0,0.0];

    %dir for vertices
    v1dir = dirxy_norm * rot45;
    v2dir = v1dir * rot90;
    v3dir = -v1dir;
    v4dir = -v2dir;

    %bottom vertices
    bottom = [[v1dir*radius*sqrt(2),0] + from - [0,0,below]; ...
                 [v2dir*radius*sqrt(2),0] + from - [0,0,below]; ...
                 [v3dir*radius*sqrt(2),0] + from - [0,0,below]; ...
                 [v4dir*radius*sqrt(2),0] + from - [0,0,below]];

    verts0 = [bottom; bottom + repmat([0,0,height],[4,1])];
    verts1 = verts0 + dir;
    allVerts = [verts0;verts1];
    %hullVerts = allVerts;

    %prune hull verts
    %moving on xy
    if ((abs(dir(1)) > smallNum) || (abs(dir(2)) > smallNum)) 
        if (abs(dir(3)) < smallNum) %moving horizontally
            hullVerts = allVerts([2,3,6,7,9,12,13,16],:);
        elseif (dir(3) > 0) %moving +z diagonal
            hullVerts = allVerts([1,2,3,4,6,7,9,12,13,14,15,16],:);
        elseif (dir(3) < 0) %moving -z diagonal
            hullVerts = allVerts([2,3,5,6,7,8,9,10,11,12,13,16],:);
        end
    %moving vertically on z
    elseif (dir(3) > 0) %+z
        hullVerts = allVerts([1,2,3,4,13,14,15,16],:);
    elseif (dir(3) < 0) %-z
        hullVerts = allVerts([5,6,7,8,9,10,11,12],:);
    end
    
end

