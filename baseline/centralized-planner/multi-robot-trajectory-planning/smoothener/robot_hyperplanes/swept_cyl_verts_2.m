function [verts] = swept_cyl_verts_2(cyl,waypoints,sides)
%SWEPT_CYL_VERTS_2 Summary of this function goes here
%   Detailed explanation goes here

%expand cylinder definition
    radius = cyl(1);
    above = cyl(2);
    below = cyl(3);
    height = above + below;
    
    %alocate space for vertices
    npath = size(waypoints,1);
    npolyverts = sides*2;
    verts = zeros(npath*npolyverts,3);
    
    %template cylinder polytope
    polyr = radius/(cos(pi/sides)); %radius of circumscribed polygon
    theta = (0:(2*pi/sides):(2*pi-(2*pi/sides)))';
    %xy of circle polygon
    polyx = polyr * cos(theta); 
    polyy = polyr * sin(theta);
    
    polytope = [polyx,polyy,-below*ones(sides,1);...
                polyx,polyy,above*ones(sides,1)];
    
    %enumerate template at waypoints
    for t = 1:npath
        s = (t-1)*npolyverts + 1;
        e = t*npolyverts;
        verts(s:e,:) = polytope + repmat(waypoints(t,:),npolyverts,1);
    end
end

