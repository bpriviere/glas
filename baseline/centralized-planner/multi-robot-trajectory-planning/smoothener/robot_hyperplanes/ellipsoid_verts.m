function [ verts ] = ellipsoid_verts( rx,ry,rz, ntheta, nphi )
%ELLIPSOID_VERTS generates list of vertices of an axis aligned ellipsoid
%INPUT:
%   all inputs are scalar values
%   rx,ry,rz: float > 0
%             lengths of x,y,z aligned semi-axes
%   ntheta: integer > 2
%           number of divisions along theta angle (spherical coordinates)
%   nphi:  integer > 4
%          number of divisions along phi angle
%OUTPUT:
%   verts: [#verts, 3]
%          each row contains [x,y,z] coordinate of one vertex from
%          ellipsoid
%          #verts = (ntheta - 2) * (nphi - 1) + 6
%NOTES: 
%   vertices at the extents of the elipsoids are guarenteed, and may be
%   duplicated if they also exist on the given phi/theta divisions


%theta sample points
thetaDivs = linspace(0,pi,ntheta);
thetaDivs = thetaDivs(2:(end-1)); %theta = 0,pi handled by extents

phiDivs = linspace(0, 2*pi, nphi);
phiDivs = phiDivs(1:(end-1)); %phi = 0, 2pi are the same


%these are the vertices at the extents of the ellipsoids
extents = [0,0,rz;...
           0,0,-rz;...
           0,ry,0;...
           0,-ry,0;...
           rx,0,0;...
           -rx,0,0];

%grid divisions
[theta,phi] = meshgrid(thetaDivs,phiDivs);
sintheta = sin(theta);
sinphi = sin(phi);
costheta = cos(theta);
cosphi = cos(phi);

%calculate vertex points
x = rx .* cosphi .* sintheta;
y = ry .* sinphi .* sintheta;
z = rz .* costheta;

%reshape points into list
verts = [extents;x(:),y(:),z(:);];

end

