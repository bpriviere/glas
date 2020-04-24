function [An,bn] = noredund(A,b,c)
% NOREDUND - Remove redundant linear inequalities from a set; i.e.,
%            remove redundant linear constraints defining a feasible
%            region. Note that the feasible region satisfies A*x <= b,
%            where A is a fixed matrix, b is a fixed vector, and x
%            is the vector of coordinates in your space; i.e., all
%            values of x (or equivalently, all ordered n-tuples of
%            coordinate numbers) which satisfy the inequality A*x <= b
%            are inside the feasible region.
% 
% [An,bn] = noredund(A,b)
% 
% For n variables:
% A  = m  x n matrix, where m >= n (m constraints)
% b  = m  x 1 vector (m constraints)
% An = mm x n matrix, where mm >= n (mm nonredundant constraints)
% bn = mm x 1 vector (mm nonredundant constraints)
% 
% NOTES: (1) Unbounded feasible regions are permitted.
%        (2) This program requires that the feasible region have some
%            finite extent in all dimensions. For example, the feasible
%            region cannot be a line segment in 2-D space, or a plane
%            in 3-D space.
%        (3) At least two dimensions are required.
%        (4) See function CON2VERT which is limited to bounded feasible
%            regions but also outputs vertices for the region.
%        (5) Written by Michael Kleder, June 2005.
%
% EXAMPLE (two figures produced):
%
% n=20;
% A=rand(n,2)-.5;
% b=rand(n,1);
% figure('renderer','zbuffer')
% hold on
% [x,y]=ndgrid(-3:.01:3);
% p=[x(:) y(:)]';
% p=(A*p <= repmat(b,[1 length(p)]));
% p = double(all(p));
% p=reshape(p,size(x));
% h=pcolor(x,y,p);
% set(h,'edgecolor','none')
% set(h,'zdata',get(h,'zdata')-1) % keep in back
% axis equal
% set(gca,'color','none')
% title(['Original feasible region, with ' num2str(size(A,1)) ' constraints.'])
% [A,b]=noredund(A,b);
% figure('renderer','zbuffer')
% hold on
% [x,y]=ndgrid(-3:.01:3);
% p=[x(:) y(:)]';
% p=(A*p <= repmat(b,[1 length(p)]));
% p = double(all(p));
% p=reshape(p,size(x));
% h=pcolor(x,y,p);
% set(h,'edgecolor','none')
% set(h,'zdata',get(h,'zdata')-1) % keep in back
% axis equal
% set(gca,'color','none')
% title(['Final feasible region, with ' num2str(size(A,1)) ' constraints.'])

if length(b) <= 2
    An = A;
    bn = b;
    return;
end

if nargin < 3
    % move polytope so that origin is included:
    % first, attempt to locate a feasible point:
    c = A\b; % least-squares soln, correct if no redundant constraints
    % find another solution if above isn't *inside* feasible region:
end

if ~all(A*c < b) % exclude exterior and also boundary points
    opts = optimset('display', 'off');
    [c, ~, flag] = linprog(0*A(1,:), A, b, [], [], [], [], [], opts);
    assert(flag == 1);
    %{
    [c,f,ef] = fminsearch(@obj,c,'params',{A,b});
    if ef ~= 1
        warning('Unable to locate a point within the interior of a feasible region.')
        An = A;
        bn = b;
        return;
    end
    %}
end

% move polytope to contain origin
bk = b; % preserve
b = b - A*c; % polytope A*x <= b now includes the origin
% obtain dual polytope vertices
D = A ./ repmat(b,[1 size(A,2)]);
try
    k = convhulln(D);
    % record which constraints generate points on the convex hull
    nr = unique(k(:));
    An=A(nr,:);
    bn=bk(nr);
catch
    % if convhull fails, just use the original
    An = A;
    bn = bk;
end

function d = obj(c,params)
A=params{1};
b=params{2};
d = A*c-b;
k=(d>=-1e-15); % exclude quasi-boundary points
d(k)=d(k)+1;
d = max([0;d]);
return

