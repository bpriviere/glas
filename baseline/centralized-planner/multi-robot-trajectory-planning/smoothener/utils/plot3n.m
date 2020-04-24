% for [3 n] array X, replace
%   plot3(X(1,:), X(2,:), X(3,:), args...)
% with
%   plot3n(X, args...)
%
% also handles [n 3] arrays.
%
function h = plot3n(x, varargin)
    sz = size(x);
    assert(length(sz) == 2);
    if sz(1) == 3
        h = plot3(x(1,:), x(2,:), x(3,:), varargin{:});
    elseif sz(2) == 3
        h = plot3(x(:,1), x(:,2), x(:,3), varargin{:});
    else
        error('argument x must be 3xN or Nx3');
    end
end

