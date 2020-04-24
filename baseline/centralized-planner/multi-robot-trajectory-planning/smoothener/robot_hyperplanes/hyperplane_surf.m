function [x,y,z] = hyperplane_surf(A,b,xrange,yrange,zrange,xyzstep)

    maxdim = find(abs(A)==max(abs(A)));
    %solve for dim that has largest normal coeff (avoid 1/coeff making huge
    %numbers)
    if (maxdim == 1)
        [z, y] = meshgrid(zrange(1):xyzstep:zrange(2),...
                    yrange(1):xyzstep:yrange(2)); % Generate z and y data
        x = (-1/A(1))*(A(3)*z + A(2)*y + b); % Solve for x data
    elseif (maxdim == 2)
        [x, z] = meshgrid(xrange(1):xyzstep:xrange(2),...
                    zrange(1):xyzstep:zrange(2)); % Generate x and z data
        y = (-1/A(2))*(A(1)*x + A(3)*z + b); % Solve for y data
    else
        [x, y] = meshgrid(xrange(1):xyzstep:xrange(2),...
                    yrange(1):xyzstep:yrange(2)); % Generate x and y data
        z = (-1/A(3))*(A(1)*x + A(2)*y + b); % Solve for z data
    end
    
end