function plot_constraints(path1,path2,hull,A,b)
%PLOT_CONSTRAINTS Summary of this function goes here
%   Detailed explanation goes here
    % agent 1 is green, agent 2 is red if given. hull is blue if given
    figure;
    plot3(path1(1,:),path1(2,:),path1(3,:),'-go','LineWidth',3)
    ax = gca;
    hold on;
    if (~isempty(hull))
        scatter3(hull(:,1),hull(:,2),hull(:,3),'bo')
    end
    if (~isempty(path2))
        plot3(path2(1,:),path2(2,:),path2(3,:),'-ro','LineWidth',3)
    end
    
    for c = 1:size(A,1)
        plotA = -A(c,:);
        plotb = b(c);
        buf = 2;
        bbx = [min([path1(1,:),path2(1,:)])-buf,max([path1(1,:),path2(1,:)])+buf];
        bby = [min([path1(2,:),path2(2,:)])-buf,max([path1(2,:),path2(2,:)])+buf];
        bbz = [min([path1(3,:),path2(3,:)])-buf,max([path1(3,:),path2(3,:)])+buf];
        [debx,deby,debz] = hyperplane_surf(plotA,plotb,bbx,bby,bbz,2);
        u = plotA(1)*ones(size(debx,1),size(debx,2));
        v = plotA(2)*ones(size(deby,1),size(deby,2));
        ww = plotA(3)*ones(size(debz,1),size(debz,2));
        quiver3(debx,deby,debz,u,v,ww,0.1);
        surf(debx,deby,debz,'FaceAlpha',0.5,'FaceColor',[0.4,0.1,0.4],'edgecolor','none');
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    ax.Projection = 'perspective';
    ax.DataAspectRatioMode = 'manual';
    ax.DataAspectRatio = [1 1 1];
    axis vis3d;
    hold off;
end

