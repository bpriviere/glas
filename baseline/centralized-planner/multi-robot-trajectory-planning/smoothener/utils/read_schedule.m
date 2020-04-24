% read a schedule from discrete planner (yaml file)
% into a [3 x Kpts x Nrobots] waypoint array
%
function [paths,names,types] = read_schedule(fname, read_agents)
   
    yamlData = yaml.ReadYaml(fname);
    
    % find number of agents
    agents = fieldnames(yamlData.schedule);
    N = numel(agents);
    
    % find total timesteps
    k = 0;
    for i=1:N
        T = size(yamlData.schedule.(agents{i}), 2);
        k = max(k, T);
    end
    
    paths = nan(3,k,N);
    names = cell(N,1);
    types = cell(N,1);
    iprime = 1;
	for i=1:N
        %get path
		p = yamlData.schedule.(agents{i});
		len = size(p, 2);
        for j=1:len
            paths(1,j,iprime) = p{j}.x + 0.5;
            paths(2,j,iprime) = p{j}.y + 0.5;
            %paths(3,j,iprime) = p{j}.z;
            paths(3,j,iprime) = 0.5;
        end
        
        for j=(len+1):k
            paths(:,j,iprime) = paths(:,len,iprime);
        end
        %get name
        names{iprime} = agents{i};
        %get type
        types{iprime} = 'ground';
        
        iprime = iprime + 1;
	end
	assert(~any(isnan(paths(:))));
end
