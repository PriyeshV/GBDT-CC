function [labels ] = get_ExpLabels_Graph(Iteration,weights,link,view_id,ML)
    
    n_links = size(weights,2);
    [n_ids,n_labels] = size(Iteration(1).view(1).prediction);
        
    %ul_ids = find([1:n_ids]' .* unlabelled_indices);    
    labels = zeros(n_ids,n_labels) -1;
    prediction = zeros(n_ids,n_labels);
    for label_id = 1:n_labels       
                
        for i = 1:n_ids            
            for link_id = 1:n_links               
                neigh = find(link(1,link_id).links(i,:));  
                if(isempty(neigh))
                    continue;
                end
                prediction(i,label_id) = (weights(1,link_id)*((1/size(neigh,2)) * sum(Iteration(1).view(1,view_id).prediction(neigh,label_id))));              
            end
        end    
        
    end
    if ML == 0
       [~,idx] = max(prediction');
       for label_id = 1:n_labels
           labels(idx == label_id,label_id) = 1;
       end
    else    
        labels(prediction >= 0) = 1;        
    end    
end

