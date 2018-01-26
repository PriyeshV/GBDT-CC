function [labels] = get_ExpLabels_Att(Iteration,weights,ML)

    n_views = size(weights,2);
    [n_ids,n_labels] = size(Iteration(1).view(1).prediction);
    
    labels = zeros(n_ids,n_labels) - 1;
    prediction = zeros(n_ids,n_labels);
    for label_id = 1:n_labels
        for view_id = 1:n_views
            prediction(:,label_id) = prediction(:,label_id) + ( weights(1,view_id) * Iteration(1).view(1,view_id).prediction(:,label_id));            
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

