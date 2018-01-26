function [g_weights] = update_graph_weights(link,prediction)

    n_links = size(link,2);
    [n_ids,n_labels,n_views] = size(prediction);        
    g_weights = zeros(1,n_links);
    
    for link_id = 1:n_links
        loss_neigh = zeros(n_ids,1);
        for i = 1:n_ids
            neigh = find(link(1,link_id).links(i,:));  
            if isempty(neigh)
                continue;
            end
            for j = neigh
                loss_views = 0;
                for view_id = 1:n_views
                    loss_views_labels = 0;
                    for label_id = 1:n_labels
                        loss_views_labels = loss_views_labels + ~(prediction(i,label_id,view_id) == prediction(j,label_id,view_id));
                    end
                    loss_views = loss_views + (loss_views_labels/n_labels);
                end   
                loss_neigh(i,1) = loss_neigh(i,1) + (loss_views/n_views);
            end   
            loss_neigh(i,1) = loss_neigh(i,1)/size(neigh,2);
        end        
        g_weights(1,link_id) = exp(-sum(loss_neigh)/n_ids);
    end
    g_weights = g_weights./sum(g_weights);
end
          