function [ex_acc,precision,recall,f_measure,h_accuracy] = calc_acc_CoTraining(truth,prediction)

     n_unlabelled = size(truth,1);
     hamming_dist = zeros(n_unlabelled,1);
     ex_acc = 0;
     precision = 0;
     recall = 0;
     f_measure = 0;
     n = n_unlabelled;
     for i = 1:n_unlabelled
          acc = 0;
           hamming_dist(i,1) = pdist([prediction(i,:);truth(i,:)],'hamming');         
           t_labels = find(truth(i,:));           
           n_t_l = size(t_labels,2); 
           
           p_labels = find(prediction(i,:));
           n_p_l = size(p_labels,2);
           
            if n_t_l == 0 && isempty(find(prediction(i,:), 1))
                n = n - 1;
                continue;
            end
            if n_p_l == 0
                continue;
            end
            for p_l = p_labels
                if sum(p_l == find(truth(i,:))) ~= 0
                    acc = acc + 1;
                end
            end
            ex_acc = ex_acc + (acc / size(union(t_labels,p_labels),2));
            precision = precision + (acc/n_t_l);
            recall = recall + (acc/n_p_l);
            f_measure = f_measure + ((2*acc)/(n_t_l + n_p_l));
     end
     h_accuracy = 1 - mean(hamming_dist);
     ex_acc = ex_acc/n;
     precision = precision/n;
     recall = recall/n;
     f_measure = f_measure/n;
end

