%This code is an implementation of Gradient Boosted Decision Trees(GBDT)[http://david.grangier.info/papers/2012/shi_sdm_2012.pdf]
%for muti-label classification

clear all;
datasets = {'MLGene/'};
ML = 1; % 0 -multi-class and 1-multi-label
train_percentage = [0.02 0.04 0.06 0.08 0.1];
%To be decided by cross-validation
a_lambda = 0.75; %Weights for Attribute views Collectively
g_lambda = 1 - a_lambda; %Weights for Relational views Collectively

%Learning Algorithms parameters
Learner.type = 'CART';
Learner.attri_type = 1;
Learner.para = 'none';

K = 5;%Set number of folds for cross-valdiation.
max_iter = 10;%Set maximum number of boosting iterations.
%--------------------------------------------------------------------------
accl = zeros(5,1);
cnta = 0;
%for a_lambda = 0.25:0.25:1
%    cnta = cnta + 1;
%    g_lambda = 1 - a_lambda;
    for d = 1:size(datasets,2)
        %load ids
        load(char(strcat(datasets(d),'raw_ids.mat')));
        n_ids = length(ids);

        %load truth labels
        load(char(strcat(datasets(d),'truth.mat')));
        truth(truth == 0) = -1;    
        n_labels = size(truth,2);

        for train_perc = train_percentage
            disp(train_perc);
            fid = fopen(char(strcat(datasets(d),'Base3_results_',num2str(train_perc*100),'.txt')),'w'); 
            accuracy = zeros(K,1);        h_accuracy = zeros(K,1);        precision = zeros(K,1);        recall = zeros(K,1);        f_measure = zeros(K,1);

            %cross-validate
            for k = 1:K

                disp(strcat('K ============',num2str(k)));
                load(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100),'/',num2str(k),'.mat')));            
                unlabelled_indices  = ~labelled_indices;
                n_labelled = nnz(labelled_indices);
                n_unlabelled  = nnz(unlabelled_indices);            
    %             truth_2 = truth(unlabelled_indices,:);
    %             truth_2(truth_2 == -1) = 0;

                %prepare attribute views
                name_views = {'view'};%Attribute views
                n_views = length(name_views);     

                V_Iteration = struct;
                V_Iteration(1,1).view = struct;   
                V_Iteration(1,1).view.classifier = struct;             
                view = struct;   

                for view_id = 1:n_views
                    tmp_obj = load(char(strcat(datasets(d),name_views(1,view_id),'.mat')));    
                    view(view_id).r_view = tmp_obj.('view');                
                end       
                a_weights = ones(1,n_views) * (1/n_views); %Weights of attribute views

                %prepare relational views            
                name_links = {'adjmat1','adjmat2'};			
                n_links = length(name_links);
                link = struct;

                for link_id = 1:n_links                  
                    tmp_obj = load(char(strcat(datasets(d),name_links(1,link_id),'.mat')));
                    link(link_id).links = tmp_obj.('links');     
                end            
                g_weights = ones(1,n_links) * (1/n_links); %Weights of relational views

                %Fix Line search range for computing rho
                %Set Range [-100.0 - 100.0] at [0.1] intervals 
                line_pts = zeros(1,0);
                l= 1;
                for p = -100:100
                    for q = 0.1:1                                
                        line_pts(1,l) = p*q;
                    l = l + 1;
                    end                            
                end

                %Initialization
                iter_id = 1;          
                for view_id = 1:n_views                  
                    V_Iteration(1,iter_id).view(view_id).regressor = struct;
                    prediction = zeros(n_ids,n_labels);
                    for label_id = 1:n_labels                 
                        V_Iteration(1,iter_id).view(view_id).rho(label_id) = 1;
                        %Training
                        V_Iteration(1,iter_id).view(view_id).regressor(label_id).model = BinaryClassify_train(Learner,view(view_id).r_view(labelled_indices,:),truth(labelled_indices,label_id));            

                        %Prediction
                        [prediction(:,label_id),~] = eval(V_Iteration(1,iter_id).view(view_id).regressor(label_id).model.model,view(view_id).r_view);                       
                        V_Iteration(1,iter_id).view(view_id).reg_val(:,label_id) = prediction(:,label_id);                                                           
                    end
                    if ML == 0
                        [~,idx] = max(prediction');
                        for label_id = 1:n_labels
                            V_Iteration(1,iter_id).view(view_id).prediction(:,label_id) = zeros(n_ids,1) - 1;                    
                            V_Iteration(1,iter_id).view(view_id).prediction(idx == label_id,label_id) = 1;
                        end                                       
                    else
                        prediction(prediction >=0 ) = 1;
                        prediction(prediction < 0) = -1;                    
                        V_Iteration(1,iter_id).view(view_id).prediction = prediction;                                            
                    end                                
                end

                                 
                %Boosting
                graph_labels = zeros(n_ids,n_labels,n_views);                                           
                for iter_id = 1:max_iter

                    %Get expected labels from attribute views collectively                
                    att_labels = get_ExpLabels_Att(V_Iteration(1,iter_id),a_weights,ML);

                    %Get expected labels from relational views collectively for labels of each attribute view
                    for view_id = 1:n_views                    
                       graph_labels(:,:,view_id) = get_ExpLabels_Graph(V_Iteration(1,iter_id),g_weights,link,view_id,ML); 
                    end

                    for label_id = 1:n_labels
                        for view_id = 1:n_views

                            %Calculate Loss:                        
                            %Calculate loss on Attribute views
                            att_loss = calc_Logistic_Loss(V_Iteration(1,iter_id).view(view_id).prediction(unlabelled_indices,label_id), att_labels(unlabelled_indices,label_id));
                            %Calculate loss on Relational views
                            graph_loss = calc_Logistic_Loss(V_Iteration(1,iter_id).view(view_id).prediction(unlabelled_indices,label_id), graph_labels(unlabelled_indices,label_id,view_id));
                            %Calculate loss on Labeled set
                            training_loss = calc_Logistic_Loss(V_Iteration(1,iter_id).view(view_id).prediction(labelled_indices,label_id), truth(labelled_indices,label_id));

                            %Calculate residue
                            residue = zeros(n_ids,1);
                            residue(labelled_indices,1) = training_loss;
                            residue(unlabelled_indices,1) = (a_lambda*att_loss) + (g_lambda*graph_loss);
                            residue = -a_weights(1,view_id) .* residue;

                            %Fit a model over the residue
                            V_Iteration(1,iter_id+1).view(view_id).regressor(label_id).model = BinaryClassify_train(Learner,view(view_id).r_view,residue);                
                            [prediction,~] = eval(V_Iteration(1,iter_id+1).view(view_id).regressor(label_id).model.model,view(view_id).r_view);                         
                            V_Iteration(1,iter_id+1).view(view_id).reg_val(:,label_id) = prediction;                        

                            %Compute Rho by Line Search                                                
                            loss = zeros(size(line_pts,2),1);
                            for rho_id = 1:size(line_pts,2)
                                prediction = V_Iteration(1,iter_id).view(view_id).prediction(:,label_id);
                                prediction = prediction + (line_pts(rho_id) .* V_Iteration(1,iter_id +1).view(view_id).reg_val(:,label_id));                            

                                prediction(prediction >= 0) = 1;
                                prediction(prediction <0) = -1;

                                att_loss = calc_Logistic_Loss(prediction(unlabelled_indices,1), att_labels(unlabelled_indices,label_id));
                                graph_loss = calc_Logistic_Loss(prediction(unlabelled_indices,1), graph_labels(unlabelled_indices,label_id));
                                training_loss = calc_Logistic_Loss(prediction(labelled_indices,1), truth(labelled_indices,label_id));

                                residue = zeros(n_ids,1);
                                residue(labelled_indices,1) = training_loss;
                                residue(unlabelled_indices,1) = (a_lambda*att_loss) + (g_lambda*graph_loss);
                                residue = -a_weights(1,view_id) .* residue;
                                loss(rho_id,1) = sum(abs(residue));
                            end                            
                            %Update rho
                            V_Iteration(1,iter_id+1).view(view_id).rho(label_id) = line_pts(1,find(loss == min(loss),1));                        
                        end
                    end

                    %Compute Predictions and Update weights 
                    prediction = zeros(n_ids,n_labels,n_views) - 1;                                              
                    for view_id = 1:n_views
                        tmp = zeros(n_ids,n_labels);  
                        %Compute Predictions
                        for label_id = 1:n_labels
                            tmp(:,label_id) = V_Iteration(1,iter_id).view(view_id).prediction(:,label_id);
                            tmp(:,label_id) = tmp(:,label_id) + (V_Iteration(1,iter_id+1).view(view_id).rho(label_id) .* V_Iteration(1,iter_id +1).view(view_id).reg_val(:,label_id));                                                                               
                        end 
                        if ML == 0
                            [~,idx] = max(tmp');
                            for label_id = 1:n_labels
                                prediction(idx == label_id,label_id,view_id) = 1;
                                V_Iteration(1,iter_id+1).view(view_id).prediction(:,label_id) = zeros(n_ids,1) - 1;                    
                                V_Iteration(1,iter_id+1).view(view_id).prediction(idx == label_id,label_id) = 1;                            
                            end                                       
                        else                        
                            for label_id = 1:n_labels
                                prediction(tmp(:,label_id,view_id) >= 0,label_id,view_id) = 1;
                                %prediction(tmp(:,label_id,view_id) < 0,label_id,view_id) = -1;   
                                V_Iteration(1,iter_id+1).view(view_id).prediction(:,label_id) = prediction(:,label_id,view_id);
                            end
                        end

                        %Update Weights 
                        loss = zeros(1,n_labels);
                        total_loss = zeros(1,n_views);
                        tmp = zeros(1,n_views);
                        for label_id = 1:n_labels
                            att_loss = calc_Logistic_Loss(prediction(unlabelled_indices,label_id,view_id), att_labels(unlabelled_indices,label_id));
                            graph_loss = calc_Logistic_Loss(prediction(unlabelled_indices,label_id,view_id), graph_labels(unlabelled_indices,label_id,view_id));
                            training_loss = calc_Logistic_Loss(prediction(labelled_indices,label_id,view_id), truth(labelled_indices,label_id));

                            residue = zeros(n_ids,1);
                            residue(labelled_indices,1) = training_loss;
                            residue(unlabelled_indices,1) = (a_lambda*att_loss) + (g_lambda*graph_loss);
                            residue = -a_weights(1,view_id) .* residue;
                            loss(1,label_id) = mean(abs(residue));
                        end
                        total_loss(1,view_id) = mean(loss);
                        comp = prediction(:,:,view_id) == truth;                      
                        total_loss(1,view_id) = size(find(comp == 0),1)/(n_ids*n_labels);
                    end

                    for i = 1:n_views                   
                       tmp(1,i) = exp(-total_loss(1,i));
                    end

                    %update Attribute view weights
                    a_weights = tmp./sum(tmp);                             

                    %update Relational view weights
                    g_weights = update_graph_weights(link,prediction);   
                end

                tmp = zeros(n_ids,n_labels); 
                for view_id = 1:n_views
                    tmp = tmp + (a_weights(1,view_id) * prediction(:,:,view_id));                    
                end                
                prediction = zeros(n_ids,n_labels) -1;
                if ML == 0
                     [lab,idx] = max(tmp');           
                     for label_id = 1:n_labels
                         prediction(idx == label_id,label_id) = 1;
                     end
                else                           
                    prediction(tmp >=0) = 1;
                    %final_prediction(tmp <0) = -1;   
                end                              
                truth_2 = truth(unlabelled_indices,:);
                truth_2(truth_2 == -1) = 0;
                final_predictions = prediction(unlabelled_indices,:);
                final_predictions(final_predictions == -1) = 0;
                if ML == 0
                    [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1)] = calc_acc_CoTrainingmc(truth_2,final_predictions);
                else
                    [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)] = calc_acc_CoTrainingml(truth_2,final_predictions);
                end 
                fprintf(fid,'%f %f %f %f %f\n',[accuracy(k,1) precision(k,1) recall(k,1) f_measure(k,1) h_accuracy(k,1)]);          
            end
            fprintf(fid,'%f %f %f %f %f\n',[mean(accuracy) mean(precision) mean(recall) mean(f_measure) mean(h_accuracy)]);          
             disp([accuracy precision recall f_measure h_accuracy]);
             fclose('all');
        end  
    end
    
%    accl(cnta) = accuracy(k,1);
%end