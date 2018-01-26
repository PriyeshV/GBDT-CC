function Classifier=BinaryClassify_train(Learner,data,labels)
%BinaryClassify_train trains a classifier from given data using the specified learner
%
%    Syntax
%
%       Classifier=BinaryClassify_train(Learner,data,labels)
%
%    Description
%
%       BinaryClassify_train takes,
%           Learner         - A struct variable with two fields 'type', 'attri_type' which determines the properties of base learners used by CoTrade:           
%                             1) Learner.type gives the type of learner used in training, which can take the value of 'NB' (naive bayes), 'SVM' (support vector
%                                machines) or 'CART' (CART decision tree);
%                             2) Learner.attri_type indicates what kind of attributes that the base learners will deal with, 0 for binary features while 1 for
%                                real-valued features;
%                             3) Learner.para gives the specific parameters used by Learner.
%           data        	- An Mxd array, where the i-th training instance is stored in data(i,:);
%           labels          - An Mx1 vector, if the i-th training instance is positive, labels(i,1)=1, otherwise labels(i,1)=0;
%
%      and returns,
%           Classifier      - A struct variable with three fields 'type', 'attri_type' and 'model' that specifies the learned classifier's relevant information:
%                             1) Classifier.type gives the type of classifiers, which can take the value of 'NB' (naive bayes), 'SVM' (support vector
%                                machines) or 'CART' (CART decision tree);
%                             2) Classifier.attri_type indicates which kinds of attributes it can deal with, 0 for binary features while 1 for
%                                real-valued features;
%                             3) Classifier.model contains the specific model information:
%                                a) If strcmp(Classifier.type,'NB')==1, Classifier.model is again a struct variable with three fields 'prior', 'paraPos' (2xd1 array)
%                                   and 'paraNeg' (2xd1 array):
%                                   a1) Classifier.model.prior(1) gives the prior probability of an instance being positive while Classifier.model.prior(2)
%                                       gives the prior probability of an instance being negative.
%                                   a2) The meanings of the other two fields (i.e. ParaPos and paraNeg) depend on the value of Classifier.attri_type 
%                                       i)  If Classifier.attri_type is 0 (binary features), Classifier.model.paraPos(1,d) gives the conditional probability that an instance
%                                           will take a value of 1 on its d-th dimension given it is positive, while Classifier.model.paraPos(2,d) gives the conditional
%                                           probability that an instance will take a value of 0 on its d-th dimension given it is positive. Similarly, Classifier.model.paraNeg(1,d)
%                                           gives the conditional probability that an instance will take a value of 1 on its d-th dimension given it is negative, while 
%                                           Classifier.model.paraNeg(2,d) gives the conditional probability that an instance will take a value of 0 on its d-th
%                                           dimension given it is negative.
%                                       ii) If Classifier.attri_type is 1 (real-valued features), Classifier.model.paraPos(1,d) gives the mean of the Gaussian distribution on the 
%                                           instance's d-th dimension given it is positive, while Classifier.model.paraPos(2,d) gives the standard deviation of the
%                                           Gaussian distribution on the instance's d-th dimension given it is positive. Similarly, Classifier.model.paraNeg(1,d) 
%                                           gives the mean of the Gaussian distribution on the instance's d-th dimension given it is negative, while Classifier.model.paraNeg(2,d) 
%                                           gives the standard deviation of the Gaussian distribution on the instance's d-th dimension given it is negative.
%                                b) If strcmp(Classifier.type,'SVM')==1, Classifier.model is a struct variable used by Libsvm [1] 
%                                c) If strcmp(Classifier.type,'CART')==1, Classifier.model is a tree struct variable used by MATLAB (see Statistics Toolbox
%                                   on classification and regression trees)
%
% [1] C.-C. Chang and C.-J. Lin. LIBSVM: A library for support vector machines, 2001. software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

    [num_inst,dim]=size(data);
    num_pos=sum(labels==1);
    num_neg=num_inst-num_pos;
    
    Classifier.type=Learner.type;
    Classifier.attri_type=Learner.attri_type;
    
    switch Classifier.type
        case 'NB'
            Classifier.model.prior(1)=(1+num_pos)/(2+num_inst); %Laplace smoothing
            Classifier.model.prior(2)=1-Classifier.model.prior(1);

            index_pos=find(labels==1);
            index_neg=find(labels==0);
            pos_data=data(index_pos,:);
            neg_data=data(index_neg,:);

            if(Classifier.attri_type==0)
                Classifier.model.paraPos(1,:)=(1+sum(pos_data==1,1))/(2+num_pos);
                Classifier.model.paraPos(2,:)=1-Classifier.model.paraPos(1,:);
                Classifier.model.paraNeg(1,:)=(1+sum(neg_data==1,1))/(2+num_neg);
                Classifier.model.paraNeg(2,:)=1-Classifier.model.paraNeg(1,:);
            else
                [Classifier.model.paraPos(1,:),Classifier.model.paraPos(2,:)]=normfit(pos_data);
                [Classifier.model.paraNeg(1,:),Classifier.model.paraNeg(2,:)]=normfit(neg_data);
            end
        case 'CART'
            if(isempty(Learner.para))
                T=classregtree(data,labels,'categorical',1:dim,'method','classification','splitcriterion','gdi','splitmin',3);
            else
                if(strcmp(Learner.para,'none')==1)
                    T=classregtree(data,labels,'method','regression','splitcriterion','gdi');
                else
                    T=classregtree(data,labels,'categorical',Learner.para,'method','classification','splitcriterion','gdi');
                end
            end
            Classifier.model=T;
        case 'SVM'            
            Classifier.model=svmtrain(labels,data,'-t 2 -b 1');
        otherwise
            Classifier.model = NaiveBayes.fit(data,labels,'Distribution','mn');
    end