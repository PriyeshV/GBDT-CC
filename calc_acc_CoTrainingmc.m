function [accuracy,precision,recall,f_measure] = calc_acc_CoTraining(truth,prediction)
%     n = size(truth,1);
%     accuracy = nnz(truth == prediction)/n;
%     precision = length(intersect(find(truth), find(prediction)))/length(find(prediction));
%     recall = length(intersect(find(truth), find(prediction)))/length(find(truth));
      [~,t] = max(truth');
      [~,p] = max(prediction');
      
      c = classperf(t,p);
      accuracy = c.CorrectRate;      
      precision = c.PositivePredictiveValue;
      if isnan(precision)
          precision = 0.00001;
      end
      recall = c.Sensitivity;
      if isnan(recall)
          recall = 0.00001;
      end

      f_measure = 2*(precision * recall)/(precision + recall);  
      if isnan(f_measure)
          f_measure = 0.00001;
      end
end

