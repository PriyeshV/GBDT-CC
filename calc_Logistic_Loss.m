function [loss] = calc_Logistic_Loss(prediction,truth)
  
    n_ids = size(prediction,1);
    loss = zeros(n_ids,1);
    for i = 1:n_ids
        tmp = exp(-1*prediction(i)*truth(i));
        numerator =  (-1*truth(i)*tmp);
        denominator = 1 + tmp;
        loss(i,1) = (numerator/denominator);
    end
end

