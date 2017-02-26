function fit = fitness(position,classnum,totsize)                           % position = particle under consideration's xcol x yrow
                                                                            % position vector with 1's and 0's 
global classmeanarray totalmeanarray                                        % classnum = no of subjects
                                                                            % totsize = no of extracted features
  %  normal_fact = sum(position);
    
for i = 1 : totsize
    modct(i) = double(position(i)).* double(totalmeanarray(i));             % modct holds the selected features average of the database
end

summ=0;

for i = 1 : classnum
   str = int2str(i);
   
   for j = 1 : totsize
        midct(j) = double(position(j)).* double(classmeanarray(1,j,i));      % midct hold the selected features average of the subject
   end
   
    diff = abs(midct - modct) ;                                              % diff is the deviation(scatter) between the database and a
                                                                            % particular subject that has to be maximised.
     tpose = diff';
    
     mul = mtimes(diff,tpose);
     summ = double(summ)+double(mul); 
end

 fit = sqrt(summ);

