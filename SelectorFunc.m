
function best = Selector( NoOfSubjects, NumExtracted )

global classmeanarray totalmeanarray ReductionFactor DatabaseVariance SubjectVariance ReductionMethod Separation

for(i = 1 : NumExtracted)
    for(j = 1 : NoOfSubjects)
        temp(j) = (DatabaseVariance(1,i)./ SubjectVariance(1,i,j));
    end
    importance(i) = sum(temp);
end

% stem(importance,'MarkerSize',3,'MarkerFaceColor','red');
% pause(10);
best = zeros(1,NumExtracted);

if ReductionMethod == 1
    
    [pks,locs] = findpeaks(importance,'minpeakdistance',Separation,'minpeakheight',ReductionFactor);
    
    importance1 = zeros(1,NumExtracted);
    for i = 1 : NumExtracted
        if  ismember(i,locs)
            importance1(i) = 1;
        end
    end
    best = importance1;
    
%     subplot(2,1,1), stem(importance,'MarkerSize',3,'MarkerFaceColor','red');
%     subplot(2,1,2), stem(best, 'MarkerSize',3,'MarkerFaceColor','red');
%      pause(10);

%     x = 1;
%     for j = 1 : width / 4
%         segment(j) = (sum(importance(x:x + height*2 - 1)) ./ (height * 2));
%         x = x + height * 2;
%     end
    
%        ( sum(segment) ./ width ./ 4 )
        
%       subplot(2,1,2), stem(segment, 'MarkerSize',3,'MarkerFaceColor','red'); 
      

    
else
    
  
%     subplot(2,1,1), stem(importance,'MarkerSize',3,'MarkerFaceColor','red');
    
    for(i = 1 : round(NumExtracted./ReductionFactor))
        [val,index] = max(importance);
        importance(index) = 0;
        best(index) = 1;
        
    end
%     subplot(2,1,2), stem(best,'MarkerSize',3,'MarkerFaceColor','red');
    
end


end


