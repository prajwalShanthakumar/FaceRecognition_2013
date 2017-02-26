function output = ScaleNorm(I,FilterWidth,EdgeSelectivity1,EdgeSelectivity2);

[row,col] = size(I);
storeI = I;


for i = 1 : row
    for j = 1 : col
gaussian_filter(i,j) = ( exp(   - 1 .* ( (((row./2)-i).^ 2) + (((col./2)-j).^ 2) )  ./ (FilterWidth.^2) ));
    end
end

I = double(I) .* (gaussian_filter);

for i = 1 : row
    for j = 1 : col
        if(j == col)
            I(i,j) = I(i,j);
        else
            I(i,j) = 2 .* (abs(I(i,j) - I(i,j+1)));
        end
    end
end

for i = 1 : col
    for j = 1 : row
        if(j == row)
            I(j,i) = I(j,i);
        else
            I(j,i) = 2 .* (abs(I(j,i) - I(j+1,i)));
        end
    end
end


for i = 1 : row
    for j = 1 : col
        if( (I(i,j) <= EdgeSelectivity1 && i <= round(row./2)) || (I(i,j) <= EdgeSelectivity2 && i > round(row./2)))
            I(i,j) = 0;
        end
    end
end

leftmost = col;
rightmost = 1;
for i = 1 : round(row./2)
    for j = 1 : col
       if(I(i,j) ~= 0 && j < leftmost)
           leftmost = j;
       end
       if(I(i,j) ~= 0 && j > rightmost)
           rightmost = j;
       end
    end
end
leftmost;
rightmost;

topmost = row;
bottommost = 1;
for i = 1 : col
    for j = 1 : row
        if(I(j,i) ~= 0 && j < topmost)
            topmost = j;    
        end
        if(I(j,i) ~= 0 && j > bottommost)
            bottommost = j;    
        end
    end
end
topmost;
bottommost;

output = storeI(topmost:bottommost,leftmost:rightmost);
%figure, imshow(uint8(storeI(topmost:bottommost,leftmost:rightmost)));
%surf(gaussian_filter);