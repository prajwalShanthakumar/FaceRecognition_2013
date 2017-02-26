% Face Recognition System with DWT/DCT as feature extractor
% and the proposed DOFS (dual objective feature selector) (or BPSO) as the feature selector
% Euclidean distance is the metric for the classifier
% Prajwal Shanthakumar (1MS10EC079) and Siddharth Srivatsa(1MS10EC120) 
clc
close all
clear all

global ReductionFactor  classmeanarray totalmeanarray GlobalBestP DatabaseVariance SubjectVariance ReductionMethod Separation;



DatabaseChoice = 2; % FERET database

FilterWidth = 50;
EdgeSelectivity1 = 20;
EdgeSelectivity2 = 20;

% WaveletChoice = input('1. Haar\n2. Daubechies db4\n3. Biorthogonal 1.3\n4. Reverse biorthogonal 1.3\n5. Symlet 4\n6. Co-iflet 1 \n');
WaveletChoice = 0; 
% NoOfDecompositions = input('\nEnter no. of DWT decompositions ');
NoOfDecompositions = 0;
%DctFact = input('Enter DCT extraction factor ');
DctFact = 8;
%Adjust = input('Enter DCT adjust ');
Adjust = 3;

Selector = 1;
%Selector = input('1.Proposed Dual Objective Feature Selector \n 2. Binary Particle Swarm Optimization ');
if( Selector == 2 )
    
    IniSigmoidalPar = input('Enter Sigmoidal Parameter ');
    Threshold = input('Enter BPSO selection threshold ');
    SigmoidalPar = IniSigmoidalPar;
    
else
    ReductionMethod = 2;
    %ReductionMethod = input('1.Selecting above a Threshold 2.Specifying reduction factor ');
    if(ReductionMethod == 1)
        ReductionFactor = input('Enter threshold ');
        Separation = input('Enter spacing between selected features ');
    else
        %ReductionFactor = input('Enter Selector Reduction Factor ');
        ReductionFactor = 4;
    end
    
end
trials  = input('Enter no. of trials ');

percentsum(:,:) = zeros(1,trials);                                  % Variable to store the RR percentages of each trial


switch(DatabaseChoice)
    case 1,
        database = 'C:\Users\Prajwal\Documents\Research stuff\Databases\ORL\s'; training = 4; testing = 6; subjects = 40; ext = '.pgm';
        databasename = 'ORL';
    case 2,
        %database = 'C:\Users\Prajwal\Documents\Face Recognition\Databases\New_Feret\'; training = 8; testing = 12; subjects = 35; ext = '.ppm';
        database = '\FERET\'; training = 8; testing = 12; subjects = 35; ext = '.ppm';
        curdir = pwd;
        database = strcat(curdir,database);
        databasename = 'FERET';
        %current_dir = pwd;
        %database = dir([current_dir '..\FERET\']);
    case 3,
        database = 'C:\Users\Prajwal\Documents\Research stuff\Databases\YaleB\'; training = 3; testing = 16; subjects = 28; ext = '.pgm';
        databasename = 'Extended Yale B';
    case 4,
        database = 'C:\Users\Prajwal\Documents\Research stuff\Databases\Pie\OriginalCMUPIEMixed\s'; training = 5; testing = 8; subjects = 20; ext = '.jpg';
        databasename = 'CMU PIE';
end
imagespersubject = testing + training;                              % Images per subject in the database


switch(WaveletChoice)
    case 1,
        wavelet = 'haar';
    case 2,
        wavelet = 'db4';
    case 3,
        wavelet = 'bior1.3';
    case 4,
        wavelet = 'rbio1.3';
    case 5,
        wavelet = 'sym4';
    case 6,
        wavelet = 'coif1';
end





Velocity = 1;                                                               % A 'particle' is a 4D array ( particle(:,:,:,:) )
Position = 2;                                                               % The 4th dimension is the particle Number
Cost = 3;                                                                   % The 3rd dimension for the each 'particle' is one of it's characteristics.
LocalBestPosition = 4;                                                      % The fields are 1) It's present velocity 2) It's present position.
LocalBestCost = 5;                                                          % 3) It's present fitness(cost) 4) It's best position 5) It's best fitness.

% The 1st and 2nd dimensions (row matrix) store the extracted feature set.



% Original Height, Original Width
% height, width


for x = 1 : trials
    FaceNum = 1;                                                            % Variable which generates the Serial Number of the training face
    % takes values from 1 : NumSubject * NumTraining
    
    
    
    
    %%                                                                TRAINING
    
    
    
    
    trainingtemp = tic;                                                                         % Training begins
    
    % PRE - PROCESSING
    
    
    for(j = 1 : subjects)
        
        TrainingSet(1,:,j) = randperm(imagespersubject,training);
        
        for(i = 1 : training)
            
            trainface = imread(strcat(database,num2str(j),'\',num2str(TrainingSet(1,i,j)),ext));
            
            if(x == 1 && j == 1 && i == 1)
                [originalheight,originalwidth,m] = size(trainface);
            end
            
            
            
            trainface = rgb2gray(trainface);
            
            
            trainface = (imresize(trainface, 1 ./ (2 .^ (DatabaseChoice))));
            
            [height,width] = size(trainface);
            
            storeface1{FaceNum} = trainface;
            
            %              face = ScaleFunc(face);
            storeface1{FaceNum} = ScaleNorm(trainface,FilterWidth,EdgeSelectivity1,EdgeSelectivity2);
            
            
            storeface1{FaceNum} = imresize(storeface1{FaceNum},[height, width]);
            
            
            [ReducedHeight,ReducedWidth] = size(storeface1{FaceNum});
            
            FaceNum = FaceNum + 1;
            
            
            
        end
    end
    
    
    
    FaceNum = 1;
    
    
    
    
    
    
    % EXTRACTION
    
    
    
    for j = 1 : subjects
        
        for i = 1 : training
            
            
            trainface = storeface1{FaceNum};
            
            
            trainface = trainface';
            trainface = double(reshape(trainface,1,height * width));        % Raster scanning and converting the face to a row vector
            
            
            
            for( decomp = 1 : NoOfDecompositions )
                trainface = dwt(trainface,wavelet);                             % DWT
            end
            [RowsAfterDwt,ColsAfterDwt] = size(trainface);
            
            
            temptrainface = dct(trainface);                                             % DCT
            
            trainface = temptrainface(1,1:height * DctFact + Adjust );
            
            
            
            
            [yrow,xcol] = size(trainface);
            
            
            
            if(j ==1 && i == 1)
                storeface = zeros(1, yrow*xcol , subjects*training);
            end
            
            if( j == 1 && i == 1 )
                TrainingTotal = zeros(1,yrow .* xcol);
            end
            
            if( i == 1)
                SubjectTotal = zeros(1,yrow .* xcol);
            end
            
            
            
            storeface(:,:,FaceNum) = trainface;                                   % storeface contains the extracted face row vectors
            FaceNum =  FaceNum+1;
            
            SubjectTotal = double(SubjectTotal)+double(trainface);
            
        end
        SubjectAvg = SubjectTotal ./ training ;
        classmeanarray(:,:,j) = SubjectAvg;
        
        TrainingTotal = double(TrainingTotal) + double(SubjectTotal);
    end
    TrainingAvg = TrainingTotal ./ subjects ./ training ;
    totalmeanarray = TrainingAvg;
    
    FaceNum = 0;
    
    SubjectVariance = zeros(1,yrow*xcol,subjects);
    DatabaseVariance = zeros(1,yrow*xcol);
    for j = 1 : subjects
        
        for i = 1 : training
            
            FaceNum = FaceNum + 1;
            SubjectVariance(:,:,j) = SubjectVariance(:,:,j) + abs(storeface(:,:,FaceNum) - classmeanarray(:,:,j)) .^ 2;
            
        end
        SubjectVariance(:,:,j) = sqrt(SubjectVariance(:,:,j)) ./ training;
    end
    
    SV(:,:) = zeros(1,xcol .* yrow);
    for j = 1 : subjects
        
        DatabaseVariance(:,:) = DatabaseVariance(:,:) + abs(totalmeanarray(:,:) - classmeanarray(:,:,j)) .^ 2;
        SV(:,:) = SV(:,:) + SubjectVariance(:,:,j) ;
        
    end
    
    DatabaseVariance(:,:) =  sqrt(DatabaseVariance(:,:)) ./ subjects;
    
    SV(:,:) = SV(:,:) ./ subjects ;
    
    
    
    
    
    NumExtracted = (xcol .* yrow);
    
    
    
    if(Selector ~= 2)
        FinalGlobalBest = SelectorFunc(subjects,yrow*xcol);
        GlobalBestP = FinalGlobalBest ;
        
    else
        %% BPSO
        
        % BPSO parameters
        
        NumofParticles = 30;
        MaxIterations = 10;
        Damp = 0.9;
        C1 = 0.6;
        C2 = 1.6;
        
        ff='FIT_FUNC';
        
        GlobalBestP = round(rand(1,NumExtracted));
        GlobalBestC = 0;
        
        change = 0;
        
        
        
        
        % Particles initialization
        for n = 1:NumofParticles
            
            Particle(:,:,Velocity,n)= (rand(1,NumExtracted));
            R = rand(1,NumExtracted);
            Particle(:,:,Position,n) = R < IniSigmoidalPar ./ (1 + exp(-Particle(:,:,Velocity,n)));  % assigning a (yrow * xcol) bit binary string as the particle's position.
            Particle(1,1,Cost,n) = feval(ff,Particle(:,:,Position,n),subjects,NumExtracted);   % evaluating the fitness of a particular particle
            Particle(:,:,LocalBestPosition,n) = Particle(:,:,Position,n);              % initially, this is the best position of particle
            Particle(1,1,LocalBestCost,n) = Particle(1,1,Cost,n);                      % initially, the corresponding fitness is the best fitness of particle
            
            
            if (Particle(1,1,Cost,n) > GlobalBestC)                                    % assigning the position of the particle with the best position
                % as the global best position.
                GlobalBestP = Particle(:,:,Position,n);
                GlobalBestC = Particle(1,1,Cost,n);
                
            end
            
        end
        
        % Iterative Selection
        
        for t = 1:MaxIterations
            
            for n = 1 : NumofParticles
                
                r1 = rand(1,NumExtracted);
                r2 = rand(1,NumExtracted);
                w = rand(1,NumExtracted);
                
                Particle(:,:,Velocity,n) = Damp .* Particle(:,:,Velocity,n) - ...                % Velocity equation
                    r1*C1.*(Particle(:,:,LocalBestPosition,n) - Particle(:,:,Position,n)) - ...
                    r2*C2.*(GlobalBestP - Particle(:,:,Position,n));
                
                
                R = rand(1,NumExtracted);
                Particle(:,:,Position,n) = R < SigmoidalPar ./ (1 + exp(-Particle(:,:,Velocity,n)));       % Position equation
                Particle(1,1,Cost,n) = feval(ff,Particle(:,:,Position,n),subjects,NumExtracted);
                
                if (Particle(1,1,Cost,n) > Particle(1,1,LocalBestCost,n))                       % Updating particle's best position
                    Particle(:,:,LocalBestPosition,n) = Particle(:,:,Position,n);               % and the corresponding fitness.
                    Particle(1,1,LocalBestCost,n) = Particle(1,1,Cost,n);
                end
                
                if (Particle(1,1,Cost,n) > GlobalBestC)                                         % Updating global best position
                    GlobalBestP = Particle(:,:,Position,n);                                     % and the corresponding fitness.
                    GlobalBestC = Particle(1,1,Cost,n);
                    change = change + 1;
                end
                
                
            end
        end
        
        iniGlobalBestP = GlobalBestP;
        finalbest = ones(1,NumExtracted);
        for(s1 = 1 : NumofParticles - 1)
            
            for(s2 = (s1 + 1) :  NumofParticles)
                
                if (  Particle(1,1,LocalBestCost,s1) < Particle(1,1,LocalBestCost,s2) )
                    
                    sorttemp = Particle(:,:,LocalBestPosition,s1) ;
                    Particle(:,:,LocalBestPosition,s1) = Particle(:,:,LocalBestPosition,s2) ;
                    Particle(:,:,LocalBestPosition,s2) = sorttemp ;
                    
                    sorttemp = Particle(1,1,LocalBestCost,s1) ;
                    Particle(1,1,LocalBestCost,s1) = Particle(1,1,LocalBestCost,s2) ;
                    Particle(1,1,LocalBestCost,s2) = sorttemp ;
                    
                end
                
            end
            
        end
        
        
        for( thresh = 1 : Threshold )
            finalbest = finalbest .* Particle(:,:,LocalBestPosition,thresh);
        end
        GlobalBestP = finalbest;
    end
    
    
    %       for( ij = 1 : subjects .* training)
    %           storeface(:,:, ij) = storeface(:,:, ij)  .* GlobalBestP;
    %       end
    
    GalleryCount = 0;
    for FeatNum = 1 : NumExtracted
        if GlobalBestP(:,FeatNum) == 1
            GalleryCount = GalleryCount + 1;
            Gallery(GalleryCount) = FeatNum;
        end
    end
    
    
    FeaturesSelected(x) = GalleryCount
    
    gallery = storeface;
    
    trainingtime(x) =  toc(trainingtemp);                                        % Training complete
    
    
    
    
    %% RECOGNITION
    
    hits = 0;
    
    if(x == 1)
        testingtime = zeros(1,trials);
    end
    
    testingtemp = tic;
    
    
    
    proceed = 0;
    for j = 1 : subjects
        
        FullSet = 1:imagespersubject;
        TestingSet(1,:,j) = setdiff( FullSet , TrainingSet(1,:,j) );
        
        for i = 1:testing
            
            face = imread(strcat(database,num2str(j),'\',num2str(TestingSet(1,i,j)),ext));
            
            
            % PREPARING FOR PRE-PROCESSING
            
            
            face = rgb2gray(face);
            %  trainface = .3  .*  trainface(:,:,1)       +        .5  .*   trainface(:,:,2)      +     .11 .*  trainface(:,:,3);
            
            
            
            
            face = (imresize(face, 1 ./ (2 .^ (DatabaseChoice ))));
            
            [height,width] = size(face);
            
            %              face = ScaleFunc(face);
            
            
            facetemp = ScaleNorm(face,FilterWidth,EdgeSelectivity1,EdgeSelectivity2);
            face = imresize(facetemp, [ height width ]);
            
            
            flipface = fliplr( face);                                        % flip the face about the vertical axis
            
            
            face = face';
            flipface = flipface';
            % raster scan to form a FACE and FLIPFACE row vector
            face = double(reshape(face,1,height * width));
            flipface = double(reshape(flipface,1,height * width));
            
            
            for(decomp = 1 : NoOfDecompositions)
                face = dwt(face,wavelet);
                flipface = dwt(flipface,wavelet);                              % DWT
            end
            [RowsAfterDwt,ColsAfterDwt] = size(face);
            
            
            tempface = dct(face);
            %             plot(tempface);
            %             pause(1);
            face = tempface(1,1:height * DctFact + Adjust );
            %             face = tempface(1,1:208 );
            % DCT
            tempflipface = dct(flipface);
            flipface = tempflipface(1,1:height * DctFact + Adjust );
            %              face = tempface(1,1:208 );
            
            
            dist = zeros(1, subjects * training);
            flipdist = zeros(1, subjects * training);
            
            
            
            
            for testnum = 1:subjects * training                             % Computing Euclidian distance
                
                
                for cellnum = 1 : GalleryCount
                    
                    dist(testnum) = dist(testnum) + ( ( ( (gallery(1,Gallery(cellnum),testnum) - face(1,Gallery(cellnum)))   ) .^ 2 )   );
                    flipdist(testnum) = flipdist(testnum) + ( ( ( (gallery(1,Gallery(cellnum),testnum) - flipface(1,Gallery(cellnum)))   ) .^ 2 )   );
                    
                end
                
                dist(testnum) = sqrt(dist(testnum));
                flipdist(testnum) = sqrt(flipdist(testnum));
                
            end
            round(dist);
            % Finding the min Euclidian distance and
            % corresponding subject in the gallery
            
            
            
            [val,index] = min(dist);
            
            
            [flipval,flipindex] = min(flipdist);
            
            
            
            ProbSub1 =  ceil(index/training);
            round( dist( ((ProbSub1-1) .* training) + 1 : ((ProbSub1-1) .*  training) + training ));
            
            ProbSub2 =  ceil(flipindex/training);
            round( flipdist( ((ProbSub2-1) .* training) + 1 : ((ProbSub2-1) .*  training) + training ));
            
            
            
            if(ProbSub1 == j)
                hits = hits + 1;
            else
                if(ProbSub2 == j)
                    hits = hits + 1;
                end
            end
            
            
        end
        
        
    end
    
    
    
    
    testingtime(x) = (toc(testingtemp)) / subjects / testing ;
    
    
    
    %% DISPLAY the result of the trial
    
    
    disp('Features extracted = ');
    disp(yrow * xcol);
    
    % disp('IniFeatures selected = ');
    % disp(sum(iniGlobalBestP));
    
    disp('Features selected = ');
    disp(FeaturesSelected(x));
    
    disp('The recognition rate');
    percent = (hits  / (subjects * testing)) * 100;
    
    disp(percent);
    
    disp('The avg recognition rate');
    tempp = percent;
    
    for tempc = 1 : x-1
        tempp = tempp + percentsum(tempc);
    end
    disp(tempp/x);
    
    percentsum(x) = percent;
    
    
end


%% DISPLAY FULL RESULTS


disp(databasename);

disp('No of subjects');
disp(subjects);

disp('No of training images per subject = ');
disp(training);

disp('No of testing images per subject = ');
disp(testing);

% disp('Original image size');
% disp(originalheight);
% disp(originalwidth);

if(NoOfDecompositions ~= 0)
    disp(' 1 dimensional DWT');
    disp(wavelet);
    disp('NoOfDecompositions = ')
    disp(NoOfDecompositions);
end

disp(' 1 dimensional DCT');

disp('Features extracted = ');
disp(yrow * xcol);

disp('Features selected = ');
disp((sum(FeaturesSelected))./trials);

disp('Average Recognition Rate = ');
disp(sum(percentsum)/trials);

disp('Total training time in seconds');
disp((sum(trainingtime))/trials);

disp('Testing time per image in seconds');
disp((sum(testingtime))/(trials));


