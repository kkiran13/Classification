clc
clear
fid = fopen('project1_data.txt');   
%Format to meet requirements
fmt=['%f %*s' repmat('%*d:%f',1,46) '%*[^\n]'];   
%Scan file and store in Cells
C = (textscan(fid,fmt));     
fclose(fid); 
%Input Data Matrix retrieved from file            
N = cell2mat(C);
save project1_data.mat N
% Returns number of rows in Matrix
Num_Rows = size(N,1);   
% 64632*41 Dimensional Input Matrix
R = N(:,[2:6,12:47]);  
T = N(:,1);
Train_set_count = round((80/100)*Num_Rows);    
Validation_set_count = round((10/100)*Num_Rows);  
Target_Training = T(1:Train_set_count,:);
Target_Validation = T(Train_set_count+1:Train_set_count+Validation_set_count,:);
Target_Testing = T(Train_set_count+Validation_set_count+1:end,:);
Training_Set = R(1:Train_set_count,:);   % 80% of Dataset as Training Set
Validation_Set = R(Train_set_count+1:Train_set_count+Validation_set_count,:);   % 10% of Dataset as Validation Set
Testing_Set = R(Train_set_count+Validation_set_count+1:end,:);   % 10% of Dataset as Testing Set
save filesplitdata.mat


for M = 2:9
DesignMatrix_cfs(M,0.85,10)
train_cfs
test_cfs
end;
M = 9;
DesignMatrix_gd(M,0.9,1)
train_gd
test_gd
surf(ERMS_VAL_CFS(2:end,:));
fprintf('My ubit name is %s\n','kkiran');
fprintf('My student number is %d\n',50133145);
fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters lambda_gd is %4.2f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', rms_gd);
