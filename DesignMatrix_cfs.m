function [] = DesignMatrix_cfs(M,S,L)
load filesplitdata.mat
Num_Columns_Validation = size(Validation_Set,2);
Num_Rows_Validation = size(Validation_Set,1);
one_validation = ones(Validation_set_count,1);
BPhi_Validation = [one_validation];
Num_Rows_Training = size(Training_Set,1);
I = eye(41,41);
one_training = ones(Train_set_count,1);
BPhi_Training = [one_training];
V_train = 2000;
Num_Rows_Testing = size(Testing_Set,1);
one_testing = ones(Num_Rows_Testing,1);
BPhi_Testing = [one_testing];
M = M;
S = S;
V_validation = 0;

%mu_cfs_Training = mean(Training_Set);
mu_cfs_Training = mean(R);
%mu_cfs_Validation = mean(Validation_Set);
mu_cfs_Validation = mu_cfs_Training;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Program %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:M-1
%V_train = 1000*i;
%B_train = V_train+3000;
%mu_cfs_Training = mean(Training_Set(V_train:B_train,:));
mu_cfs_Training = 1.9*mu_cfs_Training;
S = S.^2*I;
Z = inv(S);
for j = 1:Num_Rows_Training
AT = Training_Set(j,:) - mu_cfs_Training;
format long;
Phi_Training(j,:) = exp(-(AT*Z*AT'));
Q_Training = Phi_Training;
end;
BPhi_Training = [BPhi_Training Q_Training];
Lambda = L;
Bphirows_Training = size(BPhi_Training'*BPhi_Training,1);
Bphicols_Training = size(BPhi_Training'*BPhi_Training,2);
ID_Training = eye(Bphirows_Training,Bphicols_Training);
%for k = 1:9
ID_Train = Lambda*ID_Training;
W = inv(ID_Train+(BPhi_Training'*BPhi_Training))*(BPhi_Training'*Target_Training);
%Lambda = Lambda - 0.05;
%end;

%V_validation = V_validation + 400;
%B_validation = V_validation + 500;
%mu_cfs_Validation = mean(Validation_Set(V_validation:B_validation,:));
mu_cfs_Validation = 1.9*mu_cfs_Validation;
for l = 1:Num_Rows_Validation
AV = Validation_Set(l,:) - mu_cfs_Validation;
format long;
Phi_Validation(l,:) = exp(-(AV*Z*AV'));
Q_Validation = Phi_Validation;
end;
BPhi_Validation = [BPhi_Validation Q_Validation];
end;

for o = 1:M-1
for p = 1:Num_Rows_Testing
ATS = Testing_Set(p,:) - mu_cfs_Validation;
format long;
Phi_Testing(p,:) = exp(-(ATS*Z*ATS'));
Q_Testing = Phi_Testing;
end;
BPhi_Testing = [BPhi_Testing Q_Testing];
end;
mu_cfs = mu_cfs_Validation;
s_cfs = S;
save s_cfs.mat
save DesignMatrixdata_cfs.mat
save mu_cfs.mat
end