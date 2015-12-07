load filesplitdata.mat
load DesignMatrixdata_gd.mat
%load trainingdata_cfs.mat
eeta = 1;
M_gd = M;
Lambda_gd = L;
Wgd = zeros(M_gd,1);
ERMS_Validation_Vector_GD(1) = 4;
for i = 2:Train_set_count
%Calculate Weight for each training Phi Matrix Row
Wgd = Wgd + eeta*(Target_Training(i) - Wgd'*BPhi_Training(i,:)')*BPhi_Training(i,:)';
%Calculate Error for Phi Matrix of Validation set
E1_Validation_GD = (BPhi_Validation*Wgd-Target_Validation)'*(BPhi_Validation*Wgd-Target_Validation);
E2_Validation_GD = 0.5*Lambda_gd*(Wgd'*Wgd);
E_Validation_GD =  E1_Validation_GD + E2_Validation_GD;
ERMS_Validation_GD = sqrt(E_Validation_GD/Num_Rows_Validation);
ERMS_Validation_Vector_GD(i) = ERMS_Validation_GD;
Weight_Vector_GD(:,i) = Wgd;
if ERMS_Validation_Vector_GD(i) < ERMS_Validation_Vector_GD(i-1)
eeta = eeta;
else
eeta = 0.5*eeta;
end;
end;
save trainingdata_gd.mat

