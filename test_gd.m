load trainingdata_gd.mat
load DesignMatrixdata_gd.mat
[row,col] = find(ERMS_Validation_Vector_GD == min(ERMS_Validation_Vector_GD(:)));
[max, position] = min(ERMS_Validation_Vector_GD(:));
Wgd = Weight_Vector_GD(:,position);

E1_Testing_GD = (BPhi_Testing*Wgd-Target_Testing)'*(BPhi_Testing*Wgd-Target_Testing);
E2_Testing_GD = 0.5*Lambda_gd*(Wgd'*Wgd);
E_Testing_GD =  E1_Testing_GD + E2_Testing_GD;
ERMS_Testing_GD = sqrt(E_Testing_GD/Num_Rows_Testing);
rms_gd = ERMS_Testing_GD;
lambda_gd = Lambda_gd;
M_gd=M;
W_gd = Wgd;
save W_gd.mat
save testingdata_gd.mat



