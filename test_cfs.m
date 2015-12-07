load trainingdata_cfs.mat
load DesignMatrixdata_cfs.mat
E1_Testing = (BPhi_Testing*W-Target_Testing)'*(BPhi_Testing*W-Target_Testing);
E2_Testing = 0.5*Lamtest*(W'*W);
E_Testing =  E1_Testing + E2_Testing;
ERMS_Testing_cfs = sqrt(E_Testing/Num_Rows_Testing);
rms_cfs = ERMS_Testing_cfs;
lambda_cfs = Lamtest;
M_cfs = M;
W_cfs = W;
save W_cfs.mat
save testingdata_cfs.mat