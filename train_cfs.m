%function [ERMS_VAL_CFS,LAMBDA_VAL_CFS] = train_cfs()
load filesplitdata.mat
load DesignMatrixdata_cfs.mat
%%%%%%%%%%%%%%%%%%%%%%%%%
%Lambda = 0.7;
for m = 2:9
E1_Validation = (BPhi_Validation*W-Target_Validation)'*(BPhi_Validation*W-Target_Validation);
E2_Validation = 0.5*Lambda*(W'*W);
E_Validation =  E1_Validation + E2_Validation;
ERMS_Validation = sqrt(E_Validation/Num_Rows_Validation);
%ERMS_VAL_CFS(M,m) = ERMS_Validation;
ERMS_VAL_CFS(M,m) = ERMS_Validation;
LAMBDA_VAL_CFS(M,m) = Lambda;
%LAMBDA_VAL_CFS(M,m) = Lambda;
Lambda = Lambda - 1;
end;
Lamtest = Lambda;
%figure;
%hold on;
%axis([-20 30 0 1]);
%LAMBDA_VAL_CFS_LN = log2(LAMBDA_VAL_CFS);
%PLOT_CFS = plot(LAMBDA_VAL_CFS_LN,ERMS_VAL_CFS,'-bs','Linewidth',1.4,'Markersize',5);
%xlabel ('LambdaLN'),ylabel ('ERMS');
%title ('EMRSvsLAMBDA');
%cdata = print(fig,'-RGBImage');
%hold off;
save trainingdata_cfs.mat