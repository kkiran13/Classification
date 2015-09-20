load trainlr.mat
load filesplit.mat

Activation_Test = Phi_Test*Weight_Train;
Numer = exp(Activation_Test);
YMAT = Numer;

for i = 1:size(Numer,1)
		YMAT(i,:) = Numer(i,:)./sum(Numer(i,:));
	end
	
%Wei = transpose(Phi_Test) *(YMAT - Target_Test);
cnt = 0;
[yx,I1] = max(YMAT,[],2);
[tx,I2] = max(Target_Test,[],2);
for j = 1:1500
	if I1(j,1) == I2(j,1)
	cnt = cnt + 1;
	end
end
accuracy_lr = cnt/1500;
save testlr.mat