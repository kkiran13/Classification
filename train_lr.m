load filesplit.mat
Numer = exp(Activation_Train);
YMAT = Numer;

for i = 1:size(Numer,1)
    YMAT(i,:) = Numer(i,:)/sum(Numer(i,:));
end;

eta = 0.001;
%Wei = [];
Wei = transpose(Phi_Train) *(YMAT - Target_Train);
%Acc = [];

cnt = 0;
[yx,I1] = max(YMAT,[],2);
[tx,I2] = max(Target_Train,[],2);
for j = 1:19978
	if I1(j,1) == I2(j,1)
	cnt = cnt + 1;
	end
end
acc_prev = cnt/19978;

for k = 1 :100
   W_new = W - eta*Wei;
   W = W_new;
   Activation_Train = Phi_Train*W;
   Numer = exp(Activation_Train);
   YMAT = Numer;
   
	for l = 1:size(Numer,1)
		YMAT(l,:) = Numer(l,:)./sum(Numer(l,:));
	end
	
	[yx,I1] = max(YMAT,[],2);
	[tx,I2] = max(Target_Train,[],2);
	cnt = 0;
	Wei = transpose(Phi_Train) *(YMAT - Target_Train);
	
	for j = 1:19978
		if I1(j,1) == I2(j,1)
		 cnt = cnt + 1;
		end
		end
	acc_next = cnt/19978;

	if acc_prev < acc_next
		eta = 1.1 * eta;
	else
		eta = 0.5*eta;
	end
	acc_prev = acc_next;
	Weight_Train = W;
end
save trainlr.mat