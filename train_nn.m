fid = fopen('0_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N0 = cell2mat(C);
fid = fopen('1_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N1 = cell2mat(C);
fid = fopen('2_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N2 = cell2mat(C);
fid = fopen('3_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N3 = cell2mat(C);
fid = fopen('4_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N4 = cell2mat(C);
fid = fopen('5_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N5 = cell2mat(C);
fid = fopen('6_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N6 = cell2mat(C);
fid = fopen('7_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N7 = cell2mat(C);
fid = fopen('8_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N8 = cell2mat(C);
fid = fopen('9_train.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N9 = cell2mat(C);
N = [N0;N1;N2;N3;N4;N5;N6;N7;N8;N9];
Nnn = double(N);

zero = repmat(0,size(N0,1),1);
one = repmat(1,size(N1,1),1);
two = repmat(2,size(N2,1),1);
three = repmat(3,size(N3,1),1);
four = repmat(4,size(N4,1),1);
five = repmat(5,size(N5,1),1);
six = repmat(6,size(N6,1),1);
seven = repmat(7,size(N7,1),1);
eight = repmat(8,size(N8,1),1);
nine = repmat(9,size(N9,1),1);

label = [zero;one;two;three;four;five;six;seven;eight;nine];

x = randperm(size(Nnn,1));

trainlabel = label(x(1:15982),:);
trainset = Nnn(x(1:15982),:);
train_set_count=size(trainset,1);

vallabel = label(x(15983:end),:);
validation_set= Nnn(x(15983:end),:);
input_node = size(trainset, 2); 

trainlabelen=[];
    for k = 1 : size(trainlabel,1)
        if(trainlabel(k))== 0
         train_label = [1 0 0 0 0 0 0 0 0 0];
         elseif (trainlabel(k)== 1)
          train_label = [0 1 0 0 0 0 0 0 0 0];
         elseif (trainlabel(k)== 2)
         train_label = [0 0 1 0 0 0 0 0 0 0];
         elseif (trainlabel(k)== 3)
         train_label = [0 0 0 1 0 0 0 0 0 0];
         elseif (trainlabel(k)== 4)
         train_label = [0 0 0 0 1 0 0 0 0 0];
         elseif (trainlabel(k)== 5)
         train_label = [0 0 0 0 0 1 0 0 0 0];
         elseif (trainlabel(k)== 6)
         train_label = [0 0 0 0 0 0 1 0 0 0];
         elseif (trainlabel(k)== 7)
         train_label = [0 0 0 0 0 0 0 1 0 0];
         elseif (trainlabel(k)== 8)
         train_label = [0 0 0 0 0 0 0 0 1 0];
          else
         train_label = [0 0 0 0 0 0 0 0 0 1];
         end
         trainlabelen = [trainlabelen; train_label];
    end
lambda = 0.05;
hid_node = 200;		   
sink_node = 10;		   
bias = ones(size(trainset,1),1);
X = horzcat(bias,trainset);
alpha = 0.15;
W1 = rand(hid_node,513)*0.2-0.1;
W2 = rand(sink_node,201)*0.5-0.25;

for j = 1 : 100
       aj = X*W1';
        z = double(1./(1.0+exp(-1*aj)));
        
        bs1 = ones(size(z,1),1);
        b = [bs1 z];
        b1=b*transpose(W2);
        ak = double(1./(1.0+exp(-1*b1)));
          
		deltaw2 = ak - trainlabelen;
		gdw2 = deltaw2'*b;
		gdw2_reg= (lambda*W2);
		gdw2= (gdw2 + gdw2_reg)/train_set_count;
  
		deltaw1 = ak - trainlabelen;
		del_wt=deltaw1*W2;
		po=((1-b).*b).*del_wt;
		gd1r = po'*X;
		gdw1=(gd1r(1:(size(gd1r,1)-1),:));
    	gd1r = (lambda*W1);
		gdw1= (gdw1 + gd1r)/train_set_count;
    	W2=(W2-(alpha*gdw2));
		W1=(W1-(alpha*gdw1));
end

bst = ones(size(validation_set,1),1);
        validation_set = [bst, validation_set];
       
        testhid = validation_set*W1';
        z=double(1./(1.0+exp(-1*testhid)));
       
        bst1 = ones(size(z,1),1);
        b = [bst1 z];
        
        tout = b*W2';
        tout1=double(1./(1.0+exp(-1*tout)));
        
        pt=[];
 
        for k=1:size(validation_set,1)
            class=tout1(k,:);
            classlab = find(class==(max(max(class))));
            classlab=classlab-1;
            pt= [pt ; classlab];         
        end
        accuracy_nn = mean(double(pt == vallabel));
		save trainnn.mat