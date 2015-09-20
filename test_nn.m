load trainnn.mat
fid = fopen('0_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N0 = cell2mat(C);
fid = fopen('1_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N1 = cell2mat(C);
fid = fopen('2_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N2 = cell2mat(C);
fid = fopen('3_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N3 = cell2mat(C);
fid = fopen('4_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N4 = cell2mat(C);
fid = fopen('5_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N5 = cell2mat(C);
fid = fopen('6_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N6 = cell2mat(C);
fid = fopen('7_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N7 = cell2mat(C);
fid = fopen('8_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N8 = cell2mat(C);
fid = fopen('9_test.txt');
fmt=[repmat('%d',1,512) '%*[^\n]'];
C = (textscan(fid,fmt));     
fclose(fid);
N9 = cell2mat(C);
N = [N0;N1;N2;N3;N4;N5;N6;N7;N8;N9];
Ntest = double(N);
test_set = Ntest;
        test_set = [ones(size(test_set,1),1), test_set];
       x = randperm(size(test_set,1));
	  testlabel = label(x(1:1500),:);
	  testset = test_set(x(1:1500),:);
        testhid = test_set*W1';
        z=double(1./(1.0+exp(-1*testhid)));
       
        b = [ones(size(z,1),1) z];
        
        tout = b*W2';
        tout1=double(1./(1.0+exp(-1*tout)));
        
        pt=[];
 
        for k=1:size(test_set,1)
            class=tout1(k,:);
            classlab = find(class==(max(max(class))));
            classlab=classlab-1;
            pt= [pt ; classlab];         
        end
        accuracytest_nn = 1- mean(double(pt == testlabel));