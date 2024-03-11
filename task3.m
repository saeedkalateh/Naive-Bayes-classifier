clear
clc
%% Initialize variables.
filename = 'C:\Users\Navid\Desktop\Saeed\Assignment1\data.txt';
delimiter = '\t';
formatSpec = '%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
data = [dataArray{1:end-1}];

[rows, columns]=size(data);
p_yes=sum((data(:,5)==1))/rows;
p_no=1-p_yes;
clearvars dataArray formatSpec filename fileID delimiter
%% task3: number of levels
a=1;
for i=1:columns-1
    levels(i)=max(data(:,i));
end
clearvars dataArray formatSpec filename fileID delimiter i
%% Dividing data set into train and test
bool=0;
while(bool==0)
    n=input("Number of training examples: \n");
    if n<rows && n>3
        bool=1;
    end
end
A=randperm(rows,rows);
train_index=A(1:n);
test_index=A(n+1:rows);
for i=1:length(train_index)
    data_train(i,:)=data(train_index(i),:);
end
for i=1:length(test_index)
    data_test(i,:)=data(test_index(i),:);
end
clearvars A bool i train_index test_index
%% Creating Lookup Table
% The order is like this:
% Each column: p(feature|yes), p(feature|no)
%cond_lookup_table=zeros(10,columns-1);
% Feature columns: p(feature)
%f_prob=zeros(10,columns-1);
%outlook
v=levels(1);
f_prob(1,1)=sum(data_train(:,1) == 1)/rows;
lookup_table(1,1)=(sum(((data_train(:,1) == 1) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(1,2)=(sum(((data_train(:,1) == 1) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(2,1)=sum(data_train(:,1) == 2)/rows;
lookup_table(2,1)=(sum(((data_train(:,1) == 2) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(2,2)=(sum(((data_train(:,1) == 2) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(3,1)=sum(data_train(:,1) == 3)/rows;
lookup_table(3,1)=(sum(((data_train(:,1) == 3) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(3,2)=(sum(((data_train(:,1) == 3) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

%temperature: 1. hot 2. cold 3. mild
v=levels(2);
f_prob(1,2)=sum(data_train(:,2) == 1)/rows;
lookup_table(1,3)=(sum(((data_train(:,2) == 1) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(1,4)=(sum(((data_train(:,2) == 1) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(2,2)=sum(data_train(:,2) == 2)/rows;
lookup_table(2,3)=(sum(((data_train(:,2) == 2) .* (data_train(:,5)==1))==1)+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(2,4)=(sum(((data_train(:,2) == 2) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(3,2)=sum(data_train(:,2) == 3)/rows;
lookup_table(3,3)=(sum(((data_train(:,2) == 3) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(3,4)=(sum(((data_train(:,2) == 3) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

%humidity: 1. high 2. low
v=levels(3);
f_prob(1,3)=sum(data_train(:,3) == 1)/rows;
lookup_table(1,5)=(sum(((data_train(:,3) == 1) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(1,6)=(sum(((data_train(:,3) == 1) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(2,3)=sum(data_train(:,3) == 2)/rows;
lookup_table(2,5)=(sum(((data_train(:,3) == 2) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(2,6)=(sum(((data_train(:,3) == 2) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

%windy: 1. True 2. False
v=levels(4);
f_prob(1,4)=sum(data_train(:,4) == 2)/rows;
lookup_table(1,7)=(sum(((data_train(:,4) == 2) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(1,8)=(sum(((data_train(:,4) == 2) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

f_prob(2,4)=sum(data_train(:,4) == 1)/rows;
lookup_table(2,7)=(sum(((data_train(:,4) == 1) .* (data_train(:,5)==1)))+a)/(sum(data_train(:,5) == 1)+a*v);
lookup_table(2,8)=(sum(((data_train(:,4) == 1) .* (data_train(:,5)==2)))+a)/(sum(data_train(:,5) == 2)+a*v);

%% Time to test our Naive Bayes
[r,c]=size(data_test);
for j=1:r
% getting the test data
mat=data_test(j,:);
target=1;
num=1;
denum=1;
for i=1:columns-1
num=num*lookup_table(mat(i), 2*(i-1)+target);
denum=denum*f_prob(mat(i),i);
end
prob_yes=(num*p_yes)/denum;

target=2;
num=1;
denum=1;
for i=1:columns-1
num=num*lookup_table(mat(i), 2*(i-1)+target);
denum=denum*f_prob(mat(i),i);
end

prob_no=(num*p_no)/denum;

%validation=zeros(rows, columns+1);
validation(j,1:5)=data_test(j,1:5);
if prob_yes>prob_no
    validation(j, 6)=1;
else
    validation(j, 6)=2;
end
end
clearvars num denum c r  p_yes p_no i j mat n ans
clearvars rows columns target prob_yes prob_no
%% Print the results
error_matrix=validation(:,5)==validation(:,6);
accuracy=sum(error_matrix)/length(error_matrix);
clearvars error_matrix
display("Accuracy")
