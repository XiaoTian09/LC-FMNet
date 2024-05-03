
clear;
clc;
close all;

load('./y_pred.mat');


np_out=[128,128,128];

strike_beg=0;
strike_end=360;
dip_beg=0;
dip_end=90;
rake_beg=-180;
rake_end=180;

numev=size(y_pred,1);
for i=1:numev
        
        % get the index of maximum values
        true_flag_strike=find(y_test(i,:,1)==max(y_test(i,:,1)));
        true_flag_dip=find(y_test(i,:,2)==max(y_test(i,:,2)));
        true_flag_rake=find(y_test(i,:,3)==max(y_test(i,:,3)));
        pred_flag_strike=find(y_pred(i,:,1)==max(y_pred(i,:,1)));
        pred_flag_dip=find(y_pred(i,:,2)==max(y_pred(i,:,2)));
        pred_flag_rake=find(y_pred(i,:,3)==max(y_pred(i,:,3)));

        % convert index to strike, dip, and rake
        true_strike(i)=mapping(true_flag_strike,24,104,0,360);
        true_dip(i)=mapping(true_flag_dip,24,104,0,90);
        true_rake(i)=mapping(true_flag_rake,24,104,-180,180);
        pred_strike(i)=mapping(pred_flag_strike,24,104,0,360);
        pred_dip(i)=mapping(pred_flag_dip,24,104,0,90);
        pred_rake(i)=mapping(pred_flag_rake,24,104,-180,180);


        
        % calculate the Kagan angle (subroutine is from by matlab open source library)
        [rotangle,theta,phi]=sub_kang([true_strike(i),true_dip(i),true_rake(i)],[pred_strike(i),pred_dip(i),pred_rake(i)]);
      if (isreal(rotangle)==0)
          disp(i)
      end        
        kang(i)=rotangle;
        
        
end

% calculate percentage (kagan angle <= 20 degree)
kang=real(kang);
num1_kang=find(kang<=20);
num2_kang=find(kang>20);
fprintf('Good is %f,  bad is %f\n',length(num1_kang)/numev,length(num2_kang)/numev);

%% plot the histogram of Kagan angles
figure;
histogram(real(kang),100);



