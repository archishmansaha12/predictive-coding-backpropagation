clc;
clear;

%% -----------------import data---------------

backprop_data = readmatrix('CIFAR10_accuracy_data.xlsx','Sheet','BackpropagationCNN','Range','A2:C71');
PC_data = readmatrix('CIFAR10_accuracy_data.xlsx','Sheet','PredictiveCoding','Range','A2:C71');

backprop_train = backprop_data(:,2);
backprop_test = backprop_data(:,3);
PC_train = PC_data(:,2);
PC_test = PC_data(:,3);
epoch = backprop_data(:,1);
N = length(epoch);

%% ------------------plotting----------------------

figure;
plot(epoch,PC_train,'g--','linewidth',1.1);
hold on;
plot(epoch,backprop_train,'r--','linewidth',1.1);
hold on;
plot(epoch,PC_test,'g','linewidth',1.1);
hold on;
plot(epoch,backprop_test,'r','linewidth',1.1);
xlim([0 N]);
ylim([0 1]);
xlabel('Epoch');
ylabel('Accuracy');
legend('Predictive coding train accuracy','Backpropagation train accuracy','Predictive coding test accuracy','Backpropagation test accuracy');
title('CIFAR-10 CNN performance');
