%This function estimate the fdc with the Madsen method, based on the 
%statistic of the signs of I and Q, hypotized independent gaussians with zero mean. 
%The components I and Q selected are relative to an estimation window, 
%which is a matrix 2*wx x wr (respectively byte in range and pulses in azimuth) 
%The author suggests to select k=1, because R0_h(k) tends to zero very 
%fast when abs(k) increases. 
 
function [fc]=Madsen(window,k,PRF) 
 
PRT=1./PRF; 
 
%[twowr,wx]=size(window); 
[wr,wx]=size(window); 
%wr=twowr/2; 
for j=1:1:wx % wx azimuth complex samples 
    for i=1:1:wr % wr range complex sample 
        if real(window(i,j))>=0%   window(2*i-1,j)>=0 
             I(i,j)=1; 
        else I(i,j)=-1; 
        end 
    end 
end % I is a matrix wr x wx, containing the phase elements 
for j=1:1:wx 
    for i=1:1:wr 
        if imag(window(i,j))>=0%  window(2*i,j)>=0 
             Q(i,j)=1; 
        else Q(i,j)=-1; 
        end 
    end 
end % Q is a matrix wr x wx, containing the quadrature elements    
II=0; 
QQ=0; 
IQ=0; 
QI=0; 
 
%Estimation of the mean value of the correlation between azimuth lines (between the columns of I and Q), that 
%are far from them of k pulses.  
for j=1:1:(wx-k) 
    for i=1:1:wr 
        II=II+I(i,j+k).*I(i,j); 
        QQ=QQ+Q(i,j+k).*Q(i,j); 
        IQ=IQ+I(i,j+k).*Q(i,j); 
        QI=QI+Q(i,j+k).*I(i,j); 
    end 
end 
 
R_II=II/(wr*(wx-1)); % autocorrelation function of I 
R_QQ=QQ/(wr*(wx-1)); % autocorrelation function of Q 
R_IQ=IQ/(wr*(wx-1)); % correlation function of I and Q 
R_QI=QI/(wr*(wx-1)); % correlation function of Q and I 
RO_II=sin((pi/2)*R_II); % autocorrelation coefficient of I    
RO_QQ=sin((pi/2)*R_QQ); % autocorrelation coefficient of Q 
RO_IQ=sin((pi/2)*R_IQ); % autocorrelation coefficient of I and Q 
RO_QI=sin((pi/2)*R_QI); % autocorrelation coefficient of Q and I 
  
%complex correlation coefficient 
RO_hre=0.5*(RO_II+RO_QQ); 
RO_him=0.5*(RO_QI-RO_IQ);   
RO_h=complex(RO_hre,RO_him);% RO_h=0.5.*(RO_hre+i*(RO_him)) 
 
fc=angle(RO_h)/(2*pi*k*PRT);% The maximum value non-ambiguous of the correlation function R_xy is deltafdc_amb=1/k*PRT=PRF/k 