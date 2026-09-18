function fc=DopplerCenterEstimation_Eny(x,PRF) 
x=x';
[nrn0,nan0]=size(x); 
for n=1:nrn0 
    c=x(n,1:nan0-1).*conj(x(n,2:nan0));  % 相邻距离线信号样本的共轭乘积(自相关)，其相角为两样本间的相位差，dim(c)=Na-1
    fd_center_n(n,1)=angle(mean(c))/(2*pi)*PRF; % ACCC方法：对“差分复矢量” 平均，得平均相位增量，Fdc=Fa*Phi_accc/(2pi)
end 
fdc_center0= abs(median(fd_center_n));     

x_zlt=abs(fftshift(fft(x,[],2)));     % 方位向（行方向）傅立叶变换， 回波方位谱绝对值 
x_zlt=mean(x_zlt);                   % 方位谱绝对值在距离向 （列方向）平均
[mopt,position]=throw_unormal_value_r(x_zlt,1.7); % 距离向平均的结果再去噪平滑
nan=length(position); 
%
x_zlt=mopt;    
nan=length(x_zlt); 
E_all=0; 
for mm=2:nan 
    E_all=E_all+(x_zlt(mm)+x_zlt(mm-1))/2; 
end 
aify=2*(max(x_zlt)-min(x_zlt))/E_all;   % 最大值与最小值之差比E_all
[a,b]=min(x_zlt);   %% 方位谱距离向绝对值平均的最小值 及其index- b
fdc_center1=(mod(b+nan/2,nan)-nan/2)/nan*PRF; 
[c,d]=max(x_zlt); 
fdc_center2=(d-nan/2)/nan*PRF; 
fd_center=0; 
fdc_nan=round(fd_center/PRF*nan+nan/2);   % 理想多普勒中心频率的 index：Na/2
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   1 
E=0; 
for mm=min(b,fdc_nan)+1:max(b,fdc_nan)  % index b 和 fdc_nan之间：求E
    E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
end 
if b>fdc_nan   % indx b 大于 fdc_nan
    Er=E; 
    El=E_all-E; 
else 
    El=E; 
    Er=E_all-E; 
end 
deltaE=(Er-El)/E_all; 
modi=deltaE/aify;            % 
fd_center=fd_center+modi;    % 更新多普勒中心估计
% fdc_nan=mod(round(fd_center*PRF/(nan/2)+nan/2),nan); 
fdc_nan=round(fd_center/PRF*nan+nan/2); 
fdc_nan1=mod(fdc_nan,nan);   % 更新多普勒中心估计 fdc_nan1
if fdc_nan1==0 
    fdc_nan1=nan; 
end 
 
%第 2 次迭代求 fdc_nan
E=0; 
for mm=min(b,fdc_nan1)+1:max(b,fdc_nan1) 
    E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
end 
if b>fdc_nan1 
    Er=E; 
    El=E_all-E; 
else 
    El=E; 
    Er=E_all-E; 
end 
deltaE=(Er-El)/E_all; 
modi=deltaE/aify; 
fd_center=fd_center+modi; 
% fdc_nan=mod(round(fd_center*PRF/(nan/2)+nan/2),nan); 
fdc_nan=round(fd_center/PRF*nan+nan/2); 
fdc_nan2=mod(fdc_nan,nan); 
if fdc_nan2==0 
    fdc_nan2=nan; 
end 

 
if  fdc_nan2~=fdc_nan1 
%如果两次fdc_nan不一致，执行第3次迭代：
 E=0; 
 for mm=min(b,fdc_nan2)+1:max(b,fdc_nan2) 
    E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
 end 
 if b>fdc_nan2 
    Er=E; 
    El=E_all-E; 
 else 
    El=E; 
    Er=E_all-E; 
 end 
 deltaE=(Er-El)/E_all; 
 modi=deltaE/aify; 
 fd_center=fd_center+modi; 
% fdc_nan=mod(round(fd_center*PRF/(nan/2)+nan/2),nan); 
 fdc_nan=round(fd_center/PRF*nan+nan/2); 
 fdc_nan3=mod(fdc_nan,nan); 
 if fdc_nan3==0 
    fdc_nan3=nan; 
 end 

 if fdc_nan3~=fdc_nan2 
%如果第2、3次fdc_nan不一致，执行第4次迭代 
 
  E=0; 
  for mm=min(b,fdc_nan3)+1:max(b,fdc_nan3) 
    E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
  end 
  if b>fdc_nan3 
    Er=E; 
    El=E_all-E; 
  else 
    El=E; 
    Er=E_all-E; 
  end 
  deltaE=(Er-El)/E_all; 
  modi=deltaE/aify; 
  fd_center=fd_center+modi 
 % fdc_nan=mod(round(fd_center*PRF/(nan/2)+nan/2),nan); 
  fdc_nan=round(fd_center/PRF*nan+nan/2); 
  fdc_nan4=mod(fdc_nan,nan); 
  if fdc_nan4==0 
    fdc_nan4=nan; 
  end 
 
  if fdc_nan4~=fdc_nan3 
%如果第3、4次fdc_nan不一致，执行第5次迭代 
   E=0; 
   for mm=min(b,fdc_nan4)+1:max(b,fdc_nan4) 
    E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
   end 
   if b>fdc_nan4 
    Er=E; 
    El=E_all-E; 
   else 
    El=E; 
    Er=E_all-E; 
   end 
   deltaE=(Er-El)/E_all; 
   modi=deltaE/aify; 
   fd_center=fd_center+modi 
% fdc_nan=mod(round(fd_center*PRF/(nan/2)+nan/2),nan); 
   fdc_nan=round(fd_center/PRF*nan+nan/2); 
   fdc_nan5=mod(fdc_nan,nan); 
   if fdc_nan5==0 
    fdc_nan5=nan; 
   end 
 
   if fdc_nan5~=fdc_nan4 
%同上第 6 次迭代
 
    E=0; 
    for mm=min(b,fdc_nan5)+1:max(b,fdc_nan5) 
     E=E+(x_zlt(mm)+x_zlt(mm-1))/2; 
    end 
    if b>fdc_nan5 
     Er=E; 
     El=E_all-E; 
    else 
     El=E; 
     Er=E_all-E; 
    end 
    deltaE=(Er-El)/E_all; 
    modi=deltaE/aify; 
    fd_center=fd_center+modi 
   end
  end
 end
end
fc = fdc_center0 + fd_center
end



function [x,n]=throw_unormal_value(x,std_threshold) 
 %
 flag=0;n=0; 
 while (flag==0) 
     n=n+1; 
     if n>10 
         break; 
     end 
     a=find(x<(mean(x)-std_threshold*std(x))); 
     for k=1:length(a) 
      if a(k)==1 
          x(a(k))=x(a(k)+1); 
      elseif a(k)==length(x) 
          x(a(k))=x(a(k)-1); 
      else 
          x(a(k))=(x(a(k)-1)+x(a(k)+1))/2; 
      end 
     end 
   
     b=find(x>(mean(x)+std_threshold*std(x))); 
     for k=1:length(b) 
      if b(k)==1 
          x(b(k))=x(b(k)+1); 
      elseif b(k)==length(x) 
          x(b(k))=x(b(k)-1); 
      else 
          x(b(k))=(x(b(k)-1)+x(b(k)+1))/2; 
      end 
     end 
 end     
end 