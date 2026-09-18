%%
% RADARSAT-1数据处理CSA成像
% 2019.3.22   Modified by Peng Liu
clear all;close all;

%% 之前必须运行以下两个程序，提取数据块
% specify_parameters_step_1;
% extract_data_step_2;
% clear all;

%% 参数设置
% 雷达参数，由 P.393 表A.1给出
C = 2.9979e8;                                                               % 光速
f0 = 1.4e9;                                                                 % 中心频率
lambda = C/f0;                                                              % 波长
% 飞行平台
V = 60;                                                                   % 平台速度
% 距离向
Br=200e6;                                                                   %带宽
Tr = 10e-6;                                                                % 脉冲持续时间
Kr = - Br./Tr;                                                              % 距离向调频率；
Fr = 400e6;                                                              % 距离向r的采样率
dtau = 1/Fr;                                                                % 距离向r采样时间间隔
% 方位向
Fa = 1000.0;                                                               % 方位向a的采样率
deta = 1/Fa;                                                                % 距离向采样时间间隔
% 其他参数
t0 = 3e-6;                                                             % 数据窗开始时间
R0 = t0*C/2;                                                                % 最短斜距Slant range of first radar sample (m)
Ka = (2*V.^2)/(lambda*R0);

%% 获取原始数据（通道1）
load('data_5000_10000.mat');
data=double(data_5000_10000);
[Na,Nr] = size(data);                                                      % 原始数据维度

%% 多普勒中心估计
fc = 0;                                                              % 光盘给定设置
PRF = Fa;
DeltaR = C/(2*Fr);
chirp_rg_BW = Kr*Tr;
La=2*DeltaR;
beam_width_az = lambda/La*180/pi;
Za = 0;
ta=(Na+Za-1)*deta;
%fc = DopplerCenterEstimation_WDA(data,V,f0,PRF,Fr,Kr,Tr,lambda,R0,chirp_rg_BW)
%fc = DopplerCenterEstimation_Eny(data,PRF)               %  Good:  -7050
%fc = DopplerCenterEstimation(data,V,f0,PRF,Fr,Kr,Tr,lambda,R0)
%fc = Madsen(data,1,PRF);        % ACCC,平均互相关系数法，基于相位的多普勒中心估计，仅用于估计小数部分，300
%fc=DoppCentEsti_SLC(data,V,f0,PRF,Fr,beam_width_az,t0,chirp_rg_BW,Tr,Na,Nr)
%fc=DoppCentEsti_SLC_Liu(data,V,f0,PRF,Fr,Kr,Tr,lambda,R0,ta)
%fc=DopplerCenterEstimation_Ape(data,V,PRF,lambda,R0,DeltaR)        %  341
% fc= DopplerCenterEstimation_lowpa(data,PRF);         % 108
%fc=DopplerCenterEstimation_MID_jie(data,V,PRF,lambda,R0,DeltaR)
%% 参考数据设置
Rref = (t0+Nr/2*dtau)*C/2;                                                  % 参考距离
Vref = V;                                                                   % 参考速度
fref = fc;                                                                  % 参考频率

%% 数据末尾补零
Zr = ceil(Tr/dtau);                                                         % 距离向补零数
data = cat(2,data,zeros(Na,Zr));                                            % 距离向补零
data = cat(1,zeros(Za,Nr+Zr),data);                                         % 方位向补零
Na = Na+Za;
Nr = Nr+Zr;

figure,imagesc(abs(data));axis image;set(gcf,'Color','w');
title('时域：补零后的原始信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 时间轴、频率轴设置
tau = t0+(0:Nr-1)*dtau;                                                     % 距离向时间轴
ftau = ((0:Nr-1)-Nr/2)/Nr*Fr;                                               % 距离向频率轴
eta = ((0:Na-1)-Na/2)*deta;                                                 % 方位向时间轴
feta = fc+((0:Na-1)-Na/2)/Na*Fa;                                            % 方位向频率轴

%% 中间变量设置
D_feta_V = sqrt(1-C^2*feta.^2/(4*V^2*f0^2));                                % D(feta,V)
D_feta_Vref = sqrt(1-C^2*feta.^2/(4*Vref^2*f0^2));                          % D(feta,Vref)
D_fref_V = sqrt(1-C^2*fref^2/(4*V^2*f0^2));                                 % D(fref,V)
D_fref_Vref = sqrt(1-C^2*fref^2/(4*Vref^2*f0^2));                           % D(fref,Vr555fKsrc = (2*V^2*f0^3*D_feta_V.^3)./(C*R0*feta.^2);                            % SRC滤波器的调频率        
Ksrc = (2*V^2*f0^3*D_feta_V.^3)./(C*R0*feta.^2);                            % SRC滤波器的调频率
Km = Kr./(1-Kr./Ksrc);                                                      % 改变后的距离向调频率
clear Ksrc
RCMbulk = (1./D_feta_Vref-1/D_fref_Vref)*Rref;                              % 一致RCM
alpha = D_fref_Vref./D_feta_Vref-1;                                         % 频偏参数

%% 对齐到方位向多普勒中心
S0 = data.*exp(-1i*2*pi*fc*(eta'*ones(1,Nr)));                              % 搬移至多普勒频率中心
clear data

%% 变换到距离多普勒域，实现变标相乘
Srd = fftshift(fft(fftshift(S0,1),[],1),1);                                 % 原信号做方位向FFT
clear S0
tt = 2/C*(R0/D_fref_V+RCMbulk)-2*Rref./(C*D_feta_Vref);                     % P205 (7.26) (7.27)
Ssc = exp(1i*pi*Km.*alpha.*tt.^2);                                          % 变标方程 P207 (7.30)
clear tt
S1 = Srd.*(Ssc'*ones(1,Nr));                                                % 变标相乘
clear Srd Ssc

%% 变换到二维频域，实现RC、SRC和一致RCMC
S2 = fftshift(fft(fftshift(S1,2),[],2),2);                                  % 信号变换到二维频域
clear S1
WindowR = ones(Na,1)*kaiser(Nr,2.5)';                                       % 距离窗
WindowA = kaiser(Na,2.5)*ones(1,Nr);                                        % 方位窗
S2 = S2.*WindowR.*WindowA;                                                  % 加窗
clear WindowR WindowA
Hm = exp(1i*pi./((Km'*ones(1,Nr)).*(1+alpha'*ones(1,Nr))).*(ones(Na,1)*ftau).^2); % 合并的距离压缩和一致RCMC滤波器
clear alpha
Hrcm = exp(1i*4*pi/C*(RCMbulk'*ones(1,Nr)).*(ones(Na,1)*ftau));             % SRC滤波器
clear RCMbulk
S3 = S2.*Hm.*Hrcm;                                                          % 相位相乘，实现滤波
clear S2 Hm Hrcm

%% 变换到距离多普勒域，实现方位压缩和附加相位校正
S4 = ifftshift(ifft(ifftshift(S3,2),[],2),2);                               % 信号的距离向IFFT
clear S3

figure,imagesc(abs(S4));axis image;set(gcf,'Color','w');
title('RD域：RC、SRC和一致RCMC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

Hac = exp(-1i*4*pi*R0*f0*D_feta_V/C);                                       % 方位压缩滤波器
Hpc = exp(-1i*(4*pi*Km/C^2).*(1-D_feta_Vref/D_fref_Vref).*(R0./D_feta_V-Rref./D_feta_V).^2); % 相位校正滤波器
S5 = S4.*(Hac'*ones(1,Nr)).*(Hpc'*ones(1,Nr));                              % 相位相乘，实现滤波
clear S4 Hac Hpc Km

%% 变换到图像域
Stt1 = ifftshift(ifft(ifftshift(S5,1),[],1),1);                             % 信号的方位向IFFT
clear S5

%% 调整图像1
Img = flipud(abs(Stt1));                                                     % 上下翻转图像
Ntau = round((Tr/dtau/2-1)/2);                                              % 计算距离向弃置区长度
Img = Img(1:Na-Za,Ntau+1:Ntau+Nr-Zr);                                       % 裁剪图像有效区域
Img = Img/max(max(Img));
Img = 20*log10(Img+eps);
Img(Img<-50) = -50;
Img = uint8((Img+50)/50*255); 
figure,imagesc(Img);axis image;set(gcf,'Color','w'); 
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
imwrite(Img,'SAR_CSA.bmp','bmp');
axis off
%% 显示接口
set(gcf,'WindowButtonDownFcn',@callback);

% %% DPCA对消
% % Stt=Stt1-Stt2
% %% pga聚焦
% sig_img=Img;
% D1=-60;
% 
% % sig_img=HH(5391:5933,5380:5933).';
% % load ship.mat;
% % sig_img=sig_img2;
% 
% % figure;imagesc(20*log10(abs(sig_img.')/max(max(abs(sig_img.')))),[D1,0])
% 
% [Nr,Na] = size(sig_img);
% center_az_idx = ceil(Na/2);
% phase_final = zeros(1,Na);
% iteration =10;
% win=1;
% for ii = 1:iteration
%     
%     [~,maximum_az_idx] = max(abs(sig_img),[],2);
%     new_sig = zeros(Nr,Na);
%     for i = 1:Nr
%         new_sig(i,:) = circshift(sig_img(i,:).',center_az_idx - maximum_az_idx(i));      % circularshift
%     end
% %     figure;imagesc(20*log10(abs(new_sig)/max(max(abs(new_sig)))));caxis([-50 0])    
%     if 1
%         qq = sum(abs(new_sig),1);
%         amplitude = 20*log10(qq);
%         % figure(3);plot(amplitude);%测试是否聚集到中心位置
%         db_down = 10;       %设置大小
%         window_cutoff = max(amplitude)-db_down;
%         leftidx = find(amplitude(1:center_az_idx) - window_cutoff<0, 1, 'last' );
%         rightidx = find(amplitude(center_az_idx+1:end) - window_cutoff<0, 1, 'first');
%         rightidx = rightidx + center_az_idx;
%         window_length = rightidx-leftidx
%         window = zeros(Nr,Na);
%         window(:,leftidx:rightidx) = 1;
%         new_img = new_sig.*window;
%     else
%         window = zeros(Nr,Na);
%         window(:,center_az_idx-500:center_az_idx+500) = 1;
%         new_img = new_sig.*window;
%     end
% %     figure;imagesc(20*log10(abs(new_img)/max(max(abs(new_img)))));caxis([-50 0])
%     F_sig = fft(new_img,[],2);                                                                          % FFT to the frequency domain
%             
%     delta_phase = angle(sum(conj(F_sig(:,1:end-1)).*F_sig(:,2:end),1));                            % estimate the phase error
%     phase_estim = [0 cumsum(delta_phase)];
% %     figure(11);plot(unwrap(phase_estim));title('estimated phase error');xlim([0 Na])
%     phase_final = phase_final+phase_estim;
%    
%     F_sig_img = fft(sig_img,[],2);
%     phase_cor = ones(Nr,1)*phase_estim; 
%     F_sig_img = F_sig_img.*exp(-1j*phase_cor); 
%     sig_img = ifft(F_sig_img,[],2);
%     
% %     figure;imagesc(20*log10(abs(sig_img)/max(max(abs(sig_img)))));caxis([-50 0])
%     win=win+1;
% end
% % Ntau = round((Tr/dtau/2-1)/2);
% img = 20*log10(abs(sig_img.')/max(max(abs(sig_img.'))))
% % img = img(1:Na-Za,Ntau+1:Ntau+Nr-Zr);
% % img = img(801:1664,420:3619);
% % img = flipud(img);
% figure;imagesc(img,[D1,0])
% axis off