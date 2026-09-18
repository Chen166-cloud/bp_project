%% DOCUMENT TITLE
% INTRODUCTORY TEXT
%%
% 数据处理之CSA成像
% 2019.3.15
clear all;
close all;
%% 提取数据
% specify_parameters;
% extract_data;
% clear all;

%% 参数设置
% 雷达参数
C = 2.9979e8;                                                               % 光速
f0 = 1.4e9;                                                                 % 中心频率
lambda = C/f0;                                                      % 波长
Br=200e6;                                                            %带宽
Tp=10e-6 ;                                                             % 脉冲持续时间 
% 飞行平台
V = 17;                                                                   % 平台速度
% 距离向
Kr = - Br./Tp;                                                           % 调频率
Fr = 400e6;                                                              % 距离向采样率
dtau = 1/Fr;                                                                % 距离向采样时间间隔
% 方位向
Fa = 1000;                                                               % 方位向采样率
PRF = Fa;
PRT = 1/PRF
% fc = 6.3;                                                                 % 多普勒中心频率
deta = 1/Fa;                                                                % 方位向采样时间间隔
% 其他参数
t0 = 3e-6;                                                             % 获取数据时间
R0 = t0*C/2;                                                                % 最短斜距
H = 18
%% 获取原始数据（通道1）
% oldFolder = cd('.\scene01');
% load CDdata1.mat
% load ('2018-08-14_18-54-08（2）.mat')
% data_I(:,:) = echo(:,1:2:end).';
% data_Q(:,:) = echo(:,2:2:end).';
% raw_data=zeros(20000,10000);
% h=waitbar(0,'please wait');
% for m = 1:20000
%     raw_data(m,:) = 1i*data_I(m,:) + data_Q(m,:);
%     waitbar(m/20000,h);
% end
load('data_5000_10000.mat');
% load('data_2000.mat');
data=double(data_5000_10000);
[Na,Nr] = size(data);

%% 参考数据设置
fc = 0;                                                    % 多普勒中心频率
% fc = DopplerCenterEstimation_Eny(data,PRF)               %  Good:  -11.0603
Rref = (t0-Tp/2+Nr/2*dtau)*C/2;                                                   % 参考距离
Vref = V;                                                                   % 参考速度
fref = fc;                                                                  % 参考频率
Rs = Rref 
ff = linspace(-Fr/2,Fr/2,Nr) ;
Rnear = (t0-Tp/2)*C/2;                                                   % 近距
Rfar = (t0-Tp/2+Nr*dtau)*C/2;                                                   % 远距
DR = linspace(Rnear,Rfar,Nr);
%% 数据末尾补零
% Za = 800;                                                                   % 方位向补零数
% Zr = ceil(Tp/dtau);                                                         % 距离向补零数
% data = cat(2,data,zeros(Na,Zr));                                            % 距离向补零
% data = cat(1,zeros(Za,Nr+Zr),data);                                         % 方位向补零
% Na = Na+Za;
% Nr = Nr+Zr;

figure,imagesc(abs(data));axis image;set(gcf,'Color','w');
title('时域：补零后的原始信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 时间轴、频率轴设置
Ka = -(2*V.^2)/(lambda*R0)
tau = t0+(0:Nr-1)*dtau;                                                     % 距离向时间轴
ftau = Kr.*tau;                                                                    % 距离向频率轴
% ftau = ((0:Nr-1)-Nr/2)/Nr*Fr;                                               % 距离向频率轴
eta = ((0:Na-1)-Na/2)*deta;                                                 % 方位向时间轴
% feta = fc+((0:Na-1)-Na/2)/Na*Fa;                                            % 方位向频率轴
feta = -Ka.*eta;                                                                        % 方位向频率轴
% Kx = linspace(-1/2,1/2,Na)*PRF;
%% 中间变量设置
D_feta_V = sqrt(1-C^2*feta.^2/(4*V^2*f0^2));                                % D(feta,V) 徙动参数
% D_feta_V = sqrt(1-(C*Kx).^2/(4*V^2*f0^2));
D_feta_Vref = sqrt(1-C^2*feta.^2/(4*Vref^2*f0^2));                          % D(feta,Vref)
D_fref_V = sqrt(1-C^2*fref^2/(4*V^2*f0^2));                                 % D(fref,V)
D_fref_Vref = sqrt(1-C^2*fref^2/(4*Vref^2*f0^2));                           % D(fref,Vr555fKsrc = (2*V^2*f0^3*D_feta_V.^3)./(C*R0*feta.^2);                            % SRC滤波器的调频率        
Ksrc = (2*V^2*f0^3*D_feta_V.^3)./(C*R0*feta.^2);                            % SRC滤波器的调频率
Km = Kr./(1-Kr./Ksrc);                                                      % 改变后的距离向调频率
% Km = Kr./(1-(Kr*C*(ones(Na,1)*DR).*((Kx.^2).'*ones(1,Nr)))./(2*V^2*f0^3*(D_feta_V.^3).'*ones(1,Nr)));  % 新的调频率
clear Ksr
RCMbulk = (1./D_feta_Vref-1/D_fref_Vref)*Rref;                              % 一致RCM
alpha = D_fref_Vref./D_feta_Vref-1;                                         % 频偏参数a
alphb = D_feta_V./D_fref_V-1;                                                 % 频偏参数b
%% 对齐到方位向多普勒中心
S0 = data.*exp(-1i*2*pi*fc*(eta'*ones(1,Nr)));                              % 搬移至多普勒频率中心
figure,imagesc(abs(S0));axis image;set(gcf,'Color','w');
title('搬移至多普勒频率中心后信的号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
clear data
%% 惯导补偿
%%%%%%%%%%%% 惯导数据预处理
positionDelta = [-7;16.23;-0.01]*PRT*ones(1,Na);   %平台实际运动轨迹
position = cumsum(positionDelta,2);
position(3,:) = position(3,:) + H;
position(2,:) = position(2,:) - (min(position(2,:))+max(position(2,:)))/2; 
% 中心移到0位置
positionReal = position;
%真实值
RcenTar = sqrt((positionReal(2,:)-0).^2 + (positionReal(1,:)- sqrt(Rs^2 - H^2)).^2 + (positionReal(3,:)-0).^2);     % 测绘带中心参考目标的瞬时斜距
deltaR = RcenTar- Rref;
bcTemp = exp(1i*4*pi/C*deltaR'*(ff+f0));%矢量转置
S0 = fftshift(fft(S0,[],2),2);
S0 = S0.*bcTemp;
S0 = ifft(ifftshift(S0,2),[],2);
%% 变换到距离多普勒域，实现变标相乘
Srd = fftshift(fft(fftshift(S0,1),[],1),1);                                 % 原信号做方位向FFT
clear S0
% tt = 2/C*(R0/D_fref_V)-2*Rref./(C*D_feta_Vref);
% tt = 2/C*(R0/D_fref_V)-2*Rref./(C*D_fref_Vref);                     % P205 (7.26) (7.27)
tt = 2/C*(R0/D_fref_V+RCMbulk)-2*Rref./(C*D_feta_Vref);                     % P205 (7.26) (7.27)
Ssc = exp(1i*pi*Km.*alpha.*tt.^2);                                          % 变标方程 P207 (7.30)
% clear tt
S1 = Srd.*(Ssc'*ones(1,Nr));                                                % 变标相乘
figure,imagesc(abs(S1));axis image;set(gcf,'Color','w');
title('变标后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
clear Srd Ssc

%% 变换到二维频域，实现RC、SRC和一致RCMC
S2 = fftshift(fft(fftshift(S1,2),[],2),2);                                  % 信号变换到二维频域
clear S1
WindowR = ones(Na,1)*kaiser(Nr,2.5)';                                       % 距离窗
WindowA = kaiser(Na,2.5)*ones(1,Nr);                                        % 方位窗
S2 = S2.*WindowR.*WindowA;                                                  % 加窗
figure,imagesc(abs(S2));axis image;set(gcf,'Color','w');
title('加窗后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
clear WindowR WindowA

Hm = exp(1i*pi./((Km'*ones(1,Nr)).*(1+alphb'*ones(1,Nr))).*(ones(Na,1)*ftau).^2); % 合并的距离压缩和一致RCMC滤波器
S3 = S2.*Hm;
figure,imagesc(abs(S3));axis image;set(gcf,'Color','w');
title('RCMC滤波后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
clear alpha
Hrcm = exp(1i*4*pi/C*(RCMbulk'*ones(1,Nr)).*(ones(Na,1)*ftau));             % SRC滤波器
clear RCMbulk
S4 = S3.*Hrcm;                                                          % 相位相乘，实现滤波
figure,imagesc(abs(S4));axis image;set(gcf,'Color','w');
title('SRC滤波后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');
clear S2 Hm Hrcm

%% 变换到距离多普勒域，实现方位压缩和附加相位校正
S5 = ifftshift(ifft(ifftshift(S4,2),[],2),2);                               % 信号的距离向IFFT
clear S4

figure,imagesc(abs(S5));axis image;set(gcf,'Color','w');
title('RD域：RC、SRC和一致RCMC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

% Hac = exp(1i*4*pi*R0*f0*D_feta_V/C);                                       % 方位压缩滤波器
Hac = 4*pi*f0*(ones(Na,1)*DR).*(D_feta_V.'*ones(1,Nr))./C;
% Hpc = -4*pi/C^2*Km*(1-(D_feta_V./D_fref_V).'*ones(1,Nr))*((ones(Na,1)*DR)/(D_feta_V.'*ones(1,Nr)) - Rs./(D_feta_V.'*ones(1,Nr))).^2;
% S6 = S5.*exp(1i*Dphase1).*exp(1i*Dphase2);
Hpc = exp(-1i*(4*pi*Km/C^2).*(1-D_feta_Vref/D_fref_Vref).*(R0./D_feta_V-Rref./D_feta_V).^2); % 相位校正滤波器
% S6 = S5.*(Hac'*ones(1,Nr)).*(Hpc'*ones(1,Nr));                              % 相位相乘，实现滤波
S6 = S5.*exp(1i*Hac).*(Hpc'*ones(1,Nr));                              % 相位相乘，实现滤波

figure,imagesc(abs(S6));axis image;set(gcf,'Color','w');
title('方位压缩后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

clear S5 Hac Hpc Km

%% 变换到图像域
Stt1 = ifftshift(ifft(ifftshift(S6,1),[],1),1);                             % 信号的方位向IFFT
clear S6

%% 调整图像1
Img = flipud(abs(Stt1));                                                     % 上下翻转图像
% Ntau = round((Tp/dtau/2-1)/2);                                              % 计算距离向弃置区长度
Img = Img(1:Na,1:Nr);                                                         % 裁剪图像有效区域
Img = Img/max(max(Img));
Img = 20*log10(Img+eps);
Img(Img<-50) = -50;
Img = uint8((Img+50)/50*255); 
figure,imagesc(Img);axis image;set(gcf,'Color','w'); 
imwrite(Img,'SAR_CSA.bmp','bmp');
axis off

%% 显示接口
% set(gcf,'WindowButtonDownFcn',@callback);   



%% DPCA对消
% Stt=Stt1-Stt2
%% pga聚焦
sig_img=Img;
D1=-60;

% sig_img=HH(5391:5933,5380:5933).';
% load ship.mat;
% sig_img=sig_img2;
% 
% figure;imagesc(20*log10(abs(sig_img.')/max(max(abs(sig_img.')))),[D1,0])

[Na,Nr] = size(sig_img);
center_az_idx = ceil(Nr/2);
phase_final = zeros(1,Nr);
iteration =10;
win=1;
for ii = 1:iteration
%     
    [~,maximum_az_idx] = max(abs(sig_img),[],2);
    new_sig = zeros(Na,Nr);
    for i = 1:Na
        new_sig(i,:) = circshift(sig_img(i,:).',center_az_idx - maximum_az_idx(i));      % circularshift
    end
%     figure;imagesc(20*log10(abs(new_sig)/max(max(abs(new_sig)))));caxis([-50 0])    
    if 1
        qq = sum(abs(new_sig),1);
        amplitude = 20*log10(qq);
        % figure(3);plot(amplitude);%测试是否聚集到中心位置
        db_down = 10;       %设置大小
        window_cutoff = max(amplitude)-db_down;
        leftidx = find(amplitude(1:center_az_idx) - window_cutoff<0, 1, 'last' );
        rightidx = find(amplitude(center_az_idx+1:end) - window_cutoff<0, 1, 'first');
        rightidx = rightidx + center_az_idx;
        window_length = rightidx-leftidx;
        window = zeros(Na,Nr);
        window(:,leftidx:rightidx) = 1;
        new_img = new_sig.*window;
    else
        window = zeros(Na,Nr);
        window(:,center_az_idx-500:center_az_idx+500) = 1;
        new_img = new_sig.*window;
    end
%     figure;imagesc(20*log10(abs(new_img)/max(max(abs(new_img)))));caxis([-50 0])
    F_sig = fft(new_img,[],2);                                                                          % FFT to the frequency domain
            
    delta_phase = angle(sum(conj(F_sig(:,1:end-1)).*F_sig(:,2:end),1));                            % estimate the phase error
    phase_estim = [0 cumsum(delta_phase)];
%     figure(11);plot(unwrap(phase_estim));title('estimated phase error');xlim([0 Na])
    phase_final = phase_final+phase_estim;
   
    F_sig_img = fft(sig_img,[],2);
    phase_cor = ones(Na,1)*phase_estim; 
    F_sig_img = F_sig_img.*exp(-1j*phase_cor); 
    sig_img = ifft(F_sig_img,[],2);
    
%     figure;imagesc(20*log10(abs(sig_img)/max(max(abs(sig_img)))));caxis([-50 0])
    win=win+1;
end
% Ntau = round((Tr/dtau/2-1)/2);
img = 20*log10(abs(sig_img.')/max(max(abs(sig_img.'))));
% img = img(1:Na-Za,Ntau+1:Ntau+Nr-Zr);
% img = img(801:1664,420:3619);
% img = flipud(img);
figure;imagesc(img,[D1,0])
axis off