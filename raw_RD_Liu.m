%% DOCUMENT TITLE
% INTRODUCTORY TEXT
%%
% 数据处理之RDA成像
% 2019.3.15
clear all;
close all;
%% 参数设置
% 雷达参数
C = 2.9979e8;                                                              % 光速
f0 = 1.4e9;                                                                % 中心频率
lambda = C/f0;                                                             % 波长
Br=200e6;                                                                  %带宽
Tp=1.0e-6 ;                                                                  % 脉冲持续时间 
% 飞行平台
V = 17;                                                               % 惯导读出：17.6752 平台速度 跑车 30 km/hr = 9.3 m/s
fc_max=2*V/lambda;                                                   % 多普勒中心最大值, 由方位向带宽，V至少 54
% 距离向
Kr =  -Br./Tp;                                                           % 调频率
Fr = 400e6;                                                              % 距离向采样率
dtau = 1/Fr;                                                                % 距离向采样时间间隔
% 方位向
Fa = 1000;                                                                 % 方位向采样率
PRF = Fa;
deta = 1/Fa;                                                                % 方位向采样时间间隔
% 其他参数
t0 = 3e-6;                                                                 % 获取数据时间，采样延迟3e-6
R0 = t0*C/2;                                                                % 最短斜距

%% 获取原始数据（通道1）
load('data_5000_10000.mat');
data=double(data_5000_10000);
[Na,Nr] = size(data);
%% SAR图像设定：图像动态范围及放大图像的范围[X1 X2,Y1 Y2]
x = 1:1:Nr;  y =1:1:Na;    % 坐标轴
YC=Na/2; YW=800; Y1=YC-YW; Y2=YC+YW;  X1=1; X2=Nr;
dyn_range = 60;              % dB of dynamic range to display
%
  display 'RAW data conditioning'
  data = data - mean(mean(data));           % Subtract raw mean value
  data = data.* (hamming(Na)*hamming(Nr)');      % 加窗去噪：二维窗

%% 多普勒中心估计
fc = 0;                                                              % 多普勒中心采用给定值
Tr=Tp;
chirp_rg_BW = abs(Kr)*Tr;                                    %带宽 200M
DeltaR = C/(2*chirp_rg_BW);                                           %距离向分辨率
La=2*DeltaR;
beam_width_az = 30;
Za = 0;
ta=(Na+Za-1)*deta;
%% 参考数据设置
Rref = (t0-Tp/2+Nr/2*dtau)*C/2;                                                  % 参考距离
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

%% 对齐到方位向多普勒中心
S0 = data.*exp(-1i*2*pi*fc*(eta'*ones(1,Nr)));                              % 搬移至多普勒频率中心
clear data

%% RC
Tref = t0+Nr/2*dtau;                                                        % 距离向参考时间
ht_rc = (abs((ones(Na,1)*tau)-Tref)<Tr/2).*exp(1i*pi*Kr*((ones(Na,1)*tau)-Tref).^2); % 方式2的匹配滤波器
Hf_rc = conj(fftshift(fft(fftshift(ht_rc,2),[],2),2));                      % 滤波器的距离向FFT变换
clear ht_rc
S0_rc = fftshift(fft(fftshift(S0,2),[],2),2);                               % 信号的距离向FFT变换
clear S0
S1 = ifftshift(ifft(ifftshift(S0_rc.*Hf_rc,2),[],2),2);                     % 匹配滤波后距离向IFFT变换
clear Hf_rc S0_rc

figure,imagesc(abs(S1));axis image;set(gcf,'Color','w');
title('时域：距离压缩后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% SRC
Sff = fftshift(fft2(fftshift(S1)));                                         % 信号的二维FFT变换
WindowR = ones(Na,1)*kaiser(Nr,2.5)';                                       % 距离窗
WindowA = kaiser(Na,2.5)*ones(1,Nr);                                        % 方位窗
Sff = Sff.*WindowR.*WindowA;                                                % 加窗
clear WindowR WindowA
D = sqrt(1-lambda^2*feta.^2/(4*V^2));                                       % 距离徙动因子
Ksrc = (2*V^2*f0^3*(D'*ones(1,Nr)).^3)./(C*R0*(feta'*ones(1,Nr)).^2);       % SRC滤波器的调频率
Hsrc = exp(-1i*pi*((ones(Na,1)*ftau).^2)./Ksrc);                            % SRC滤波器
clear Ksrc
S1 = Sff.*Hsrc;                                                             % 滤波
clear Sff Hsrc
Srd = ifftshift(ifft(ifftshift(S1,2),[],2),2);                              % SRC校正后距离向IFFT变换，变换到距离多普勒域
clear S1


ABSimg=20*log10(abs(Srd)./ max(max(abs(Srd))));
imagesc(ABSimg,[-dyn_range 0]);
colormap('jet');
title('RD域：SRC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

figure; 
flagx = (x>X1)&(x<X2);
flagy = (y>Y1)&(y<Y2);
NEWimg = ABSimg(flagy,flagx);
imagesc(x(flagx),y(flagy),NEWimg,[-dyn_range 0]);
title('放大SRC后图像');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% RCMC
deltaR = dtau*C/2;                                                          % 距离向采样长度间隔
deltaRCM = Rref*(1./D-1);                                                   % 距离徙动量
deltaRCM = deltaRCM/deltaR;                                                 % 距离徙动量的量化
IntNum = ceil(deltaRCM);                                                    % 对距离徙动量向上取整;
DecNum = IntNum-deltaRCM;                                                   % 距离徙动量的小数部分量化为（1/16）的倍数
Srcmc = zeros(Na,Nr);
h = waitbar(0,'Sinc插值');
for ii = 1:Na
    SincVal = sinc(DecNum(ii)-4:1:DecNum(ii)+3)';                           % 生成插值核
    for jj = 1:Nr
        kk = jj+IntNum(ii);                                                 % 原矩阵中距离徙动的位置
        if(5<=kk && kk<=Nr-3)                                               % 防止下标溢出
            Srcmc(ii,jj) = Srd(ii,kk-4:kk+3)*SincVal;                       % 插值
        end
    end
    waitbar(ii/Na);
end
close(h);
clear deltaRCM IntNum DecNum Srd

figure,imagesc(abs(Srcmc));axis image;set(gcf,'Color','w');
title('RD域：RCMC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% AC
Haz = exp(1i*4*pi*R0*(D'*ones(1,Nr))*f0/C);                                 % 方位向匹配滤波器
clear D
S2 = Srcmc.*Haz;                                                            % 匹配滤波
clear Haz Srcmc
Stt = ifftshift(ifft(ifftshift(S2,1),[],1),1);                              % 方位向IFFT变换
clear S2

%% 调整图像
Img = flipud(abs(Stt));                                                     % 上下翻转图像
Ntau = round((Tr/dtau/2-1)/2);                                              % 计算距离向弃置区长度
Img = Img(1:Na-Za,Ntau+1:Ntau+Nr-Zr);                                       % 裁剪图像有效区域
Img = Img/max(max(Img));
Img = 20*log10(Img+eps);
Img(Img<-50) = -50;
Img = uint8((Img+50)/50*255);

figure,imagesc(Img);axis image;set(gcf,'Color','w');

imwrite(Img,'SAR_RDA_2.bmp','bmp');

