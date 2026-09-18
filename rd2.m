%% DOCUMENT TITLE
% INTRODUCTORY TEXT
%%
% 数据处理之RDA成像
% 2019.3.15
% 清除所有变量并关闭所有图形窗口
clear all;
close all;
delete(gcp('nocreate'));
%% 参数设置
% 雷达参数
C = 2.9979e8;                           % 光速 (m/s)
f0 = 1.4e9;                             % 中心频率 (Hz)
lambda = C/f0;                          % 波长 (m)
Br=200e6;                               % 带宽 (Hz)
Tp=1.0e-6 ;                             % 脉冲持续时间 (s)
% 飞行平台
V = 17;                                 % 平台速度 (m/s)
fc_max=2*V/lambda;                      % 多普勒中心最大值 (Hz)，由方位向带宽决定
% 距离向
Kr =  -Br./Tp;                          % 距离向调频率 (Hz/s)
Fr = 400e6;                             % 距离向采样率 (Hz)
dtau = 1/Fr;                            % 距离向采样时间间隔 (s)
% 方位向
Fa = 1000;                              % 方位向采样率 (Hz)
PRF = Fa;                               % 脉冲重复频率 (Hz)，等于方位向采样率
deta = 1/Fa;                            % 方位向采样时间间隔 (s)
% 其他参数
t0 = 3e-6;                              % 获取数据时间，采样延迟 (s)
R0 = t0*C/2;                            % 最短斜距 (m)

%% 获取原始数据（通道1）
load('data_5000_10000.mat');            % 加载包含雷达回波数据的.mat文件
data=double(data_5000_10000);           % 将数据转换为双精度浮点数
[Na,Nr] = size(data);                   % 获取数据维度：Na (方位向采样点数), Nr (距离向采样点数)
%% SAR图像设定：图像动态范围及放大图像的范围[X1 X2,Y1 Y2]
x = 1:1:Nr;                             % 距离向坐标轴索引
y =1:1:Na;                              % 方位向坐标轴索引
YC=Na/2;                                % 图像中心行索引
YW=800;                                 % 放大窗口的行数一半
Y1=YC-YW;                               % 放大窗口的起始行索引
Y2=YC+YW;                               % 放大窗口的结束行索引
X1=1;                                   % 放大窗口的起始列索引
X2=Nr;                                  % 放大窗口的结束列索引
dyn_range = 60;                         % 显示的动态范围 (dB)

% 原始数据预处理
display 'RAW data conditioning'         % 在命令行窗口显示处理阶段信息
data = data - mean(mean(data));         % 减去原始数据的均值（去直流分量）
data = data.* (hamming(Na)*hamming(Nr)'); % 应用二维汉明窗进行加窗去噪

%% 多普勒中心估计
fc = 0;                                 % 多普勒中心采用给定值 (Hz)
Tr=Tp;                                  % 有效脉冲持续时间 (s)
chirp_rg_BW = abs(Kr)*Tr;               % 距离向 chirp 信号带宽 (Hz)
DeltaR = C/(2*chirp_rg_BW);             % 距离向分辨率 (m)
La=2*DeltaR;                            % 方位向分辨率近似值 (m)
beam_width_az = 30;                     % 方位向波束宽度 (度)
Za = 0;                                 % 方位向补零数
ta=(Na+Za-1)*deta;                      % 总的方位时间 (s)
%% 参考数据设置
Rref = (t0-Tp/2+Nr/2*dtau)*C/2;         % 参考距离 (m)
Vref = V;                               % 参考速度 (m/s)
fref = fc;                              % 参考频率 (Hz)

%% 数据末尾补零
Zr = ceil(Tr/dtau);                     % 计算距离向需要补零的点数
data = cat(2,data,zeros(Na,Zr));        % 在距离向末尾补零
data = cat(1,zeros(Za,Nr+Zr),data);     % 在方位向开头补零
Na = Na+Za;                             % 更新方位向总点数
Nr = Nr+Zr;                             % 更新距离向总点数

% 显示补零后的原始信号幅度图
figure,imagesc(abs(data));axis image;set(gcf,'Color','w');
title('时域：补零后的原始信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 时间轴、频率轴设置
tau = t0+(0:Nr-1)*dtau;                 % 距离向时间轴 (s)
ftau = ((0:Nr-1)-Nr/2)/Nr*Fr;           % 距离向频率轴 (Hz)
eta = ((0:Na-1)-Na/2)*deta;             % 方位向时间轴 (s)
feta = fc+((0:Na-1)-Na/2)/Na*Fa;        % 方位向频率轴 (Hz)，包含多普勒中心偏移

%% 对齐到方位向多普勒中心
S0 = data.*exp(-1i*2*pi*fc*(eta'*ones(1,Nr))); % 将信号搬移到多普勒频率中心
clear data                                % 清除原始数据变量以节省内存

%% 距离压缩 (RC)
Tref = t0+Nr/2*dtau;                    % 距离向参考时间 (s)
% 生成距离向匹配滤波器（参考信号）
ht_rc = (abs((ones(Na,1)*tau)-Tref)<Tr/2).*exp(1i*pi*Kr*((ones(Na,1)*tau)-Tref).^2);
Hf_rc = conj(fftshift(fft(fftshift(ht_rc,2),[],2),2)); % 对匹配滤波器做距离向FFT并取共轭
clear ht_rc                               % 清除时域匹配滤波器
S0_rc = fftshift(fft(fftshift(S0,2),[],2),2); % 对信号做距离向FFT
clear S0                                  % 清除原始信号
S1 = ifftshift(ifft(ifftshift(S0_rc.*Hf_rc,2),[],2),2); % 匹配滤波后做距离向IFFT
clear Hf_rc S0_rc                         % 清除中间变量

% 显示距离压缩后的信号幅度图
figure,imagesc(abs(S1));axis image;set(gcf,'Color','w');
title('时域：距离压缩后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 距离徙动校正 (SRC)
Sff = fftshift(fft2(fftshift(S1)));     % 对信号做二维FFT变换到频域
WindowR = ones(Na,1)*kaiser(Nr,2.5)';   % 生成距离向Kaiser窗
WindowA = kaiser(Na,2.5)*ones(1,Nr);    % 生成方位向Kaiser窗
Sff = Sff.*WindowR.*WindowA;            % 在频域应用二维窗
clear WindowR WindowA                     % 清除窗变量
D = sqrt(1-lambda^2*feta.^2/(4*V^2));   % 计算距离徙动因子
% 计算SRC滤波器的调频率
Ksrc = (2*V^2*f0^3*(D'*ones(1,Nr)).^3)./(C*R0*(feta'*ones(1,Nr)).^2);
Hsrc = exp(-1i*pi*((ones(Na,1)*ftau).^2)./Ksrc); % 生成SRC滤波器
clear Ksrc                                % 清除调频率变量
S1 = Sff.*Hsrc;                           % 应用SRC滤波器
clear Sff Hsrc                            % 清除中间变量
Srd = ifftshift(ifft(ifftshift(S1,2),[],2),2); % SRC校正后做距离向IFFT，变换到距离多普勒域
clear S1                                  % 清除中间变量

% 显示SRC后的RD域信号幅度图
ABSimg=20*log10(abs(Srd)./ max(max(abs(Srd)))); % 转换为dB并归一化
imagesc(ABSimg,[-dyn_range 0]);         % 显示图像，动态范围为[-dyn_range, 0] dB
colormap('jet');                        % 设置颜色图
title('RD域：SRC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

% 显示放大后的SRC图像
figure;
flagx = (x>X1)&(x<X2);                  % 距离向放大区域掩码
flagy = (y>Y1)&(y<Y2);                  % 方位向放大区域掩码
NEWimg = ABSimg(flagy,flagx);           % 提取放大区域图像数据
imagesc(x(flagx),y(flagy),NEWimg,[-dyn_range 0]); % 显示放大图像
title('放大SRC后图像');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 距离徙动校正 (RCMC)
% 启动并行池
parpool('local', 4);                    % 根据 CPU 核心数量设置并行池大小

deltaR = dtau*C/2;                      % 距离向采样长度间隔 (m)
deltaRCM = Rref*(1./D-1);               % 距离徙动量 (m)
deltaRCM = deltaRCM/deltaR;             % 距离徙动量的量化（以采样点为单位）
IntNum = ceil(deltaRCM);                % 对距离徙动量向上取整 (整数部分)
DecNum = IntNum-deltaRCM;               % 距离徙动量的小数部分
Srcmc = zeros(Na,Nr);                   % 初始化RCMC后的数据矩阵

% 使用 parfor 并行处理外层循环
tic;                                    % 开始计时
parfor ii = 1:Na                        % 对每个方位向样本并行处理
    % 生成 sinc 插值核
    SincVal = sinc(DecNum(ii)-4:1:DecNum(ii)+3)'; % 生成插值核 (8点sinc核)

    % 内层循环处理距离向
    for jj = 1:Nr                       % 对每个距离向样本处理
        kk = jj + IntNum(ii);           % 计算原矩阵中徙动后的位置
        if (5 <= kk && kk <= Nr-3)      % 防止下标溢出 (确保插值核不越界)
            Srcmc(ii,jj) = Srd(ii,kk-4:kk+3) * SincVal; % 使用sinc核进行插值
        end
    end
end
toc;                                    % 结束计时并显示耗时
% 注意：在 parfor 中无法使用 waitbar 进度条，因此已移除

clear deltaRCM IntNum DecNum Srd        % 清除中间变量

% 显示RCMC后的RD域信号幅度图
figure,imagesc(abs(Srcmc));axis image;set(gcf,'Color','w');
title('RD域：RCMC后的信号幅度');
xlabel('距离向（采样点）');ylabel('方位向（采样点）');

%% 方位压缩 (AC)
Haz = exp(1i*4*pi*R0*(D'*ones(1,Nr))*f0/C); % 生成方位向匹配滤波器
clear D                                 % 清除距离徙动因子
S2 = Srcmc.*Haz;                        % 应用方位向匹配滤波器
clear Haz Srcmc                           % 清除中间变量
Stt = ifftshift(ifft(ifftshift(S2,1),[],1),1); % 做方位向IFFT变换
clear S2                                % 清除中间变量

%% 调整图像
Img = flipud(abs(Stt));                 % 上下翻转图像以便显示
Ntau = round((Tr/dtau/2-1)/2);          % 计算距离向弃置区长度 (点数)
% 裁剪图像有效区域，去除补零和边缘效应部分
Img = Img(1:Na-Za,Ntau+1:Ntau+Nr-Zr);
Img = Img/max(max(Img));                % 归一化到 [0, 1]
Img = 20*log10(Img+eps);                % 转换为dB (eps防止log(0))
Img(Img<-50) = -50;                     % 限制最小动态范围
Img = uint8((Img+50)/50*255);           % 转换为8位无符号整数图像 (0-255)

% 显示最终SAR图像
figure,imagesc(Img);axis image;set(gcf,'Color','w');

% 保存SAR图像为BMP文件
imwrite(Img,'SAR_RDA_2.bmp','bmp');



