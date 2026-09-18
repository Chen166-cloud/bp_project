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

%% BP算法核心部分

% 设置图像网格参数
% 图像分辨率设置
range_resolution = DeltaR;
azimuth_resolution = lambda/(2*V); % 理论分辨率

% 定义图像空间坐标
image_range = 0:range_resolution:(Nr*dtau*C/2 - range_resolution);
image_azimuth = -V*Na*deta/2:azimuth_resolution:V*Na*deta/2 - azimuth_resolution;

% 图像尺寸
Nr_img = length(image_range);
Na_img = length(image_azimuth);

% 初始化图像
BP_img = zeros(Na_img, Nr_img);

% 计算每一点的回波贡献
% 计算雷达位置
Radar_pos = zeros(Na, 2);
for i = 1:Na
    Radar_pos(i, :) = [V * eta(i), 0];  % 假设沿X方向匀速运动
end

% 计算每个像素到所有雷达位置的距离
for i = 1:Nr_img
    range_pixel = image_range(i);
    for j = 1:Na_img
        az_pixel = image_azimuth(j);
        
        % 计算该像素到所有雷达位置的距离
        dist_to_radar = zeros(Na, 1);
        for k = 1:Na
            dist_to_radar(k) = sqrt((az_pixel - Radar_pos(k, 1))^2 + (range_pixel - Radar_pos(k, 2))^2);
        end
        
        % 计算该像素对应的回波强度（简化处理）
        sum_val = 0;
        for k = 1:Na
            % 根据距离计算相位补偿
            phase_compensation = exp(1i*4*pi*dist_to_radar(k)*f0/C);
            % 相位补偿后的回波值
            sum_val = sum_val + data(k, i) * phase_compensation;
        end
        
        BP_img(j, i) = abs(sum_val);
    end
    
    % 显示进度
    if mod(i, 10) == 0
        fprintf('Processing range pixel %d/%d\n', i, Nr_img);
    end
end

%% 图像显示和保存

% 对BP图像进行归一化和显示
BP_img = BP_img / max(max(BP_img));
BP_img = 20*log10(BP_img + eps);
BP_img(BP_img < -dyn_range) = -dyn_range;
BP_img = uint8((BP_img + dyn_range)/dyn_range * 255);

figure, imagesc(BP_img); axis image; set(gcf,'Color','w');
title('BP算法成像结果');
xlabel('距离向（采样点）'); ylabel('方位向（采样点）');

% 放大显示
figure;
flagx = (x > X1) & (x < X2);
flagy = (y > Y1) & (y < Y2);
NEWimg = BP_img(flagy, flagx);
imagesc(x(flagx), y(flagy), NEWimg, [-dyn_range 0]);
title('BP算法放大图像');
xlabel('距离向（采样点）'); ylabel('方位向（采样点）');

% 保存图像
imwrite(BP_img, 'SAR_BP_2.bmp', 'bmp');