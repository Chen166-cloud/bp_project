%% 参数设置
% 雷达参数
C = 2.9979e8;                                                              % 光速
f0 = 1.4e9;                                                                % 中心频率
lambda = C/f0;                                                             % 波长
Br=200e6;                                                                  %带宽
Tp=1.0e-6 ;                                                                % 脉冲持续时间
% 飞行平台
V = 17;                                                                    % 平台速度
fc_max=2*V/lambda;                                                         % 多普勒中心最大值
% 距离向
Kr =  -Br./Tp;                                                             % 调频率
Fr = 400e6;                                                                % 距离向采样率
dtau = 1/Fr;                                                               % 距离向采样时间间隔
% 方位向
Fa = 1000;                                                                 % 方位向采样率
PRF = Fa;
deta = 1/Fa;                                                               % 方位向采样时间间隔
% 其他参数
t0 = 3e-6;                                                                 % 获取数据时间，采样延迟
R0 = t0*C/2;                                                               % 最短斜距

%% 获取原始数据（通道1）
% 注意：确保 'data_5000_10000.mat' 文件在当前工作目录或MATLAB路径中
load('data_5000_10000.mat');
data=double(data_5000_10000);
[Na,Nr] = size(data);

%% SAR图像设定：图像动态范围及放大图像的范围[X1 X2,Y1 Y2]
x = 1:1:Nr;
y =1:1:Na;
YC=Na/2; YW=800; Y1=YC-YW; Y2=YC+YW;  X1=1; X2=Nr;
dyn_range = 60;              % dB of dynamic range to display

display 'RAW data conditioning'
data = data - mean(mean(data));           % Subtract raw mean value
data = data.* (hamming(Na)*hamming(Nr)'); % 加窗去噪：二维窗

%% 多普勒中心估计 (参数保持不变，但实际计算未使用fc)
fc = 0;                                                              % 多普勒中心采用给定值
Tr=Tp;
chirp_rg_BW = abs(Kr)*Tr;                                    %带宽 200M
DeltaR = C/(2*chirp_rg_BW);                                           %距离向分辨率
La=2*DeltaR;
beam_width_az = 30;
Za = 0;
ta=(Na+Za-1)*deta;

%% 参考数据设置 (参数保持不变)
Rref = (t0-Tp/2+Nr/2*dtau)*C/2;
Vref = V;
fref = fc;

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

%% BP算法核心部分 (多线程版本)

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

% --- 并行池管理 ---
% 获取当前并行池（如果存在）
current_pool = gcp('nocreate');

% 如果不存在活动池，则尝试创建一个
if isempty(current_pool)
    try
        % 尝试创建并行池
        parpool; % 使用默认设置创建池
        current_pool = gcp; % 获取新创建的池对象
        fprintf('已成功创建新的并行池。\n');
    catch ME
        % 如果创建失败（例如，许可证问题），则发出警告并继续（将使用单线程）
        warning('无法创建并行池: %s', ME.message);
        fprintf('将继续以单线程模式运行。\n');
        current_pool = []; % 确保 current_pool 为空
    end
else
    fprintf('已连接到现有的并行池。\n');
end

% 获取工作进程数量
if ~isempty(current_pool)
    num_workers = current_pool.NumWorkers;
    fprintf('开始多线程BP算法计算，使用 %d 个并行工作进程...\n', num_workers);
    use_parallel = true;
else
    num_workers = 1; % 单线程
    fprintf('开始单线程BP算法计算...\n');
    use_parallel = false;
end
% --- 并行池管理结束 ---

% --- 核心计算 ---
tic; % 开始计时

if use_parallel
    % 使用 parfor 实现多线程
    parfor i = 1:Nr_img
        % parfor 循环内部的变量需要独立于其他迭代
        % 为每个线程重新计算 Radar_pos
        Radar_pos = zeros(Na, 2);
        for idx = 1:Na
            Radar_pos(idx, :) = [V * eta(idx), 0];  % 假设沿X方向匀速运动
        end

        range_pixel = image_range(i);
        temp_column = zeros(Na_img, 1);

        for j = 1:Na_img
            az_pixel = image_azimuth(j);
            dist_to_radar_vec = sqrt((az_pixel - Radar_pos(:, 1)).^2 + (range_pixel - Radar_pos(:, 2)).^2);
            phase_compensation_vec = exp(1i*4*pi*dist_to_radar_vec*f0/C);
            % 注意 data 的索引: data(方位, 距离)，即 data(行, 列)
            sum_val = sum(data(:, i) .* phase_compensation_vec);
            temp_column(j) = abs(sum_val);
        end

        BP_img(:, i) = temp_column;

        if mod(i, 50) == 0
            fprintf('Processing range pixel %d/%d\n', i, Nr_img);
        end
    end
else
    % 使用普通 for 循环实现单线程
    for i = 1:Nr_img
        Radar_pos = zeros(Na, 2);
        for idx = 1:Na
            Radar_pos(idx, :) = [V * eta(idx), 0];
        end

        range_pixel = image_range(i);
        temp_column = zeros(Na_img, 1);

        for j = 1:Na_img
            az_pixel = image_azimuth(j);
            dist_to_radar_vec = sqrt((az_pixel - Radar_pos(:, 1)).^2 + (range_pixel - Radar_pos(:, 2)).^2);
            phase_compensation_vec = exp(1i*4*pi*dist_to_radar_vec*f0/C);
            sum_val = sum(data(:, i) .* phase_compensation_vec);
            temp_column(j) = abs(sum_val);
        end

        BP_img(:, i) = temp_column;

        if mod(i, 50) == 0
            fprintf('Processing range pixel %d/%d\n', i, Nr_img);
        end
    end
end

toc; % 结束计时并显示耗时
% --- 核心计算结束 ---

%% 图像显示和保存

% 对BP图像进行归一化和显示
BP_img = BP_img / max(max(BP_img));
BP_img = 20*log10(BP_img + eps);
BP_img(BP_img < -dyn_range) = -dyn_range;
BP_img = uint8((BP_img + dyn_range)/dyn_range * 255);

figure, imagesc(BP_img); axis image; set(gcf,'Color','w');
if use_parallel
    title('BP算法成像结果 (多线程)');
else
    title('BP算法成像结果 (单线程)');
end
xlabel('距离向（采样点）'); ylabel('方位向（采样点）');

% 放大显示
figure;
flagx = (x > X1) & (x < X2);
flagy = (y > Y1) & (y < Y2);
NEWimg = BP_img(flagy, flagx);
imagesc(x(flagx), y(flagy), NEWimg, [-dyn_range 0]);
if use_parallel
    title('BP算法放大图像 (多线程)');
else
    title('BP算法放大图像 (单线程)');
end
xlabel('距离向（采样点）'); ylabel('方位向（采样点）');

% 保存图像
if use_parallel
    imwrite(BP_img, 'SAR_BP_2_multithreaded.bmp', 'bmp');
    fprintf('多线程BP算法处理完成，结果已保存为 SAR_BP_2_multithreaded.bmp。\n');
else
    imwrite(BP_img, 'SAR_BP_2_singlethreaded.bmp', 'bmp');
    fprintf('单线程BP算法处理完成，结果已保存为 SAR_BP_2_singlethreaded.bmp。\n');
end











