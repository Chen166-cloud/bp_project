%% 参数设置
C = 2.9979e8;         % 光速
f0 = 1.4e9;           % 中心频率
lambda = C/f0;        % 波长
Br = 200e6;           % 带宽（显式定义）
Tp = 1.0e-6;          % 脉冲持续时间
V = 17;               % 平台速度
Fr = 400e6;           % 距离采样率
Fa = 1000;            % 方位采样率
PRF = Fa;             
R0 = 3e-6 * C / 2;    % 最短斜距

% 成像场景参数
scene_size = 100;      % 场景大小（像素）
range_res = C/(2*Br);  % 距离分辨率
az_res = lambda/(2*V*(Tp*Fa)); % 方位分辨率

%% 生成点目标回波（示例：单个点目标）
target_pos = [0, R0];  % 目标位置（方位位置, 斜距）

% 生成距离压缩后的回波数据（简化为理想点目标模型）
N_az = 1024;           % 方位向采样点数
N_rg = 1024;           % 距离向采样点数
echo_data = zeros(N_az, N_rg);
echo_data(round(N_az/2), round(N_rg/2)) = 1; % 点目标位于中心

%% FFBP算法核心步骤
% -------------------- 子孔径划分 --------------------
num_levels = 3;        % 分级层数（示例分3级）
sub_apertures = 2^num_levels; % 初始子孔径数（示例：8个子孔径）

% 初始化子孔径图像
sub_images = cell(sub_apertures, 1);

% -------------------- 第1级子孔径BP成像 --------------------
for k = 1:sub_apertures
    % 划分子孔径范围（示例均匀划分）
    az_start = (k-1)*N_az/sub_apertures + 1;
    az_end = k*N_az/sub_apertures;
    sub_data = echo_data(az_start:az_end, :);
    
    % 在局部极坐标系下生成低分辨率图像（传递Br参数）
    [sub_image, rg_grid, az_grid] = local_BP(sub_data, R0, V, lambda, Br, scene_size, PRF, N_az,C);
    sub_images{k} = sub_image;
end

% -------------------- 逐级融合子图像 --------------------
for level = num_levels-1:-1:1
    new_sub_images = cell(2^level, 1);
    for k = 1:2^level
        % 合并相邻两个子孔径图像
        img1 = sub_images{2*k-1};
        img2 = sub_images{2*k};
        
        % 坐标系插值融合（简化为最近邻插值）
        fused_img = (img1 + img2)/2; 
        new_sub_images{k} = fused_img;
    end
    sub_images = new_sub_images;
end

% 最终图像
final_image = sub_images{1};

%% 结果显示
figure;
imagesc(abs(final_image));
title('FFBP成像结果');
xlabel('距离向'); ylabel('方位向');
colormap('gray');

%% 局部BP成像函数（添加Br作为输入参数）
function [image, rg_grid, az_grid] = local_BP(data, R0, V, lambda, Br, scene_size, PRF, N_az,C)
    [~, N_rg] = size(data);
    image = zeros(scene_size, scene_size);
    
    % 极坐标系网格
    max_theta = atan((scene_size/2 * lambda)/(2*V*N_az/PRF));
    theta = linspace(-max_theta, max_theta, scene_size);
    r = linspace(R0-50, R0+50, scene_size); % 示例斜距范围
    
    for i = 1:scene_size
        for j = 1:scene_size
            % 计算每个像素点的相干积累
            phase_sum = 0;
            for az_idx = 1:size(data, 1)
                % 计算斜距历程
                R = sqrt(r(j)^2 + (V*az_idx/PRF - theta(i)*r(j))^2);
                rg_bin = round((R - R0) * 2*Br/C * N_rg) + N_rg/2; % 使用传递的Br参数
                
                if rg_bin >=1 && rg_bin <= N_rg
                    phase = exp(1j*4*pi/lambda * R);
                    phase_sum = phase_sum + data(az_idx, rg_bin) * phase;
                end
            end
            image(i,j) = abs(phase_sum);
        end
    end
    rg_grid = r;
    az_grid = theta;
end