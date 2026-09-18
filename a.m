% 生成LFM信号
T_p = 1e-6;       % 脉冲宽度
B = 200e6;        % 带宽
K = B / T_p;      % 调频率
Fs = 400e6;       % 采样率
t = -T_p/2:1/Fs:T_p/2;
s = exp(1j * pi * K * t.^2);  % 发射信号

% 设计匹配滤波器
h = conj(fliplr(s));  % 共轭反转

% 接收信号（含目标延时）
tau = 100e-9;         % 目标延时
r = [zeros(1, round(tau*Fs)), s(1:end-round(tau*Fs))];  % 延时后的回波

% 匹配滤波（卷积）
y = conv(r, h, 'same');  % 卷积输出

% 绘图
figure;
subplot(3,1,1); plot(t, real(s)); title('发射信号');
subplot(3,1,2); plot(t, real(h)); title('匹配滤波器 h(t) = s^*(-t)');
subplot(3,1,3); plot(abs(y)); title('匹配滤波输出（脉冲压缩后）');
xlabel('时间'); ylabel('幅度');