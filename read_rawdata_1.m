% 解包程序，2018.08.07
% 载波f0 = 1.4e9;
% 采样率fs = 400e6;

clear all

% 读文件
[FileName,PathName] = uigetfile('2018-08-14_18-54-08.dat','选择待解包数据');
PathNameAndFileName = strcat(PathName,FileName);
disp(FileName)
fid = fopen(PathNameAndFileName,'r');
disp(PathNameAndFileName)

% 搜索第一帧帧头，防止前面丢数据
if 1
    temp = fread(fid,1e7,'uchar');
    offset = 0;
    while((temp(offset+1)~=0)||(temp(offset+2)~=52)||(temp(offset+3)~=0)||(temp(offset+4)~=52)||....
            (temp(offset+5)~=243)||(temp(offset+6)~=250)||(temp(offset+7)~=243)||(temp(offset+8)~=250))
        offset = offset + 1;
    end
    fseek(fid,offset,'bof');
    disp(offset)
end


%% 读取参数


fseek(fid,48,'cof');
Tp = fread(fid,1,'int16')*10e-9;
fseek(fid,10,'cof');
window_width = fread(fid,1,'int16')*10e-9;

fs = 400e6;  % 采样频率
Nr = fs*window_width;
Npulse = Tp*fs;
zhen_Size = 192 + 4*Nr;  %一帧字节数

fseek(fid,offset,'bof');


%% 算方位向点数
fseek(fid,0,'eof');
datasize = ftell(fid);
fseek(fid,offset,'bof');
Na = floor((datasize-offset)/zhen_Size);disp(Na)% 需要减去offset，不然可能会出bug

% 开头跳过Start byte num个脉冲
Startbytenum = 0*zhen_Size;
% Startbytenum = offset*zhen_Size;
fseek(fid,Startbytenum,'cof');
Na = Na - Startbytenum/zhen_Size;disp(Na)

% Na = 10000; 
% Downsampling = zhen_Size*1;

%% 解包数据
% Na = int64(Na);
Nr = int64(Nr); 
% 辅助数据大小，方位向脉冲为基准的
Startmark = zeros(Na,32);
zhen_num = zeros(Na,1);
Br = zeros(Na,1);
PPS_pulse_num = zeros(Na,1);
Tp = zeros(Na,1);
sampling_delay = zeros(Na,1);
PRF = zeros(Na,1);
PRF_start = zeros(Na,1);
window_width = zeros(Na,1);
PRF_step = zeros(Na,1);
PRF_continuous_num = zeros(Na,1);
PRF_end = zeros(Na,1);

heading = zeros(Na,1);
Pitch = zeros(Na,1);
Roll = zeros(Na,1);
Gyro_X = zeros(Na,1);
Gyro_Y = zeros(Na,1);
Gyro_Z = zeros(Na,1);
Acc_X = zeros(Na,1);
Acc_Y = zeros(Na,1);
Acc_Z = zeros(Na,1);
Latitude = zeros(Na,1);
Longtitude = zeros(Na,1);
Atitude = zeros(Na,1);
East_speed = zeros(Na,1);
North_speed = zeros(Na,1);
Vertical_speed = zeros(Na,1);
Latitude_GNSS = zeros(Na,1);
Longtitude_GNSS = zeros(Na,1);
ms_gps = zeros(Na,1);
GNSS_info1 = zeros(Na,1);
GNSS_info2 = zeros(Na,1);
solnSVs = zeros(Na,1);
V_altency = zeros(Na,1);
New_GPS = zeros(Na,1);
Endmark = zeros(Na,32);
echo=zeros(Na,1);

% 回波数据
raw_data = zeros(Na,Nr);
h=waitbar(0,'please wait');
for m = 1:Na
    Startmark(m,:) = fread(fid,32,'uchar');
    fseek(fid,4,'cof');
    zhen_num(m,:) = fread(fid,1,'uint32');   % 第m帧(距离线)的记录编号，26781，26782 ...
    Br(m,:) = fread(fid,1,'uint32'); % 带宽2e8
    PPS_pulse_num(m,:) = fread(fid,1,'uint32')*10e-9; % 秒脉冲计数
    Tp(m,:) = fread(fid,1,'int16')*10e-9; % 调频脉冲宽度1e-6
    sampling_delay(m,:) = fread(fid,1,'int16')*10e-9-0.2e-6; % 采样延迟3e-6
    PRF(m,:) =  1/(fread(fid,1,'uint32')*10e-9); % 脉冲重复频率=1000
    PRF_start(m,:) = 1/(fread(fid,1,'uint32')*10e-9); % 1000
    window_width(m,:) = fread(fid,1,'int16')*10e-9; % 采样窗宽度5*e-5
    PRF_step(m,:) = fread(fid,1,'uchar');
    PRF_continuous_num(m,:) = fread(fid,1,'uchar'); 
    PRF_end(m,:) = 1/(fread(fid,1,'uint32')*10e-9); 
    fseek(fid,3,'cof');
    %% 惯导参数
    heading(m,:) = fread(fid,1,'int16')/100; % 朝向=292.64
    Pitch(m,:) = fread(fid,1,'int16')/100; % 
    Roll(m,:) = fread(fid,1,'int16')/100; % 
    Gyro_X(m,:) = fread(fid,1,'int32'); % 陀螺仪x坐标
    Gyro_Y(m,:) = fread(fid,1,'int32'); % 陀螺仪y坐标
    Gyro_Z(m,:) = fread(fid,1,'int32'); % 陀螺仪z坐标
    Acc_X(m,:) = fread(fid,1,'int32');
    Acc_Y(m,:) = fread(fid,1,'int32');
    Acc_Z(m,:) = fread(fid,1,'int32');
    Latitude(m,:) = fread(fid,1,'int64')/1e9; % 纬度=34.0724
    Longtitude(m,:) = fread(fid,1,'int64')/1e9; % 经度=108.2131
    Atitude(m,:) = fread(fid,1,'int32')/1000; % 海拔=446.7010
    East_speed(m,:) = fread(fid,1,'int32')/100; 
    North_speed(m,:) = fread(fid,1,'int32')/100;
    Vertical_speed(m,:) = fread(fid,1,'int32')/100;
    Latitude_GNSS(m,:) = fread(fid,1,'int64')/1e9;
    Longtitude_GNSS(m,:) = fread(fid,1,'int64')/1e9;
    ms_gps(m,:) = fread(fid,1,'uint32');
    GNSS_info1(m,:) = fread(fid,1,'uchar');
    GNSS_info2(m,:) = fread(fid,1,'uchar');
    solnSVs(m,:) = fread(fid,1,'uchar');
    V_altency(m,:) = fread(fid,1,'uint16');
    New_GPS(m,:) = fread(fid,1,'uchar');
    fseek(fid,1,'cof');
         
    echo = fread(fid,2*Nr,'int16');   % 回波记录
    data_I = echo(1:2:end).';         % 实部在奇数位 
    data_Q = echo(2:2:end).';
    raw_data(m,:) = data_I + 1i*data_Q;
    
    Endmark(m,:) = fread(fid,32,'uchar');
    waitbar(m/13388,h);
end
fclose(fid);
close all;

