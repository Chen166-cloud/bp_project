function fc=DoppCentEsti_SLC(raw,Vr,f0,PRF,fs,beam_width_az,swst,chirp_rg_BW,chirp_rg_T,az_fft_size,rg_fft_size)
% slc = SAR_process(raw,Vr,f0,PRF,fs,beam_width_az,swst,chirp_rg_BW,chirp_rg_T)
%
% This function implements basic range-Doppler algorithm to focus a
% single-look-complex (SLC) image out of airborne SAR RAW data.
%
% Inputs:
%   Raw_data: baseband RAW data in double complex format (range along lines)
%   V: Effective satellite velocity [m/s]
%   f0: SAR central frequency [Hz]
%   PRF: pulse repetition ferquency [Hz]
%   fs: range sampling frequency [Hz]
%   K_r: FM Rate Range Chirp [1/s^2] --> "+" = up-chirp
%   tau_p: Chirp duration [s]
%   R_0: Range to center of antenna footprint [m]
%   ta: Aperture time [s]
%   chirp_rg_T: range emitted chirp duration [s]
%
% Outputs:
%   slc: single-look-complex focused image in double complex format
%
% Note:
%   - Range emitted up-chirp is assumed. Otherwise conjugate raw input data
%   - Processing assumes low squinted data
%   - No performance requirements have been considered in writing this code
%   since its main objective is educational
%
% Author: Peng Liu

% Additional parameters
Nr=size(raw,2);
near_rg_offset = 0;         % Samples to discard from near range (until echo is received)
c = 299792458;              % Speed of light [m/s]
lambda = c/f0;              % SAR wavelength [m]
doppler_pol_deg = 1;        % doppler centroid fitting polynomial degree

% Handler for correlating 2 signals {x,y}
compress = @(x,y,nfft)(ifft(conj(fft(x,nfft)).*fft(y,nfft)));

% Handler to evaluate slant range [m] as function of range pixel number
slant_range = @(p)(c*swst/2 + (p-1)*c/fs/2);


%% RAW data conditioning
display 'RAW data conditioning'
% Subtract raw mean value
raw = raw - mean(mean(raw));
% Display raw data figure; 
imagesc(abs(raw));colormap('gray'); title('Raw data (unfocused)'); 
xlabel('Range Pixels'); ylabel('Azimuth Pixels'); 


%% Range compression
display 'Range compression'

chirp_rg = chirp_comp(chirp_rg_BW,chirp_rg_T,fs);   % Reference range chirp

% Compress each range line
data_rg_compr = zeros(size(raw));                       % Initialize matrix
for k=1:size(raw,1)
    aux = compress(chirp_rg,raw(k,:),rg_fft_size);      % Correlation
    data_rg_compr(k,:) = aux(1:size(data_rg_compr,2));  % Save result for line k，第k个距离线上的脉压结果
end
% Display compressed figure; 
imagesc(abs(data_rg_compr)); title('Raw data compressed)'); 
xlabel('Range Pixels'); ylabel('Azimuth Pixels'); 

%% Doppler centroid estimation
display 'Doppler centroid estimation'

% Change to range-Doppler domain in power
data_doppler = abs(fftshift(fft(data_rg_compr),1)).^2;     % 计算脉压后的距离向多普勒功率谱

% Filter some noise
data_doppler = filter2(ones(101,3)/303,data_doppler,'same'); %功率谱滤波，二维FIR(Finite Impulse Response)滤波器，
         %same-返回滤波数据的中心部分，大小与 data 相同。ones：系数矩阵H，对数据滤波。同于 conv2(X,rot90(H,2))

% Search for each column peak power (position of centroid)， 功率谱矩阵每(距离单元)列 的峰值位置
[~,doppler_idx] = max(data_doppler);  % For matrix,max(X) is a row vector containing the maximum element from each column.

% Map found indexes into frequencies
doppler_frec_idx = linspace(-PRF/2,PRF/2,size(data_rg_compr,1));
doppler_centroid_v = doppler_frec_idx(doppler_idx(near_rg_offset+1:end));   % 距离向多普勒中心频率的小数部分

% Polynomial fit of Doppler centroid，多普勒中心频率的曲线拟合
warning off
doppler_centroid_coef = polyfit(near_rg_offset+1:size(data_rg_compr,2),doppler_centroid_v,doppler_pol_deg);
warning on
%
% Handler to evaluate Doppler centroid [Hz] as function of range pixel numb
doppler_centroid = @(p)(polyval(doppler_centroid_coef,p));


%% Range cell migration correction
display 'RCMC'

% Define Doppler frequencies vector
fdopp = linspace(-PRF/2,PRF/2,size(data_rg_compr,1));

% Range/Doppler domain
data_rg_compr_fft = fftshift(fft(data_rg_compr),1);

% Original slant range spacing vector
R1 = slant_range(1:size(data_rg_compr,2));

for k=1:size(data_rg_compr,1)
    % New slant range spacing vector, compensates range migration for every
    % doppler frequency fdopp(k)
    R2 = R1 + R1*(lambda*fdopp(k)).^2/(8*Vr^2);
    
    % Interpolate to compensate migration
    data_rg_compr_fft(k,:) = interp1(R1,data_rg_compr_fft(k,:),R2,'PCHIP',NaN);
end

% Back to Range-Azimuth domain
data_rg_compr_rcmc = ifft(ifftshift(data_rg_compr_fft,1));

size(data_rg_compr_rcmc) 
%% Azimuth compression
display 'Azimuth compression'

% Doppler BW to process
chirp_az_BW = 2/lambda*Vr*beam_width_az*pi/180;

% Compress each column in azimuth
slc = zeros(size(data_rg_compr_rcmc));       % initialize matrix
phi_accc=zeros(Nr,1); 
Fdfrac=zeros(Nr,1);
for k=near_rg_offset+1:size(data_rg_compr_rcmc,2)
    
    % Evaluate Doppler centroid
    DC = doppler_centroid(k);
    
    phi_accc(k) = 2*pi*DC/PRF;   %  ACCC 值
    Fdfrac(k)=DC;
        if DC<0 
            Fdfrac(k)=Fdfrac(k)+PRF; 
        end 

    % Azimuth chirp duration: azimuth footprint length [m] / Vr
    chirp_az_T = slant_range(k)*(beam_width_az*pi/180)/Vr;
    
    % generate azimuth chirp and conjugate (down-chirp)
    chirp_az = chirp_comp(chirp_az_BW,chirp_az_T,PRF,DC)';
    
    % Compress and save result
    aux = compress(chirp_az,data_rg_compr_rcmc(:,k),az_fft_size);
    slc(:,k) = aux(1:size(slc,1));
end
F_fr=median(Fdfrac);
    Fr =fs;
    fr=linspace(-Fr/2,Fr/2,Nr); 
    clf;    
    plot(fr, phi_accc ),   grid,   hold on
    % the slope of the ACCC angle versus range frequecy is measured by using the linear polyfit 
    [accc_new,n]=throw_unormal_value_r(phi_accc,0.1);
    coeff = polyfit( fr',phi_accc, 1 ); %第一个元素是斜率,第二个元素是截距
%    coeff = polyfit( fr',accc_new, 1 ); 
%   p=polyfit(x,y,n): x,y为数据点，n为多项式阶数，返回p为幂次从高到低的多项式系数向量p。x必须是单调的。
%   多项式曲线求值函数： y=polyval(p,x) ,返回自变量x在给定系数P的多项式的值。
    plot(fr, accc_new,'g--' );
    p_val=polyval(coeff,fr');
    plot( fr, p_val, 'r--' );
    xlabel('频率 （Hz）', 'FontS', 12 )
    ylabel('ACCC 相角(弧度)','FontS', 12 )
    title('多波长法相角估计', 'FontS', 13 )
    alfa=coeff(1);    %  A是phi_accc 相对于距离频率 f_tao 的斜率
    %estimate the doppler ambiguity number k: 
    Nspect=1;
    f_etac=PRF*f0*alfa/(2*pi)/Nspect;
    Mamb=round((f_etac-F_fr)/PRF); 
   figure,imagesc(abs(slc));axis image;set(gcf,'Color','w');
   fc = F_fr+Mamb*PRF;
display 'END'
end


function [y,t] = chirp_comp( BW, T, fs, CA, phi)
% [y,t] = chirp_comp( BW , T, fs , CA , phi )
%
% Generates baseband (complex) up-chirp vector.
%
% Inputs:
%   BW: bandwidth [Hz]
%   T: duration [s]
%   fs: sampling frequency [Hz]
%   CA: carrier [Hz] (optional)
%   phi: initial phase [rad] (optional)
% 
% Outputs:
%   y: sampled complex up-chirp vector
%   t: time vector used to sample the chirp
%
% Example:
%   [y,t] = chirp_comp( 40e6 , 10e-6 , 50e6);
%   specgram(y,[],50e6,hanning(70),65)
%
%   Author: Mario Azcueta <mazcueta@gmail.com>

if ~exist('CA','var')
    CA = 0;
end
if ~exist('phi','var')
    phi = 0;
end
if nargin<3
    error('Error. Not enough input arguments.')
end
if BW/2+CA>fs
    display('Warning. Aliasing will be produced since BW/2+CA < fs');
end

b = -BW/2 + CA;
a = (BW/2 + CA - b)/(2*T);

t = 0:1/fs:T;
y = exp(1i*2*pi*(a*t.^2+b*t+phi/(2*pi)));

end