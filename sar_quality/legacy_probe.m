% Small deterministic evaluation of the ORIGINAL RD phase filters.
% Does not edit or run the old scripts' image writers.
C=2.9979e8; f0=1.4e9; lambda=C/f0; V=17; Fa=1000;
base=fileparts(mfilename('fullpath'));
vars=whos('-file',fullfile(base,'..','data_5000_10000.mat'));
idx=find(strcmp({vars.name},'data_5000_10000'),1);
Na=vars(idx).size(1); R0=3e-6*C/2;
feta=((0:Na-1)-Na/2)/Na*Fa;
D=sqrt(1-lambda^2*feta.^2/(4*V^2));
Haz=exp(1i*4*pi*R0*D'*f0/C);
record=struct('Na',Na,'cutoff_hz',2*V/lambda,...
 'complex_D_count',sum(imag(D)~=0),'nonfinite_Haz_count',sum(~isfinite(Haz)),...
 'Haz_count',numel(Haz),'conjugate_transpose_used',true);
record.zero_times_infinite_is_nan=any(isnan(zeros(size(Haz)).*Haz));
fid=fopen(fullfile(fileparts(mfilename('fullpath')),'results','legacy_filter_probe.json'),'w');
fprintf(fid,'%s',jsonencode(record,PrettyPrint=true));fclose(fid);
disp(record);
