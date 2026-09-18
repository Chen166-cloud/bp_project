% Execute the existing demonstration, preserving its arithmetic and outputs.
base=fileparts(mfilename('fullpath'));
set(groot,'defaultFigureVisible','off');
run(fullfile(base,'..','dot.m'));
record=struct('shape',size(final_image),'nonzero_pixels',nnz(final_image),...
 'maximum',max(final_image(:)),'finite',all(isfinite(final_image(:))));
fid=fopen(fullfile(base,'results','dot_probe.json'),'w');
fprintf(fid,'%s',jsonencode(record,PrettyPrint=true));fclose(fid);
close all;disp(record);
