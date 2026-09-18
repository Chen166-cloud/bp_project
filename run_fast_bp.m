function run_fast_bp(pythonExecutable)
% Reproduce FFBP imaging from the original DAT using the CPU-parallel engine.
% Usage: run_fast_bp('D:\develop\Anaconda\python.exe')
% Results: fast_bp/results/ffbp_final_ground.mat (complex SAR image + metre axes)
if nargin < 1
    pythonExecutable = 'D:\develop\Anaconda\python.exe';
end
root = fileparts(mfilename('fullpath'));
script = fullfile(root, 'fast_bp', 'run.py');
positionFile = fullfile(root, 'fast_bp', 'results', 'autofocus_position.npy');
if ~isfile(positionFile)
    error('FFBP:MissingAutofocus', 'Run fast_bp/diagnose.py, run.py, and autofocus.py first; see fast_bp/README.md.');
end
cmd = sprintf('"%s" "%s" --position-file "%s" --name ffbp_final', pythonExecutable, script, positionFile);
[status, output] = system(cmd, '-echo');
if status ~= 0
    error('FFBP:ExecutionFailed', '%s', output);
end
result = load(fullfile(root, 'fast_bp', 'results', 'ffbp_final_ground.mat'));
amplitude = abs(result.image_complex);
imageDb = 20*log10(amplitude / max(amplitude(:)) + eps);
figure('Color', 'w');
imagesc(result.cross_track_m, result.along_track_m, imageDb, [-65 -10]);
axis xy image; colormap gray; colorbar;
xlabel('Cross-track distance (m)'); ylabel('Along-track coordinate (m)');
title('Real-data fast factorized backprojection');
end
