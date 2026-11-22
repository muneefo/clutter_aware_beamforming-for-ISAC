function path = record_pc_resources(runFolder)
% Record Windows hardware and recover historical fields from the run manifest.
% Call from the MATLAB PC used for the benchmark. No simulations are started.
if nargin < 1, runFolder = pwd; end
assert(isfolder(runFolder),'Select the existing experiment folder.');
r = struct;
r.capturedAt = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd HH:mm:ss Z'));
r.currentMATLAB = version;
r.currentRelease = version('-release');
r.currentPlatform = computer;
r.recordScope = 'Current machine snapshot; run-time fields are separately identified.';
manifestFile = fullfile(runFolder,'manifest.mat');
if isfile(manifestFile)
    data = load(manifestFile,'manifest');
    if isfield(data,'manifest')
        names = intersect({'matlab','platform','backend'},fieldnames(data.manifest));
        for k = 1:numel(names)
            r.atRun.(names{k}) = data.manifest.(names{k});
        end
    end
end
if ispc
    r.cpu = winquery('Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors');
    r.os = winquery('Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber');
    r.memory = winquery('Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory');
    r.adapters = winquery('Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion');
end
r.toolbox = ver('optim');
r.BLAS = version('-blas');
if exist('cvx_solver','file')
    try
        r.currentCVXDefault = strtrim(evalc('cvx_solver'));
    catch
        r.currentCVXDefault = 'unavailable';
    end
end
r.currentPoolWorkers = [];
if exist('gcp','file')
    try
        pool = gcp('nocreate');
        if isempty(pool), r.currentPoolWorkers = 0;
        else, r.currentPoolWorkers = pool.NumWorkers; end
    catch
        % Leave unavailable if the toolbox cannot be queried.
    end
end
path = [tempname(runFolder) '_pc_resources.json'];
fid = fopen(path,'w');
assert(fid ~= -1,'Could not create the PC resource report.');
closeFile = onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(r,'PrettyPrint',true));
fprintf('PC resource report: %s\n',path);
end

function result = winquery(query)
cmd = ['powershell.exe -NoProfile -NonInteractive -Command "(' query ') | ConvertTo-Json -Compress"'];
[status,raw] = system(cmd);
if status == 0
    try
        result = jsondecode(strtrim(raw));
        return
    catch
    end
end
result = 'unavailable';
end
