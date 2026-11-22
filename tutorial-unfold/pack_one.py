from pathlib import Path
from scipy.io import loadmat
import numpy as np
p=Path(__file__).parent
s=(p/'compare_tutorial_unfolding.m').read_text()
s=s.replace('compare_tutorial_unfolding','run_tutorial_comparison')
s=s.replace("if nargin<2, opt=struct; end","""if nargin<1 || isempty(repo)
    [name, folder]=uigetfile('solve_BLP_AO.m','Select the authors'' solve_BLP_AO.m');
    if isequal(name,0), out=[]; return; end
    repo=fileparts(fileparts(fullfile(folder,name)));
end
if nargin<2, opt=struct; end""")
s=s.replace("'resume',false,'folder','tutorial_unfold_results'","'resume',true,'folder',''")
s=s.replace('o=defaults;',"""o=defaults;
if isempty(o.folder)
    o.folder=fullfile(repo,sprintf('unfold_K%d_G%g_D%d_T%d_E%d_seed%d', ...
        o.K,o.gammaDB,o.depth,o.nTrain,o.epochs,o.seed));
end""")
s=s.replace("if o.selfTest, selftest(); out=true; return; end","selftest();\nif o.selfTest, out=true; return; end")
s=s.replace("if o.resume\n    assert(isfile(checkpoint),'Missing trained_model.mat. Use the original run folder or set opt.folder.');","fprintf('Results folder: %s\\n',o.folder);\nif o.resume && isfile(checkpoint)")
s=s.replace("f=load(fullfile(fileparts(mfilename('fullpath')),'beam_checks.mat'));","f=embedded_fixture();")
s+='\nfunction f=embedded_fixture()\n% Independent reference values stored in column-major order.\n'
for key,a in loadmat(p/'beam_checks.mat').items():
 if key.startswith('_'): continue
 vals=a.flatten(order='F')
 def emit(values):
    return '[ ...\n'+' ...\n'.join(' '.join(format(float(x),'.17g') for x in values[i:i+6]) for i in range(0,len(values),6))+']'
 arr=emit(vals.real)
 if np.iscomplexobj(a):arr='complex('+arr+', ...\n'+emit(vals.imag)+')'
 s+=f"f.{key}=reshape({arr},[{ ' '.join(map(str,a.shape)) }]);\n"
s+='end\n'
(p/'run_tutorial_comparison.m').write_text(s)
