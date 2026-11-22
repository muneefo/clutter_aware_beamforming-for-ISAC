from pathlib import Path
import os,subprocess
root=Path('/workspace/scratch/ecac02d9f31b')
out=root/'paper1-fixed-array'
env=os.environ.copy()
env['TEXMFDBS']='/nonexistent'
env['TEXMF']='{/usr/share/texmf,/usr/share/texlive/texmf-dist}'
env['TEXINPUTS']=str(out)+':'
def command(name):
 return ['pdftex','-fmt='+str(root/'research-synthesis-work/engine/local-pdflatex.fmt'),'-interaction=nonstopmode','-halt-on-error',r'\pdfoutput=1 \pdfmapfile{+/usr/share/texmf/fonts/map/dvips/lm/lm.map} \pdfmapfile{+/usr/share/texlive/texmf-dist/fonts/map/dvips/amsfonts/cm.map} \pdfmapfile{+/usr/share/texlive/texmf-dist/fonts/map/dvips/amsfonts/symbols.map} \input{'+name+'.tex}']
commands=[command('fixed-array-system'),['bibtex','fixed-array-system'],command('fixed-array-system'),command('fixed-array-system'),command('system-figure')]
for i,cmd in enumerate(commands):
 p=subprocess.run(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (out/'work'/f'compile-{i}.txt').write_text(p.stdout)
 print('Pass',i,'exit',p.returncode)
 if p.returncode:
  print(p.stdout[-3500:]);raise SystemExit(1)
print('Build completed')
