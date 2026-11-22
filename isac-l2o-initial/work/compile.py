from pathlib import Path
import os,subprocess
root=Path('/workspace/scratch/ecac02d9f31b');out=root/'isac-l2o-initial'
env=os.environ.copy();env['TEXMFDBS']='/nonexistent';env['TEXMF']='{/usr/share/texmf,/usr/share/texlive/texmf-dist}';env['TEXINPUTS']=str(out)+':'
cmd=['pdftex','-fmt='+str(root/'research-synthesis-work/engine/local-pdflatex.fmt'),'-interaction=nonstopmode','-halt-on-error',r'\pdfoutput=1 \pdfmapfile{+/usr/share/texmf/fonts/map/dvips/lm/lm.map} \pdfmapfile{+/usr/share/texlive/texmf-dist/fonts/map/dvips/amsfonts/cm.map} \pdfmapfile{+/usr/share/texlive/texmf-dist/fonts/map/dvips/amsfonts/symbols.map} \input{derivation.tex}']
for i,command in enumerate([cmd,['bibtex','derivation'],cmd,cmd]):
 p=subprocess.run(command,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'work'/f'compile-{i}.txt').write_text(p.stdout);print('Pass',i,p.returncode)
 if p.returncode:print(p.stdout[-2500:]);raise SystemExit(1)
