"""Run the ONNX tooth landmark model with standard FDI outputs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

FDI_ORDER = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,
             31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
IMG_H, IMG_W = 512, 1024


def prepare(image):
    h,w=image.shape
    scale=min(IMG_W/w,IMG_H/h)
    nw,nh=int(w*scale),int(h*scale)
    left,top=(IMG_W-nw)//2,(IMG_H-nh)//2
    canvas=np.zeros((IMG_H,IMG_W),dtype=image.dtype)
    canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    rgb=cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB).astype(np.float32)/255.0
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None]),(scale,left,top)


def refine_peak(hm,y,x):
    h,w=hm.shape
    x=max(1,min(w-2,int(x))); y=max(1,min(h-2,int(y)))
    vl,v0,vr=map(float,[hm[y,x-1],hm[y,x],hm[y,x+1]])
    du=vl-2*v0+vr
    dx=0.0 if abs(du)<1e-6 else 0.5*(vl-vr)/du
    vu,vd=map(float,[hm[y-1,x],hm[y+1,x]])
    du=vu-2*v0+vd
    dy=0.0 if abs(du)<1e-6 else 0.5*(vu-vd)/du
    return x+dx,y+dy


def peaks(hm):
    hm=hm.copy(); h,w=hm.shape; found=[]
    for _ in range(2):
        idx=int(hm.argmax()); y,x=divmod(idx,w); value=float(hm[y,x])
        if value<=0: break
        found.append((x,y,value))
        hm[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]=0
    return found


def resolve_conflicts(predictions,prob,min_sep,refine):
    attempted=set(); changed=True; h,w=prob.shape[-2:]
    while changed:
        changed=False
        points=[(i,p['x_lb'],p['y_lb'],p['confidence']) for i,p in enumerate(predictions) if p['exist']]
        for i,a in enumerate(points):
            for b in points[i+1:]:
                if np.hypot(a[1]-b[1],a[2]-b[2])>=min_sep: continue
                move=a[0] if a[3]<b[3] else b[0]
                if move in attempted: continue
                attempted.add(move)
                found=peaks(prob[move])
                if len(found)<2 or found[1][2]<0.6*max(found[0][2],1e-6): continue
                x,y,_=found[1]
                if refine: x,y=refine_peak(prob[move],y,x)
                predictions[move]['x_lb']=float(x/max(1,w-1)*(IMG_W-1))
                predictions[move]['y_lb']=float(y/max(1,h-1)*(IMG_H-1))
                changed=True


def predict(session,image,confidence=.30,refine=True,tta=True,min_sep=10):
    tensor,(scale,left,top)=prepare(image)
    suffix='_tta' if tta else ''
    heat,presence=session.run(['heatmap_logits_fdi'+suffix,'presence_logits_fdi'+suffix],{'image':tensor})
    # Stable sigmoid; all model outputs are logits.
    prob=1/(1+np.exp(-np.clip(heat[0],-80,80)))
    conf=1/(1+np.exp(-np.clip(presence[0],-80,80)))
    h,w=prob.shape[-2:]
    predictions=[]
    for i,fdi in enumerate(FDI_ORDER):
        exist=bool(conf[i]>=confidence and prob[i].max()>=.30)
        p=dict(fdi=fdi,exist=exist,confidence=float(conf[i]),x=None,y=None,x_lb=None,y_lb=None)
        if exist:
            y,x=divmod(int(prob[i].argmax()),w)
            if refine: x,y=refine_peak(prob[i],y,x)
            p['x_lb']=float(x/max(1,w-1)*(IMG_W-1))
            p['y_lb']=float(y/max(1,h-1)*(IMG_H-1))
        predictions.append(p)
    resolve_conflicts(predictions,prob,min_sep,refine)
    for p in predictions:
        if p['exist']:
            p['x']=float(np.clip((p['x_lb']-left)/scale,0,image.shape[1]-1))
            p['y']=float(np.clip((p['y_lb']-top)/scale,0,image.shape[0]-1))
    return predictions


def create_session(model,device='auto'):
    available=ort.get_available_providers()
    if device=='cuda' and 'CUDAExecutionProvider' not in available:
        raise RuntimeError('CUDA provider unavailable; install onnxruntime-gpu and its runtime dependencies, or use --device cpu')
    providers=['CPUExecutionProvider']
    if device in ('auto','cuda') and 'CUDAExecutionProvider' in available:
        providers=['CUDAExecutionProvider','CPUExecutionProvider']
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=providers)
    if device=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('CUDA initialization failed; check GPU runtime dependencies or use --device cpu')
    expected=json.dumps(FDI_ORDER)
    if session.get_modelmeta().custom_metadata_map.get('fdi_order')!=expected:
        raise ValueError('Model FDI channel-order metadata does not match this inference interface')
    return session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().with_name('toothmap_fdi.onnx'))
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('predictions'))
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--conf',type=float,default=.30)
    parser.add_argument('--min-sep',type=float,default=10)
    parser.add_argument('--no-refine',action='store_true')
    parser.add_argument('--no-tta',action='store_true')
    args=parser.parse_args()
    if not 0<=args.conf<=1 or args.min_sep<0: parser.error('Invalid confidence or min-sep')
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    paths=sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths: raise FileNotFoundError('No input images')
    if len({p.stem for p in paths})!=len(paths): raise ValueError('Input images must have unique filename stems')
    session=create_session(args.model,args.device)
    print('Providers:',session.get_providers(),flush=True)
    for folder in ('images_fdi','json_fdi','details'): (args.output/folder).mkdir(parents=True,exist_ok=True)
    for path in paths:
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(f'Cannot read {path}')
        predictions=predict(session,image,args.conf,not args.no_refine,not args.no_tta,args.min_sep)
        visual=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        shapes=[]
        for p in predictions:
            if not p['exist']: continue
            x,y=round(p['x']),round(p['y'])
            cv2.circle(visual,(x,y),4,(0,255,0),-1)
            cv2.putText(visual,str(p['fdi']),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA)
            shapes.append(dict(label=f"tooth_{p['fdi']:02d}",points=[[p['x'],p['y']]],group_id=None,shape_type='point',
                               flags=dict(fdi=p['fdi'],exist=True,confidence=round(p['confidence'],6))))
        labelme=dict(version='5.4.1',flags={},shapes=shapes,imagePath=path.name,imageData=None,
                     imageHeight=image.shape[0],imageWidth=image.shape[1])
        target=args.output/'images_fdi'/f'{path.stem}_fdi.png'
        if not cv2.imwrite(str(target),visual): raise OSError(f'Cannot write {target}')
        (args.output/'json_fdi'/f'{path.stem}_fdi.json').write_text(json.dumps(labelme,ensure_ascii=False,indent=2),encoding='utf-8')
        (args.output/'details'/f'{path.stem}.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{path.name}: {len(shapes)} teeth',flush=True)


if __name__=='__main__': main()
