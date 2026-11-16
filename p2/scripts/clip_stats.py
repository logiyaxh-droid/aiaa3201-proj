import os, glob, json, numpy as np, cv2
B="/data/disk1/shared/aiaa3201/project1_video_sr"
GT=B+"/val_sharp"
out={}
for clip in sorted(os.listdir(GT)):
    fs=sorted(glob.glob(os.path.join(GT,clip,"*.png")))
    if len(fs)<10: continue
    idx=np.linspace(0,len(fs)-1,6).astype(int)
    lap=[]; diff=[]; prev=None
    for i in idx:
        g=cv2.imread(fs[i],0)
        lap.append(float(cv2.Laplacian(g,cv2.CV_32F).var()))
        if prev is not None: diff.append(float(np.mean(np.abs(g.astype(np.float32)-prev))))
        prev=g.astype(np.float32)
    out[clip]=dict(sharp=float(np.mean(lap)), motion=float(np.mean(diff)), n=len(fs))
json.dump(out,open("/tmp/p2/clip_stats.json","w"),indent=1)
print("ok", len(out))
