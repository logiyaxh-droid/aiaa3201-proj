from modelscope.hub.api import HubApi
api = HubApi()

cands = [
    "AI-ModelScope/pyramid-flow-sd3",
    "AI-ModelScope/pyramid-flow-miniflux",
    "NJU-PCALab/STAR",
    "AI-ModelScope/STAR",
    "AI-ModelScope/BasicVSR",
    "AI-ModelScope/Real-ESRGAN",
    "AI-ModelScope/stable-video-diffusion-img2vid-xt",
    "AI-ModelScope/AnimateDiff",
    "iic/cv_rrdb_image-super-resolution",
    "damo/cv_rrdb_image-super-resolution",
]
for c in cands:
    try:
        info = api.get_model(c)
        print("OK  ", c, "|", (info.get('Name') or '')[:50], flush=True)
    except Exception as e:
        print("MISS", c, repr(e)[:80], flush=True)
