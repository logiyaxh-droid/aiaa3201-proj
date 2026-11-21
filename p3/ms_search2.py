from modelscope.hub.api import HubApi
api = HubApi()

print("--- raw list_models", flush=True)
r = api.list_models({'Name': 'Pyramid'}, page_number=1, page_size=5)
print(r, flush=True)

print("--- known model probes", flush=True)
cands = [
    "Rain1017/Pyramid-Flow-v2",
    "Rain1017/pyramid-flow-sd3",
    "iic/cv_rrdb_image-super-resolution",
    "AI-ModelScope/Real-ESRGAN",
    "damo/cv_rrdb_image-super-resolution_x2",
    "wutianyi/Pyramid-Flow",
]
for c in cands:
    try:
        info = api.get_model(c)
        print("OK  ", c, '->', (info.get('Name') or '')[:40], flush=True)
    except Exception as e:
        print("MISS", c, repr(e)[:100], flush=True)
