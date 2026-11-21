import sys
from modelscope.hub.api import HubApi

api = HubApi()
for q in ["Pyramid-Flow", "STAR", "video super resolution", "VSR", "Real-ESRGAN", "BasicVSR"]:
    print("=== %s" % q, flush=True)
    try:
        r = api.list_models({'Name': q}, page_number=1, page_size=10)
    except Exception as e:
        print("  err", repr(e)[:150])
        continue
    models = (r or {}).get('Models') or (r or {}).get('models') or []
    if not models and isinstance(r, dict):
        print("  keys:", list(r.keys())[:8])
    for m in models[:10]:
        print("   ", m.get('Path', '?') + '/' + m.get('Name', '?'),
              '|', (m.get('Description') or '')[:70])
