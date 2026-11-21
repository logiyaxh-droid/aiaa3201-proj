from modelscope.hub.api import HubApi
api = HubApi()

for repo in ["AI-ModelScope/pyramid-flow-sd3", "AI-ModelScope/pyramid-flow-miniflux"]:
    print("=" * 20, repo, flush=True)
    try:
        files = api.get_model_files(repo, recursive=True)
    except Exception as e:
        print("  err", repr(e)[:150], flush=True)
        continue
    tot = 0
    for f in files:
        if f.get('Type') == 'tree':
            continue
        sz = f.get('Size') or 0
        tot += sz
        if sz > 1e6 or f['Path'].endswith('.json'):
            print("   %-62s %8.2f GB" % (f['Path'], sz / 1e9), flush=True)
    print("   TOTAL %.2f GB" % (tot / 1e9), flush=True)
