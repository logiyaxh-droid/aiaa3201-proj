import sys

OLD = 'DATA = os.path.join(ROOT, "data", "realvsr")'
NEW = ('DATA = (os.environ.get("RVQ_DATA")\n'
       '        or os.path.join(ROOT, "data", "realvsr"))')

for p in sys.argv[1:]:
    s = open(p).read()
    if NEW in s:
        print("already patched", p)
        continue
    if OLD not in s:
        print("PATTERN MISSING", p)
        continue
    s = s.replace(OLD, NEW)
    open(p, "w").write(s)
    print("patched", p)
