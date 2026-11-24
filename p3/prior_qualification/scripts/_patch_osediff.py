import sys

P = "scripts/run_osediff_realvsr.py"
s = open(P).read()

OLD_PATH = 'DATA = os.path.join(ROOT, "data", "realvsr")'
NEW_PATH = ('DATA = (os.environ.get("RVQ_DATA")\n'
            '        or os.path.join(ROOT, "data", "realvsr"))')

OLD_IMP = "import cv2\nimport numpy as np\nimport torch"
NEW_IMP = ('# OSEDiff was written against diffusers 0.25 and uses APIs that 0.32\n'
           '# removed (e.g. FromOriginalVAEMixin).  DOVE needed 0.32 for CogVideoX,\n'
           '# so keep the old wheel in site_old/ and put it first on sys.path here\n'
           '# instead of downgrading the shared environment.\n'
           '_OLD = os.path.join(ROOT, "site_old")\n'
           'if os.path.isdir(os.path.join(_OLD, "diffusers")):\n'
           '    sys.path.insert(0, _OLD)\n'
           '\n'
           'import cv2\nimport numpy as np\nimport torch')

if NEW_PATH in s:
    print("path already patched")
else:
    s = s.replace(OLD_PATH, NEW_PATH)
    print("path patched")

if "site_old" in s:
    print("sys.path shim already present")
else:
    s = s.replace(OLD_IMP, NEW_IMP)
    print("shim inserted")

open(P, "w").write(s)
