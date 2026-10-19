"""Follow every artifacts/logs/*.log (including files created later) and print lines matching progress/failure."""
import glob, os, re, sys, time

PAT = re.compile(r'^step |supervise|phase2|Traceback|Error|Killed|non-finite|FT_EVAL_DONE|resumed|AssertionError')
pos = {}
key = lambda f: (f, os.stat(f).st_ino)  # inode-aware: a renamed/recreated log starts from 0
for f in glob.glob('artifacts/logs/*.log'):
    pos[key(f)] = os.path.getsize(f)  # only new lines
while True:
    for f in glob.glob('artifacts/logs/*.log'):
        try:
            k = key(f)
        except FileNotFoundError:
            continue
        start = pos.get(k, 0)
        size = os.path.getsize(f)
        if size > start:
            with open(f, errors='replace') as fh:
                fh.seek(start)
                for line in fh.read(size - start).splitlines():
                    if PAT.search(line):
                        print(f'{os.path.basename(f)[:-4]}: {line[:220]}', flush=True)
            pos[k] = size
    time.sleep(3)
