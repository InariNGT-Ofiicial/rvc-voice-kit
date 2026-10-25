# -*- coding: utf-8 -*-
"""多线程分段下载 hf-mirror 大文件（单连接被限速时用）。"""
import sys
import os
import concurrent.futures as cf
import requests

URL = sys.argv[1]
OUT = sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TOTAL = int(sys.argv[4]) if len(sys.argv) > 4 else 0

hdr = {"User-Agent": "curl/8.0"}
sess = requests.Session()
sess.trust_env = False  # 直连，不走代理

if not TOTAL:
    r = sess.head(URL, headers=hdr, allow_redirects=True, timeout=60)
    TOTAL = int(r.headers["Content-Length"])
    print(f"total {TOTAL/1048576:.1f} MB", flush=True)

parts = []
step = TOTAL // THREADS


def fetch(i):
    a = i * step
    b = TOTAL - 1 if i == THREADS - 1 else (i + 1) * step - 1
    tmp = f"{OUT}.part{i}"
    if os.path.exists(tmp) and os.path.getsize(tmp) >= b - a + 1:
        return tmp
    got = 0
    for attempt in range(5):
        try:
            h = dict(hdr)
            if got:
                h["Range"] = f"bytes={a+got}-{b}"
            else:
                h["Range"] = f"bytes={a}-{b}"
            with sess.get(URL, headers=h, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "ab" if got else "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                        got += len(chunk)
            if os.path.getsize(tmp) >= b - a + 1:
                return tmp
        except Exception as e:
            print(f"  part{i} retry{attempt}: {e}", flush=True)
    return tmp


with cf.ThreadPoolExecutor(THREADS) as ex:
    futs = [ex.submit(fetch, i) for i in range(THREADS)]
    done = 0
    for f in cf.as_completed(futs):
        done += 1
        print(f"  {done}/{THREADS} 完成", flush=True)

with open(OUT, "wb") as out:
    for i in range(THREADS):
        p = f"{OUT}.part{i}"
        with open(p, "rb") as f:
            out.write(f.read())
        os.remove(p)

print(f"OK {OUT} {os.path.getsize(OUT)/1048576:.1f} MB", flush=True)
