import os, sys, mmap
for d in sys.argv[1:]:
    p = os.path.join(d, ".odirect_test")
    try:
        fd = os.open(p, os.O_CREAT|os.O_WRONLY|os.O_TRUNC, 0o644)
        os.write(fd, b"x"*1048576); os.close(fd)
        fd = os.open(p, os.O_RDONLY|os.O_DIRECT)
        buf = mmap.mmap(-1, 1048576)
        n = os.readv(fd, [buf]); os.close(fd)
        print(d, "O_DIRECT read OK bytes", n)
    except OSError as e:
        print(d, "O_DIRECT FAILED:", e)
    finally:
        try: os.remove(p)
        except: pass
