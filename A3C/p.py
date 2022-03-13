import multiprocessing as mp
import time
import os

def a(q):
    
    for i in range(100):
        if not q.empty():
            print(os.getpid())
            aa = q.get()
            print(aa)
            aa += 1
            q.put(aa)
            time.sleep(0.5)

if __name__ == '__main__':
    q = mp.Queue()

    ps = []
    q.put(1)
    for i in range(2):
        
        p = mp.Process(target=a, args=(q, ))
        ps.append(p)

    for qq in ps:
        qq.start()

    for qq in ps:
        qq.join()