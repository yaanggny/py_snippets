import os, sys
import multiprocessing
from multiprocessing import Process, Lock
from multiprocessing import Pool as processPool

def f(l, i):
    # l = Lock()
    l.acquire()
    try:
        print('hello world', i)
        print('*'*20)
    finally:
        l.release()
        pass

def writeLog(l, i, lxx):
    l.acquire()
    with open('log.txt', 'a') as fd:
        fd.write('PID= %d  i=%d:\n'%(os.getpid(), i))
        lst = [v for v in range(i)]
        lxx.append(lst)
        fd.write(str(lst))
        fd.write('\n\n')
    l.release()

def init(l):
    global lock
    lock = l    

if __name__ == '__main__':
    # lock = Lock()
    lock = multiprocessing.Manager().Lock()
    lxx = multiprocessing.Manager().list()

    # for num in range(50):
    #     Process(target=f, args=(lock, num)).start()
    args = []
    for i in range(50):
        args.append((lock, i, lxx))

    with processPool(5) as p:
    # with processPool(8, initializer=init, initargs=(lock,)) as p:
        p.starmap(writeLog, args)
    
    print(lxx)