import os
import sys

path='./'+sys.argv[1]
files = os.listdir(path)
for i, brand in enumerate(files):
    f = open(path+'/'+brand)
    lines = f.readlines()
    for ix, line in enumerate(lines):
        sep_line = line.strip().split(' ')
        try:
           # print(sep_line)
           # print(path+'/'+brand)
            x1 = float(sep_line[0])
            y1 = float(sep_line[1])

            x2 = float(sep_line[0])+float(sep_line[2])
            y2 = float(sep_line[1])+float(sep_line[3])
            cls = int(sep_line[4])
        except Exception:
            print('Error: Type')
            print(path+'/'+brand)
            print(x1)
            print(y1)
            print(x2)
            print(y2)
            print(cls)
            exit(1)
        else:
            if (x1 >= x2) or ( y1 >= y2) or (x1 <= 0) or (x2 <= 0) or (y1 <= 0) or (y2 <= 0):
                print('value error')
                print(path+'/'+brand)
                exit(1)
            if (cls < 1 or cls > 27):
                print('value error')
                print(path+'/'+brand)
                print('cls: {}'.format(cls))
                exit(1) 
    f.close()
