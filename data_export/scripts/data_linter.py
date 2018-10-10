from os import listdir
from os.path import isfile, join
import re
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',action='store',default='text',type=str)
    args = parser.parse_args(sys.argv[1:])

    mypath = args.path

    files = [join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath, f)) and 'static' in f)]

    lines = 0
    nonempty = 0
    valid = 0

    for file in files:
        print file
        with open(file, 'r') as f:
            for line in f:
                lines += 1
                code = re.search('\'([0-9]+[0-9,]+)\'', line)
                if code != None:
                    codeline = code.group(1)
                    nonempty += 1
                    ok = True
                    for token in codeline.split(','):
                        if token != '' and int(token) > 2000:
                            ok = False
                            break
                    if ok:
                        valid += 1

    print lines, nonempty, valid



