# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to data")
    parser.add_argument("--max_len", type=int, default=512,
                        help="max input length")
    parser.add_argument("--append_offset", type=int, default=4,
                        help="offset for appending/prepending bos eos, etc")

    args = parser.parse_args() 
    L = args.max_len - args.append_offset
    for fs in glob.glob(os.path.join(args.path, "*.src")):
        print(fs)
        wf = open(fs + ".truncated", 'w')
        with open(fs, 'r') as rf:
            for l in rf:
                x = " ".join(l.strip().split()[:L])
                wf.write(x + '\n')
        wf.close()
        os.system("mv {}.truncated {}".format(fs, fs))
