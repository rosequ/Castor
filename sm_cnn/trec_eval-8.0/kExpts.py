import argparse


def create_files(fname, k, outFile):
    with open(fname) as f:
        for line in f:
            det = line.strip().split()
            rank = int(det[3])

            if rank <= int(k):
                outFile.write(line)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='stop at k rank', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('-k', default="10")
    ap.add_argument('-input', default="run.qa.sm.h200.k65000.txt")
    ap.add_argument('-output', default="sm")
    args = ap.parse_args()
    create_files(args.input, args.k, open("out.{}.k{}.txt".format(args.output, args.k), "w"))