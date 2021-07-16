import pandas as pd
import argparse
import os
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', dest='datafiles', nargs='+',
        help='Path to the data files', required=True,
    )
    parser.add_argument(
        '--outfile', dest='outfile', metavar='FILE',
        help='Path to the output file', required=True,
    )
    args = parser.parse_args()

    summary = pd.DataFrame()

    for file in args.datafiles:
        df = pd.read_csv(file)
        df = df[["Drug", "balanced_accuracy"]]

        head, tail = os.path.split(file)
        model = re.sub("(-.*)?\.csv", "", tail)
        df.insert(0, "Model", model)

        summary = pd.concat([summary, df], axis=0)
    
    print(summary)
    print("Saving summary to ", args.outfile)
    summary.to_csv(args.outfile)

main()