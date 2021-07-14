import pandas as pd
import argparse
import os

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
        df.set_index(df.columns[0], inplace=True, drop=True)
        df = df[["balanced_accuracy"]]

        head, tail = os.path.split(file)
        model = tail.replace(".csv", "")
        df = df.rename(columns={"balanced_accuracy": model})

        summary = pd.concat([summary,df], axis=1)
    
    print(summary)
    print("Saving summary to ", args.outfile)
    summary.to_csv(args.outfile)

main()