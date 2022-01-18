import pandas as pd
import argparse
import os
import re

input_files = snakemake.input
output_file = str(snakemake.output)

def main():
    summary = pd.DataFrame()

    for file in input_files:
        df = pd.read_csv(file)
        summary = pd.concat([summary, df], axis=0)
    
    print(summary)
    print("Saving summary to ", output_file)
    summary.to_csv(output_file, index=False)

main()