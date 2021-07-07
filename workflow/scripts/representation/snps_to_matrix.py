
import pandas as pd
import numpy as np

file_list = snakemake.input
output_file = str(snakemake.output)

matrix_df = pd.DataFrame()

for file in file_list:

    strain_id = file.split("/")[-1].split(".")[0]

    # reading snps from single strain and preparing to append to dataframe
    strain_data = pd.read_csv(file, delimiter="\t")
    strain_data.index = strain_data["Chrom"] + "_" + strain_data["Position"].astype(str)
    strain_data[strain_id] =  np.ones(len(strain_data.index), dtype = np.int8)

    matrix_df = matrix_df.append(strain_data[strain_id].T)

matrix_df.fillna(value=0, inplace=True)

matrix_df.to_csv(output_file, index=False)
