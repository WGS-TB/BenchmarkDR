import pandas as pd
import numpy as np

file_list = snakemake.input
output_file = str(snakemake.output)

dict_dfs = {}

for file in file_list:

    strain_id = file.split("/")[-1].split(".")[0]

    # reading snps from single strain and preparing to append to dataframe
    strain_data = pd.read_csv(file, delimiter="\t")

    # filter out SNPs detected as heterozygous, since likely wrong for haploid bacterial genome
    strain_data = strain_data.loc[strain_data["SamplesHom"] == 1]

    strain_data.index = strain_data["Chrom"] + "_" + strain_data["Position"].astype(str)
    strain_data[strain_id] = np.ones(len(strain_data.index), dtype=np.int8)

    dict_dfs[strain_id] = df_strain

df_matrix = pd.concat([df for df in dict_dfs.values()], axis=1).T
matrix_df.fillna(value=0, inplace=True)

matrix_df.to_csv(output_file)
