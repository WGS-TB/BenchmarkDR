
import pandas as pd

file_list = snakemake.input
output_file = snakemake.output

matrix_df = pd.DataFrame()

for file in file_list:

    strain_id = file.split("/")[-1].split(".")[0]

    # reading kmers from single strain and preparing to append to dataframe
    strain_data = pd.read_csv(file, header=None, delimiter="\t")
    strain_data = strain_data.T
    strain_data.columns = strain_data.iloc[0]
    strain_data = strain_data[1:]
    strain_data.index = [strain_id]

    matrix_df = matrix_df.append(strain_data)

matrix_df.fillna(value=0, inplace=True)
matrix_df.reset_index(inplace=True)

matrix_df.to_csv(output_file)
