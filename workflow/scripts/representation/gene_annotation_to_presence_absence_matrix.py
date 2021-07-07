
import pandas as pd
import numpy as np

file_list = snakemake.input["strains"]
reference = snakemake.input["reference"]
output_file = str(snakemake.output)

'''
Function to determine gene presence/ absence of strain in comparison to reference genome and return dataframe with
index = strain id, columns = genes, data = presence / absence as binary 
'''

def create_strain_df(strain_id, genes_reference, genes_strain):
    
    df_strain_present_genes = pd.DataFrame(columns = genes_strain, index = [strain_id], data = np.ones(len(genes_strain))[np.newaxis, :].astype(int))
    genes_reference_only = genes_reference.difference(genes_strain)
    
    df_genes_reference_only =pd.DataFrame(columns = genes_reference_only, index = [strain_id], data = np.zeros(len(genes_reference_only))[np.newaxis, :].astype(int))
    
    df_strain = pd.concat(
    [df_strain_present_genes, df_genes_reference_only], axis = 1)
    
    return df_strain


reference_data = pd.read_csv(reference, delimiter = "\t")
genes_reference = set(reference_data["gene"][np.logical_not(reference_data["gene"].isnull())])

## rows = strains, columns = genes

df_matrix = pd.DataFrame()

for file in file_list:
    
    strain_id = file.split("/")[-1].split(".")[0]
    
    ## reading genes from single strain and preparing to append to dataframe
    strain_data = pd.read_csv(file, delimiter = "\t")
    genes_strain = set(strain_data["gene"][np.logical_not(strain_data["gene"].isnull())])
    
    df_strain = create_strain_df(strain_id, genes_reference, genes_strain)
    
    df_matrix = df_matrix.append(df_strain)

df_matrix.fillna(value = 0, inplace = True)    

df_matrix.to_csv(output_file, index=False)
