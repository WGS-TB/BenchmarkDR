# BenchmarkDR

BenchmarkDR is a modular end-to-end pipeline to train and evaluate machine learning prediction models on genomic short-read sequencing data. The predicted phenotype can be either a binary classification (e.g. susceptible to a drug/ resistant) or a regression (e.g. minimum inhibitory concentration (MIC)).

## Installation

Clone this repo and set up an environment containing [Snakemake](https://snakemake.readthedocs.io/en/stable/) 

```bash
git clone https://github.com/WGS-TB/BenchmarkDR.git
conda activate base
mamba create -c conda-forge -c bioconda -n snakemake snakemake
```
## Usage

To run the pipeline on your own data, you have to structure your input data (bacterial read and labels) in following way:
<pre>
your_bacterium_1  
|  
|---- data (place paired-end short reads here)   
|   
|---- reference   
|         |                          
|         |                          
|        reference_for_your_bacterium_1.fasta   
|    
|---- AllLabels.csv    


To adapt configuration of the pipeline, there are different config files. It is necessary to adapt the config in the main folder:

|-- workflow    
|      |--rules    
|      |    |-- prediction    
|      |             |-- models_config     
|      |             |-- ...
|      |-- ...      
|    
|-- config          
|-- Snakefile    
</pre>

```python

PATH_DATA: "path to folder containing your bacterial folders"

BACTERIA: ["your_bacterium_1"]

OUTPUT_DIR:  "path to folder where you would like to save all files created by the pipeline"

REPRESENTATION: ["gene_presence_absence", "snps"] # "kmer"

MODE: "MIC" # "Classification"

DRUGS: ["drug_1", ...] Column names of in your AllLabels.csv file

METHODS: ["sklearn_LinR", "sklearn_LinR_l1", "sklearn_LinR_l2", "sklearn_LinR_elasticnet", "sklearn_SVMR", "sklearn_GBTR", "sklearn_RFR"] #Classification ["sklearn_LR_l1", "sklearn_LR_l2", "sklearn_LR_elasticnet", "sklearn_SGD_l1", "sklearn_SGD_l2", "sklearn_SGD_elasticnet", "sklearn_DT", "sklearn_RFC", "sklearn_ET", "sklearn_ADB", "sklearn_GBTC", "sklearn_GNB", "sklearn_CNB", "sklearn_SVM_l1", "sklearn_SVM_l2", "sklearn_KNN", "INGOT"]

OPTIMIZATION: "None" # 'GridSearchCV', 'RandomizedSearchCV' and "None"
```

```bash
snakemake -s Snakefile --use-conda
```



