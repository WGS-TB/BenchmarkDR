from pathlib import Path

configfile: "config.yaml"

##################################
# Import other Snakemake Modules #
##################################

include: "workflow/rules/representation/snakefile"


########################
# One to rule them all #
########################

rule all:
   input:  expand(Path(config["PATH_DATA"] + "/{bacterium}/representation/{representation}/0_matrix.csv"), bacterium = config["BACTERIA"], representation = config["REPRESENTATION"])