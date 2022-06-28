configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################


include: "workflow/rules/representation/snakefile"
include: "workflow/rules/prediction/snakefile"


########################
# One to rule them all #
########################


rule all:
    input:
        expand(
            config["OUTPUT_DIR"] + "/{bacterium}/prediction/summary.csv",
            bacterium=config["BACTERIA"],
        )
