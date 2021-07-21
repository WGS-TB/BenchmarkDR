from joblib import dump
import argparse
import sys
import os
import utils

def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Building model")
    print("_______________________________")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', dest='config', metavar='FILE',
        help='Path to the config file', required=True,
    )
    parser.add_argument(
        '--model-name', dest='model',
        help='Name of model in config file', required=True
    )
    parser.add_argument(
        '--outfile', dest='outfile',
        help='Name of output file', required=True
    )

    args = parser.parse_args()
    model = args.model
    print("________________")
    print(model)

    config_file = utils.config_reader(args.config)
    
    current_module = utils.my_import(config_file['Models'][model]['module'])
    dClassifier = getattr(current_module, config_file['Models'][model]['model'])
    dClassifier = dClassifier(**config_file['Models'][model]['params'])

    print(dClassifier)
    dump(dClassifier, args.outfile)

main()