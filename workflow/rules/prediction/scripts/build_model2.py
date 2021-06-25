from joblib import dump
import argparse
import sys
import os
import utils
from support import saveObject

def main(sysargs=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--config', dest='config', metavar='FILE',
            help='Path to the config file', required=True,
        )
    parser.add_argument(
        '--model-name', dest='models', nargs='+',
        help='Name of model in config file', required=True
    )
    args = parser.parse_args()
    models = args.models
    print("________________")
    print(models)

    for model in models:
        config_file = utils.config_reader(args.config)
        current_module = utils.my_import(config_file['Models'][model]['module'])
        dClassifier = getattr(current_module, config_file['Models'][model]['model'])
        dClassifier = dClassifier(**config_file['Models'][model]['params'])

        print(dClassifier)
        filename = os.path.join("ml", model + ".pkl")
        saveObject(dClassifier, filename)

main()