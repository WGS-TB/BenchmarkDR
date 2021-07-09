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
        '--model-name', dest='models', nargs='+',
        help='Name of model in config file', required=True
        )

    args = parser.parse_args()
    models = args.models
    print("________________")
    print(models)

    config_file = utils.config_reader(args.config)
    
    for model in models:
        current_module = utils.my_import(config_file['Models'][model]['module'])
        dClassifier = getattr(current_module, config_file['Models'][model]['model'])
        dClassifier = dClassifier(**config_file['Models'][model]['params'])

        print(dClassifier)
        filename = os.path.join("workflow/output/prediction/", model + ".joblib")
        dump(dClassifier, filename)

main()