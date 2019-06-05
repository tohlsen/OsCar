import subprocess
import argparse
import os
import json
import _jsonnet

class FileArgumentParser(argparse.ArgumentParser):
    def __is_valid_jsonnet(self, parser, arg):
        output_model_dir = 'out/' + arg.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '/model.tar.gz'
        if not os.path.isfile(arg):
            parser.error('The file {} does not exist!'.format(arg))
        elif arg.rsplit('.', 1)[1] != 'jsonnet':
            parser.error('The file {} is not a .jsonnet file!'.format(arg))
        elif not os.path.isfile(output_model_dir):
            parser.error('The model {} does not exist!'.format(output_model_dir))
        else:
            # File exists so return the filename
            return arg

    def __is_valid_json_data(self, parser, arg):
        if not os.path.isfile(arg):
            parser.error('The file {} does not exist!'.format(arg))
        elif arg.rsplit('.', 1)[1] != 'json':
            parser.error('The file {} is not a .json file!'.format(arg))
        elif 'drop' not in arg.rsplit('/', 1)[1]:
            parser.error('The file {} does not seem to be a drop_dataset file! We are looking for \'drop\' in the filename'.format(arg.rsplit('/',1)[1]))
        else:
            # File exists so return the filename
            return arg
   
    def __is_valid_txt_data(self, parser, arg):
       	if not os.path.isfile(arg):
            parser.error('The file [] does not exist!'.format(arg)) 
        elif arg.rsplit('.', 1)[1] != 'txt':
            parser.error('The file [] is not a .txt file!'.format(arg))
        else:
            return arg

    def __is_valid_directory(self, parser, arg):
        if not os.path.isdir(arg):
            parser.error('The directory {} does not exist!'.format(arg))
        if os.path.abspath(arg).split('/')[-1] == 'out':
            parser.error('Specify a specific output folder, not the generic out/ folder.')
        else:
            # File exists so return the directory
            return arg

    def add_argument_with_check(self, *args, **kwargs):
        # Look for your FILE or DIR settings
        if 'metavar' in kwargs and 'type' not in kwargs:
            if kwargs['metavar'] is 'JSONNET':
                type=lambda x: self.__is_valid_jsonnet(self, x)
                kwargs['type'] = type
            if kwargs['metavar'] is 'JSON':
                type=lambda x: self.__is_valid_json_data(self, x)
                kwargs['type'] = type
            if kwargs['metavar'] is 'DIR':
                type=lambda x: self.__is_valid_directory(self, x)
                kwargs['type'] = type
            if kwargs['metavar'] is 'TXT':
                type=lambda x: self.__is_valid_txt_data(self, x) 
                kwargs['type'] = type
        self.add_argument(*args, **kwargs)

def parseArgs():
    parser = FileArgumentParser(description="Evaluates a model on a dataset (creates an evaluate.sh file as reference)")
    parser.add_argument_with_check('filename', help='DROP config file (.jsonnet)', metavar='JSONNET')
    parser.add_argument_with_check('dataset', help='QA dataset file (.txt)', metavar='TXT')
    args = parser.parse_args()
    return args.filename, args.dataset

def get_dataset_reader(config_filename):
    config_jsonnet = ''
    with open(config_filename, 'r') as f:
        config_jsonnet = f.read()
    config_json_str = _jsonnet.evaluate_snippet("snippet", config_jsonnet)
    config_json = json.loads(config_json_str)
    dataset_reader_value = json.dumps(config_json['validation_dataset_reader'])
    dataset_reader_json = '{"dataset_reader": ' + dataset_reader_value + '}'
    return dataset_reader_json

def write_evaluate_script(dataset_reader, output_model, dataset):
    evaluate_filename = 'scripts/predict_txt.sh'

    with open(evaluate_filename, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('allennlp predict ' + output_model + ' ' + dataset + ' --predictor naqanet --include-package drop_library --cuda-device 0 -o \'' + dataset_reader + '\'\n')
    print('Running \'./scripts/evaluate.sh\' ' + output_model + '...')

def main():
    config_filename, dataset = parseArgs()
    dr_json = get_dataset_reader(config_filename)
    output_model = 'out/' + config_filename.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '/model.tar.gz'
    write_evaluate_script(dr_json, output_model, dataset)
    # Run the evaluation
    subprocess.call(['./scripts/predict_txt.sh'])

if __name__=='__main__':
    main()
