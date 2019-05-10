import argparse
import json
import os
import glob
import csv

class DirArgumentParser(argparse.ArgumentParser):
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
            if kwargs['metavar'] is 'DIR':
                type=lambda x: self.__is_valid_directory(self, x)
                kwargs['type'] = type
        self.add_argument(*args, **kwargs)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def readMetrics(dir_path):
    metric_jsons = []

    for filename in glob.glob(os.path.join(dir_path, 'metrics_epoch_*.json')):
        f = open(filename, 'r')
        metric_jsons.append(json.load(f))
    
    return metric_jsons

def writeTSV(metrics, out_name):
    output_dir = 'metrics_tsv/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + out_name + '.tsv', 'w') as metrics_file:
        writer = csv.writer(metrics_file, delimiter='\t')
        writer.writerow(['Epoch', 'Train EM', 'Train F1', 'Dev EM', 'Dev F1'])
        for cur_metric in metrics:
            epoch_num = str(cur_metric['epoch'])
            train_em = str(cur_metric['training_em']*100)
            train_f1 = str(cur_metric['training_f1']*100)
            dev_em = str(cur_metric['validation_em']*100)
            dev_f1 = str(cur_metric['validation_f1']*100)
            writer.writerow([epoch_num, train_em, train_f1, dev_em, dev_f1])
    print(bcolors.OKGREEN + 'Wrote to: ' + output_dir + out_name + '.tsv' + bcolors.ENDC)


def main():
    parser = DirArgumentParser(description="Parses output directory and creates a .csv with training/dev accuracies.")
    parser.add_argument_with_check('output_dir', help='Output Directory with Results', metavar='DIR')
    args = parser.parse_args()
    
    metric_jsons = readMetrics(args.output_dir)

    out_name = args.output_dir.split('/')[-1]
    if out_name == '':
        out_name = args.output_dir.split('/')[-2]

    writeTSV(metric_jsons, out_name)


if __name__ == '__main__':
    main()
