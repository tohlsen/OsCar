import argparse
import os
import json

class FileArgumentParser(argparse.ArgumentParser):
    def __is_valid_file(self, parser, arg):
        if not os.path.isfile(arg):
            parser.error('The file {} does not exist!'.format(arg))
        elif arg.rsplit('.', 1)[1] != 'json':
            parser.error('The file {} is not a .json file!'.format(arg))
        elif 'drop' not in arg.rsplit('/', 1)[1]:
            parser.error('The file {} does not seem to be a drop_dataset file! We are looking for \'drop\' in the filename'.format(arg.rsplit('/',1)[1]))
        else:
            # File exists so return the filename
            return arg

    def add_argument_with_check(self, *args, **kwargs):
        # Look for your FILE or DIR settings
        if 'metavar' in kwargs and 'type' not in kwargs:
            if kwargs['metavar'] is 'FILE':
                type=lambda x: self.__is_valid_file(self, x)
                kwargs['type'] = type
        self.add_argument(*args, **kwargs)

def parseArgs():
    parser = FileArgumentParser(description="Partitions the given dataset file into span/no-span")
    parser.add_argument_with_check('filename', help='DROP Dataset file (.json)', metavar='FILE')
    args = parser.parse_args()
    return args.filename

def writePartition(dataset_filename, span):
    f = open(dataset_filename, 'r')
    jsons = json.load(f)

    deleted_items = []

    for field in jsons:
        item = jsons[field]
        qa_pairs = item['qa_pairs']
        numRemoved = 0
        for i in range(len(qa_pairs)):
            if (span):
                if (len(qa_pairs[i-numRemoved]['answer']['spans']) == 0):
                    del qa_pairs[i-numRemoved]
                    numRemoved+=1
            else:
                if (len(qa_pairs[i-numRemoved]['answer']['spans']) != 0):
                    del qa_pairs[i-numRemoved]
                    numRemoved+=1
        # If no more qa_pairs exist for this passage, remove
        # the whole passage from the dataset
        if (len(qa_pairs) == 0):
            deleted_items.append(field)
    
    for item in deleted_items:
        del jsons[item]
    
    # Create partitioned files
    dir_name = os.path.dirname(dataset_filename)
    span_name = ''
    if (span):
        span_name = '_span'
    else:
        span_name = '_no_span'
    out_filename = dataset_filename.rsplit('/',1)[-1].split('.',1)[0] + span_name + '.json'
    with open(os.path.join(dir_name, out_filename), 'w') as outfile:
        json.dump(jsons, outfile, indent=2)
        print('Created new file: ' + os.path.join(dir_name, out_filename))

def main():
    dataset_filename = parseArgs()
    writePartition(dataset_filename, True)
    writePartition(dataset_filename, False)

if __name__ == '__main__':
    main()