# Helper script to count extracted numbers from the datasets

import json
import string
import argparse
import os
import glob
import csv
from allennlp.data.dataset_readers.reading_comprehension.util import (IGNORED_TOKENS,
                                                                       STRIPPED_CHARACTERS,
                                                                       make_reading_comprehension_instance,
                                                                       split_tokens_by_hyphen)
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from word2number.w2n import word_to_num # need to 'pip install word2number' to use

PERCENT_INCREMENT = 10

class FileArgumentParser(argparse.ArgumentParser):
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

    def add_argument_with_check(self, *args, **kwargs):
        # Look for your FILE or DIR settings
        if 'metavar' in kwargs and 'type' not in kwargs:
            if kwargs['metavar'] is 'JSON':
                type=lambda x: self.__is_valid_json_data(self, x)
                kwargs['type'] = type
        self.add_argument(*args, **kwargs)

# returning count of numbers (old DatasetReader)
def old_reader_get_count(word):
    WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                    "five": 5, "six": 6, "seven": 7, "eight": 8,
                    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                    "thirteen": 13, "fourteen": 14, "fifteen": 15,
                    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

    no_comma_word = word.replace(",", "")
    number = None
    if no_comma_word in WORD_NUMBER_MAP:
        return 1
    else:
        try:
            number = int(no_comma_word)
        except ValueError:
            return 0
    if number is not None:
        return 1
    else:
        return 0

# returning count of numbers (new DatasetReader v1)
def new_reader_v1_get_count(word):
    # strip all punctuations from the sides of the word, except for the negative sign
    punctruations = string.punctuation.replace('-', '')
    word = word.strip(punctruations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    number = None
    # word2num will convert hundred, thousand ... to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion"]:
        number = None
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                number = None

    if number is not None:
        return 1
    else:
        return 0

# returning count of numbers (new DatasetReader v2)
def new_reader_v2_get_count(word):
    # strip all punctuations from the sides of the word, except for the negative sign
    punctuations = string.punctuation.replace('-', '')
    word = word.strip(punctuations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    numbers = []
    # word2num will convert hundred, thousand ... and point to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion", "point"]:
        return numbers, len(numbers)
    try:
        numbers += [word_to_num(word)]
    except ValueError:
        try:
            numbers += [int(word)]
        except ValueError:
            try:
                numbers += [float(word)]
            except ValueError:
                if "-" in word:
                    word_chunks = word.split("-")
                    for chunk in word_chunks:
                        converted_nums, _ = new_reader_v2_get_count(chunk)
                        numbers += converted_nums
    return numbers, len(numbers)

# parses the necessary arguments
def parseArgs():
    parser = FileArgumentParser(description="Analyzes the given drop dataset file")
    parser.add_argument_with_check('dataset', help='DROP dataset file (.json)', metavar='JSON')
    parser.add_argument('passage_limit', help='Passage limit size')
    parser.add_argument('question_limit', help='Question limit size')
    parser.add_argument('passage_bucket_size', help='Size of passage token buckets for histogram, default = 10', nargs='?', default=10)
    parser.add_argument('question_bucket_size', help='Size of question token buckets for histogram, default = 5', nargs='?', default=5)
    args = parser.parse_args()
    return args.dataset, args.passage_bucket_size, args.question_bucket_size, int(args.passage_limit), int(args.question_limit)

def percentIncrease(old, new):
    return (new - old) / old * 100

def numberExtractionResults(old_reader_count, new_reader_v1_count, new_reader_v2_count):
    percent_increase_v1 = percentIncrease(old_reader_count, new_reader_v1_count)
    percent_increase_v2 = percentIncrease(old_reader_count, new_reader_v2_count)

    print('Number Extraction Analysis')
    print('\tOld DatasetReader Number Count: {:d}'.format(old_reader_count))
    print('\tNew DatasetReader v1 Number Count: {:d}'.format(new_reader_v1_count))
    print('\tNew DatasetReader v1 Percent Increase: {:.2f}%'.format(percent_increase_v1))
    print('\tNew DatasetReader v2 Number Count: {:d}'.format(new_reader_v2_count))
    print('\tNew DatasetReader v2 Percent Increase: {:.2f}%'.format(percent_increase_v2))

def passageAnalysisResults(dataset_filename, num_passages, num_passage_tokens, passage_buckets, passage_bucket_size, passage_limit):
    print('\nPassage Analysis')
    avg_passage_length = num_passage_tokens / num_passages
    print('\tNumber of Passages: {:d}'.format(num_passages))
    print('\tAvg Passage Length: {:.2f}'.format(avg_passage_length))
    num_passages_under_limit = 0
    num_passage_over_limit = 0
    for bucket, bucket_amount in sorted(passage_buckets.items()):
        if (bucket+passage_bucket_size-1 <= passage_limit):
            num_passages_under_limit += bucket_amount
        else:
            num_passage_over_limit += bucket_amount
    percent = num_passage_over_limit / num_passages * 100
    print('\tNumber of Passages Under {:d} Tokens: {:d}'.format(passage_limit, num_passages_under_limit))
    print('\tNumber of Passages Over {:d} Tokens: {:d}'.format(passage_limit, num_passage_over_limit))
    print('\tPercentage of Passages Over Passage Limit {:d}: {:.2f}%'.format(passage_limit, percent))

    output_file = 'drop_analysis/' + dataset_filename + '_passage_histogram.tsv'
    writeHistogramToTSV(output_file, passage_buckets, passage_bucket_size)

def questionAnalysisResults(dataset_filename, num_questions, num_question_tokens, question_buckets, question_bucket_size, question_limit):
    print('\nQuestion Analysis')
    avg_question_length = num_question_tokens / num_questions
    print('\tNumber of Questions: {:d}'.format(num_questions))
    print('\tAvg Question Length: {:.2f}'.format(avg_question_length))
    num_questions_under_limit = 0
    num_questions_over_limit = 0
    for bucket, bucket_amount in sorted(question_buckets.items()):
        if (bucket+question_bucket_size-1 <= question_limit):
            num_questions_under_limit += bucket_amount
        else:
            num_questions_over_limit += bucket_amount
    percent = num_questions_over_limit / num_questions * 100
    print('\tNumber of Questions Under {:d} Tokens: {:d}'.format(question_limit, num_questions_under_limit))
    print('\tNumber of Questions Over {:d} Tokens: {:d}'.format(question_limit, num_questions_over_limit))
    print('\tPercentage of Questions Over Question Limit {:d}: {:.2f}%'.format(question_limit, percent))

    output_file = 'drop_analysis/' + dataset_filename + '_question_histogram.tsv'
    writeHistogramToTSV(output_file, question_buckets, question_bucket_size)

def writeHistogramToTSV(filename, hist, bucket_size):
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, 'w') as hist_file:
        writer = csv.writer(hist_file, delimiter='\t')
        writer.writerow(['Bucket', 'Amount'])
        for bucket, amount in sorted(hist.items()):
            bucket = '{:d}-{:d}'.format(bucket, bucket+bucket_size-1)
            writer.writerow([bucket, amount])
        print('\tWrote Histogram: ' + filename)

def analyzeDataset(filePath, passage_bucket_size, question_bucket_size, passage_limit, question_limit):
    tokenizer = WordTokenizer()

    num_passages = 0
    num_passage_tokens = 0
    passage_buckets = {}
    num_questions = 0
    num_question_tokens = 0
    question_buckets = {}

    old_reader_count = 0
    new_reader_v1_count = 0
    new_reader_v2_count = 0

    num_passages_analyzed = 0
    maxPercent = 0
    with open(filePath) as file_:
        train_dataset = json.load(file_)

        # iterate through all the passages, counting all extracted numbers
        train_items = train_dataset.items()
        num_passages = len(train_items)
        for passage_id, passage_info in train_items:
            # Track progress and notify user/client
            num_passages_analyzed += 1
            percent_analyzed = float(num_passages_analyzed) / num_passages * 100.0
            if (int(percent_analyzed) // PERCENT_INCREMENT > maxPercent):
                maxPercent = int(percent_analyzed) / PERCENT_INCREMENT
                print("{:3d}% of passages analyzed".format(int(percent_analyzed)))
            
            # Analyze passages
            passage_text = passage_info["passage"]
            passage_words = passage_text.split(" ")

            passage_tokens = tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            num_tokens_in_passage = len(passage_tokens)
            num_passage_tokens += num_tokens_in_passage
            passage_bucket = num_tokens_in_passage // passage_bucket_size * passage_bucket_size
            if passage_bucket not in passage_buckets:
                passage_buckets[passage_bucket] = 0
            passage_buckets[passage_bucket] += 1

            # get all number counts from each passage
            for passage_word in passage_words:
                new_reader_v1_count += new_reader_v1_get_count(passage_word)
                _, temp_count = new_reader_v2_get_count(passage_word)
                new_reader_v2_count += temp_count
                old_reader_count += old_reader_get_count(passage_word)

            # for each passage, iterate through all the questions
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                question_words = question_text.split(" ")
                question_tokens = tokenizer.tokenize(question_text)
                question_tokens = split_tokens_by_hyphen(question_tokens)
                num_questions += 1
                num_tokens_in_question = len(question_tokens)
                num_question_tokens += num_tokens_in_question
                question_bucket = num_tokens_in_question // question_bucket_size * question_bucket_size
                if question_bucket not in question_buckets:
                    question_buckets[question_bucket] = 0
                question_buckets[question_bucket] += 1

                # get all number counts from each question
                for question_word in question_text:
                    new_reader_v1_count += new_reader_v1_get_count(question_word)
                    _, temp_count = new_reader_v2_get_count(question_word)
                    new_reader_v2_count += temp_count
                    old_reader_count += old_reader_get_count(question_word)

    # print out results
    dataset_filename = os.path.basename(filePath).rsplit('.', 1)[-2]
    print("\nAnalysis Complete!")
    numberExtractionResults(old_reader_count, new_reader_v1_count, new_reader_v2_count)
    passageAnalysisResults(dataset_filename, num_passages, num_passage_tokens, passage_buckets, passage_bucket_size, passage_limit)
    questionAnalysisResults(dataset_filename, num_questions, num_question_tokens, question_buckets, question_bucket_size, question_limit)

def main():
    dataset_file_path, passage_bucket_size, question_bucket_size, passage_limit, question_limit = parseArgs()
    print("Starting analysis on " + dataset_file_path + " now...")
    analyzeDataset(dataset_file_path, passage_bucket_size, question_bucket_size, passage_limit, question_limit)

if __name__ == '__main__':
    main()
