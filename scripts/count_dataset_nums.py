# Helper script to count extracted numbers from the datasets

import json
import string
import numpy as np
from word2number.w2n import word_to_num # need to 'pip install word2number' to use
from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# returning a number if found (old DatasetReader)
def old_reader_get_numbers(word):
    orig_word = word
    WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                    "five": 5, "six": 6, "seven": 7, "eight": 8,
                    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                    "thirteen": 13, "fourteen": 14, "fifteen": 15,
                    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

    no_comma_word = word.replace(",", "")
    number = None
    if no_comma_word in WORD_NUMBER_MAP:
        # number = WORD_NUMBER_MAP[no_comma_word]
        number = orig_word
    else:
        try:
            temp_number = int(no_comma_word)
            number = orig_word
        except ValueError:
            return None
        return number

# returning numbers if found (new DatasetReader v1)
def new_reader_v1_get_numbers(word):
    orig_word = word
    # strip all punctuations from the sides of the word, except for the negative sign
    punctuations = string.punctuation.replace('-', '')
    word = word.strip(punctuations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    # word2num will convert hundred, thousand ... to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion"]:
        return None
    try:
        temp_number = word_to_num(word)
        number = orig_word
    except ValueError:
        try:
            temp_number = int(word)
            number = orig_word
        except ValueError:
            try:
                temp_number = float(word)
                number = orig_word
            except ValueError:
                number = None
    return number

# returning numbers if found (new DatasetReader v2)
def new_reader_v2_get_numbers(word):
    orig_word = word
    # strip all punctuations from the sides of the word, except for the negative sign
    punctuations = string.punctuation.replace('-', '')
    word = word.strip(punctuations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    numbers = set()
    # word2num will convert hundred, thousand ... and point to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion"]:
        return numbers, len(numbers)
    try:
        # numbers.add(word_to_num(word))
        num = word_to_num(word)
        numbers.add(orig_word)
    except ValueError:
        try:
            # numbers.add(int(word))
            num = int(word)
            numbers.add(orig_word)
        except ValueError:
            try:
                # numbers.add(float(word))
                num = float(word)
                numbers.add(orig_word)
            except ValueError:
                if "-" in word:
                    word_chunks = word.split("-")
                    for chunk in word_chunks:
                        converted_nums, _ = new_reader_v2_get_numbers(chunk)
                        numbers = numbers | converted_nums
    return numbers, len(numbers)

def main(): 
    # currently: only calculating for train file path
    train_file_path = "/projects/instr/19sp/cse481n/OsCar/OsCar/drop_dataset/drop_dataset_train.json"
    dev_file_path = "/projects/instr/19sp/cse481n/OsCar/OsCar/drop_dataset/drop_dataset_dev.json"

    old_reader_wordset = set()
    new_reader_v1_wordset = set()
    new_reader_v2_wordset = set()

    old_reader_count = 0
    new_reader_v1_count = 0
    new_reader_v2_count = 0

    tokenizer = WordTokenizer()

    test_word = "200-yard"
    print('v2 gets: ', new_reader_v2_get_numbers(test_word))
    print('v1 gets: ', new_reader_v1_get_numbers(test_word))
    print('old gets: ', old_reader_get_numbers(test_word))
    # print('word_to_num gets: ', word_to_num(test_word))

    with open(train_file_path) as file_:
        train_dataset = json.load(file_)

        # iterate through all the passages, counting all extracted numbers
        for passage_id, passage_info in train_dataset.items():
            # print('new passage')
            passage_text = passage_info["passage"]
            passage_tokens = tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            # passage_words = passage_text.split(" ")

            # get all number counts from each passage
            for passage_token in passage_tokens:
                temp_v2_wordset, _ = new_reader_v2_get_numbers(passage_token.text)
                temp_v1_wordset = new_reader_v1_get_numbers(passage_token.text)
                temp_old_wordset = old_reader_get_numbers(passage_token.text)
                
                # print('word: ', passage_token.text)
                # print('v2 extracted: ', temp_v2_wordset)
                # print('v1 extracted: ', temp_v1_wordset)
                # print('old extracted: ', temp_old_wordset)

                if temp_v2_wordset is not None:
                    new_reader_v2_count += len(temp_v2_wordset)
                    new_reader_v2_wordset = new_reader_v2_wordset | temp_v2_wordset
                if temp_v1_wordset is not None:
                    new_reader_v1_count += 1
                    new_reader_v1_wordset.add(temp_v1_wordset)
                if temp_old_wordset is not None:
                    old_reader_count += 1
                    old_reader_wordset.add(temp_old_wordset)

            # for each passage, iterate through all the questions
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                question_tokens = tokenizer.tokenize(question_text)
                question_tokens = split_tokens_by_hyphen(question_tokens)
                # question_words = question_text.split(" ")

                # get all number counts from each question
                for question_token in question_tokens:
                    temp_v2_wordset, _ = new_reader_v2_get_numbers(question_token.text)
                    temp_v1_wordset = new_reader_v1_get_numbers(question_token.text)
                    temp_old_wordset = old_reader_get_numbers(question_token.text)
                    if temp_v2_wordset is not None:
                        new_reader_v2_count += len(temp_v2_wordset)
                        new_reader_v2_wordset = new_reader_v2_wordset | temp_v2_wordset
                    if temp_v1_wordset is not None:
                        new_reader_v1_count += 1
                        new_reader_v1_wordset.add(temp_v1_wordset)
                    if temp_old_wordset is not None:
                        old_reader_count += 1
                        old_reader_wordset.add(temp_old_wordset)

    # print out results
    set_difference_nums = new_reader_v2_wordset.difference(old_reader_wordset)

    new_reader_v2_arr = np.array(list(new_reader_v2_wordset))
    new_reader_v1_arr = np.array(list(new_reader_v1_wordset))
    old_reader_arr = np.array(list(old_reader_wordset))

    print('Size of word set (v2): ', len(new_reader_v2_wordset))
    print('Size of word set (v1): ', len(new_reader_v1_wordset))
    print('Size of word set (old): ', len(old_reader_wordset))
    
    print('\n\nTotal numbers extracted (v2):', new_reader_v2_count)
    print('Total numbers extracted (v1): ', new_reader_v1_count)
    print('Total numbers extracted (old): ', old_reader_count)

    set_difference_v2_old = np.setdiff1d(new_reader_v2_arr, old_reader_arr)
    set_difference_v2_v1 = np.setdiff1d(new_reader_v2_arr, new_reader_v1_arr)

    # print('numbers collected by new reader v2: ', new_reader_v2_wordset)
    # print('numbers collected by reader v1: ', new_reader_v1_wordset)
    # print('numbers collected by old reader: ', old_reader_wordset)

    # print('first 10 nums in new reader: ', new_reader_v2_arr[:10])
    # print('first 10 nums in old reader: ', old_reader_arr[:10])
    # print('set difference between v2 and old: ', set_difference_v2_old)
    print('set difference between v2 and v1: ', set_difference_v2_v1)
    # print('size of v2 and old difference set: ', len(set_difference_v2_old))
    # print('size of v1 and v2 difference set: ', len(set_difference_v1_v2))

    # print("Old DatasetReader numbers: ", old_reader_wordset)
    # print("New DatasetReader v2 numbers: ", new_reader_v2_wordset)

if __name__ == '__main__':
    main()