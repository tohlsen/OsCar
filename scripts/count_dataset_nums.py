# Helper script to count extracted numbers from the datasets

import json
import string
from word2number.w2n import word_to_num # need to 'pip install word2number' to use

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

def main(): 
    # currently: only calculating for train file path
    train_file_path = "/projects/instr/19sp/cse481n/OsCar/OsCar/drop_dataset/drop_dataset_train.json"
    dev_file_path = "/projects/instr/19sp/cse481n/OsCar/OsCar/drop_dataset/drop_dataset_dev.json"

    old_reader_count = 0
    new_reader_v1_count = 0
    new_reader_v2_count = 0

    with open(train_file_path) as file_:
        train_dataset = json.load(file_)

        # iterate through all the passages, counting all extracted numbers
        for passage_id, passage_info in train_dataset.items():
            passage_text = passage_info["passage"]
            passage_words = passage_text.split(" ")

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

                # get all number counts from each question
                for question_word in question_text:
                    new_reader_v1_count += new_reader_v1_get_count(question_word)
                    _, temp_count = new_reader_v2_get_count(question_word)
                    new_reader_v2_count += temp_count
                    old_reader_count += old_reader_get_count(question_word)

    # print out results
    print("Old DatasetReader number count: ", old_reader_count)
    print("New DatasetReader v1 number count: ", new_reader_v1_count)
    print("New DatasetReader v2 number count: ", new_reader_v2_count)

if __name__ == '__main__':
    main()