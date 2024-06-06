import argparse
import pandas as pd
import numpy as np
from constraints import (
    no_commas, count_sentence, count_pos_under, count_paragraph, count_repeat, count_pos_non, count_pos_over,
    check_braced_strings, honorific_haeyo, honorific_hao, check_first_consonant, 
    check_middle_vowel, check_final_consonant
)

constraint_dict = {
    'no_commas': no_commas,
    'max_sentences': count_sentence,
    'max_verbs': lambda gen: count_pos_under(gen, '동사'),
    'max_adjectives': lambda gen: count_pos_under(gen, '형용사'),
    'max_paragraphs': lambda gen: count_paragraph(gen),
    'no_repeated_nouns': count_repeat,
    'no_pronouns': lambda gen: count_pos_non(gen, '대명사'),
    'no_dependent_nouns': lambda gen: count_pos_non(gen, '의존 명사'),
    'no_conjunctive_adverbs': lambda gen: count_pos_non(gen, '접속 부사'),
    'bracket_proper_nouns': check_braced_strings,
    'least_attributives': lambda gen: count_pos_over(gen, '관형사'),
    'honorific_haeyo': honorific_haeyo,
    'honorific_hao': honorific_hao,
    'first_consonant': check_first_consonant,
    'middle_vowel': check_middle_vowel,
    'final_consonant': check_final_consonant
}

def calculate_accuracy(data, constraint_name, check_function):
    result = np.array([0] * 90)
    # 3번의 generation에 대해서 모두 구한 후 각 결과의 총합 구하기
    for i in range(3):
        relevant_data = data[data['constraint'] == constraint_name][f'regeneration_{i}']
        result += np.array([1 for item in relevant_data if check_function(item)])

def parse_args():
    parser = argparse.ArgumentParser(description="Process model path for text generation analysis.")
    parser.add_argument("model_path", type=str, help="Specify the model path for the input and output CSV files.")
    parser.add_argument("--constraints", type=list, default=["max_verbs", "max_adjectives", "least_attributives"], help="The path or identifier for the constraints to use")
    return parser.parse_args()

def main(model_path, constraints):
    gen = pd.read_csv(f'{model_path}_gen.csv')
    kno = pd.read_csv(f'{model_path}_kno.csv')
    con = pd.read_csv(f'{model_path}_con.csv')

    results = {const: calculate_accuracy(con, const, constraint_dict[const]) for const in constraints}

    return results

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path#.split('/')[1]
    accuracies = main(model_path, args.constraints)
    pd.DataFrame(accuracies,index=[0]).to_csv(f'{model_path}-results.csv')