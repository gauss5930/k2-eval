import argparse
from datasets import load_dataset
import pandas as pd
from outlines import models
import outlines
import torch
from template import gen_template, kno_template, constraints
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm.auto import tqdm

def load_data(subset):
    return pd.DataFrame(load_dataset("HAERAE-HUB/K2-Eval", subset)['test'])

def initialize_llm(model_path):
    llm = LLM(model=model_path, #max_model_len=4096, 
              tensor_parallel_size=torch.cuda.device_count())#,token="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")
    model = models.VLLM(llm)
    generator = outlines.generate.choice(model, ["A","B","C","D"])
    return generator, llm

# OpenAI API model 로드
def initialize_openai(model_path):
    model = outlines.models.openai("gpt-4o")
    generator = outlines.generate.choice(model, ["A", "B", "C", "D"])
    return generator, model_path

def generate_answers(llm, prompts, sampling_params):
    answers = llm.generate(prompts, sampling_params)
    return [answer.outputs[0].text.strip() for answer in answers]

# OpenAI API model을 사용한 generation
def generate_answers_openai(llm, prompts):
    client = OpenAI()

    result = []
    for prompt in tqdm(prompts):  
        response = client.chat.completions.create(
            model=llm,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1600,
            stop=['###'],
            messages=[{"role": "user", "content": prompt}]
        )
        result.append(response.choices[0].message.content)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text based on specified models and constraints")
    parser.add_argument("--model_path", type=str, required=True, help="The path or identifier for the model to use")
    parser.add_argument("--model_type", type=str, required=True, help="Huggingface model은 hf, OpenAI API model은 openai로 설정")
    parser.add_argument("--constraints", type=list, default=["max_verbs", "max_adjectives", "least_attributives"], help="evaluation 진행할 constraint 선언")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    gen_data = load_data("generation")
    kno_data = load_data("knowledge")
    
    # Initialize model with command line argument
    if args.model_type == "openai":
        generator, llm = initialize_openai(args.model_path)
    elif args.model_type == "hf":
        generator, llm = initialize_llm(args.model_path)
    else:
        raise ValueError("Right option plz")
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95,
        min_tokens=20,
        max_tokens=1600,
        stop=['###']
    )
    
    # Generate answers for generation data
    gen_prompts = [gen_template.format(instruction) for instruction in gen_data.instruction]
    if args.model_type == "hf":
        gen_answers = generate_answers(llm, gen_prompts, sampling_params)
    elif args.model_type == "openai":
        gen_answers = generate_answers_openai(llm, gen_prompts)
    
    gen_data['generation'] = gen_answers
    
    # Merge knowledge data with generation results
    merged_data = gen_data.merge(kno_data, on=['instruction'])
    
    # Generate answers for knowledge data
    kno_prompts = [kno_template.format(row.instruction, row.generation, row.question, row.a, row.b, row.c, row.d) for _, row in merged_data.iterrows()]
    kno_answers = generator(kno_prompts)
        
    merged_data['predict'] = kno_answers
    
    if not args.constraints:
        constraints = {const: constraints[const] for const in args.constraints}
        
    threshold_list = [3, 5, 7, 10, 12, 15, 20]
    example_dataset, constraint_results = [], []
    
    # Apply constraints
    # threshold에 따라서 문제를 생성
    for constraint, template in constraints.items():
        example_dataset += [(i, gen, constraint, template.format(gen, thres=i)) for gen in gen_answers for i in threshold_list]
    constraint_data = pd.DataFrame(example_dataset, columns=['threshold','generation','constraint','query'])
    # 각 문제에 대해 총 3번의 generation 진행
    for n in range(3):
        if args.model_type == "hf":
            constraint_results = generate_answers(llm, constraint_data['query'].values, sampling_params)
        elif args.model_type == "openai":
            constraint_results = generate_answers_openai(llm, constraint_data['query'].values)
        constraint_data[f'regeneration_{n}'] = constraint_results
    
    # Output to CSV
    model_path = args.model_path#.split('/')[1]
    gen_data.to_csv(f'{model_path}_gen.csv')
    merged_data.to_csv(f'{model_path}_kno.csv')
    constraint_data.to_csv(f'{model_path}_con.csv')

if __name__ == "__main__":
    main()
