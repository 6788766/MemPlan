from tqdm import tqdm
import json
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.dataset import load_travelplanner_dataset


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--submission_file_dir", type=str, default="./")

    args = parser.parse_args()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'

    query_data_list = load_travelplanner_dataset(args.set_type)

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]

    submission_list = []

    for idx in tqdm(idx_number_list):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))
        plan = generated_plan[-1].get(f'{args.model_name}{suffix}_{args.mode}_parsed_results')
        submission_list.append({"idx":idx,"query":query_data_list[idx-1]['query'],"plan":plan})
    
    with open(f'{args.submission_file_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}_submission.jsonl','w',encoding='utf-8') as w:
        for unit in submission_list:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()
