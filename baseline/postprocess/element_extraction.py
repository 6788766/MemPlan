import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.dataset import load_travelplanner_dataset
from tqdm import tqdm
import json
import ast
import re


_JSON_FENCE_RE = re.compile(r"```json\\s*(.*?)\\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_JSON_RE = re.compile(r"(\\[\\s*\\{.*\\}\\s*\\]|\\{.*\\})", re.DOTALL)


def _extract_plan_text(raw_line: str) -> str:
    if raw_line is None:
        return ""
    line = str(raw_line).strip()
    if not line:
        return ""
    # Lines are written as: "<idx>\\t<content...>"
    if "\t" in line:
        _idx, rest = line.split("\t", 1)
        line = rest.strip()
    return line


def _parse_plan_from_line(raw_line: str):
    text = _extract_plan_text(raw_line)
    if not text:
        return None

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        # Try to grab the first JSON-like block.
        match = _FIRST_JSON_RE.search(text)
        candidate = match.group(1).strip() if match else text.strip()

    # Prefer strict JSON; fall back to Python literal (the upstream scripts use eval()).
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")

    args = parser.parse_args()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'


    results = open(f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt','r').read().strip().split('\n')
    
    query_data_list = load_travelplanner_dataset(args.set_type)

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]
    for idx in tqdm(idx_number_list[:]):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))
        if generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_results'] not in ["","Max Token Length Exceeded."] :
            parsed = _parse_plan_from_line(results[idx-1] if idx - 1 < len(results) else "")
            if parsed is None:
                print(
                    f"{idx}:\n{results[idx-1] if idx - 1 < len(results) else ''}\n"
                    "This plan cannot be parsed. Setting parsed_results=None and continuing."
                )
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
            else:
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = parsed
        else:
            if args.mode == 'two-stage':
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
            else:
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
  
        with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json','w') as f:
            json.dump(generated_plan,f)
