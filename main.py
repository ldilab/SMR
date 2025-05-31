import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

from smr.utils.utils import load_jsonl
from smr.retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics
from smr.smr import SMR




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge',
                                 'bge_ce', 'nomic', 'm2', 'contriever', 'reasonir'])
    parser.add_argument('--model_id', type=str, default=None, help='(Optional) Pass a different model ID for cache and output path naming.')
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--document_expansion', default=None, type=str, choices=[None, 'gold', 'full'],
                        help="Set to None to use original documents provided by BRIGHT; Set to `oracle` to use documents with oracle ones expanded'; Set to `full` to use all expanded documents.")
    parser.add_argument('--global_summary', default=None, choices=[None, 'concat'])
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--reasoning_id', type=str, default=None)
    parser.add_argument('--reasoning_length_limit', type=int, default=None)
    parser.add_argument('--separate_reasoning', action='store_true', help='Append reasoning after the original query, separate by <REASON>.')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--no_log', action='store_true', help="Disable logging to Google Sheets.")
    parser.add_argument('--sweep_output_dir', type=str, default=None)
    parser.add_argument('--skip_doc_emb', action='store_true', help="Skip document embedding.")
    parser.add_argument('--store_all_scores', action='store_true', help="The default is to store the top 1000 scores. This option will store all scores.")
    parser.add_argument('--prompt_path', type=str, default='prompt/prompt.txt', help="Prompt for the retriever.")
    parser.add_argument('--agent', type=str, default='qwen2.5:32b-instruct-q4_K_M', help="Agent to use for the retriever.")
    parser.add_argument('--agent_tokenizer', type=str, default='Qwen/Qwen2.5-32B-Instruct', help='Tokenizer to use for the model')
    parser.add_argument('--max_qlen', type=int, default=512, help="Max length of the query.")
    parser.add_argument('--max_iteration', type=int, default=16, help="Max number of iterations.")
    parser.add_argument('--topk', type=int, default=10, help='Number of top documents to retrieve')
    parser.add_argument('--retrieved_budget', type=int, default=25, help='Number of retrieved documents to keep in memory')
    args = parser.parse_args()

    # 1. set file paths
    ## 1) set the output directory
    if args.model_id is None:
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}_long_{args.long_context}")
    else:
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model_id}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    ## 2) set score_file path
    if args.reasoning is not None:
        score_file_path = os.path.join(args.output_dir,f'{args.reasoning}_score_{args.agent}_{args.model}.json')
        history_file_path = os.path.join(args.output_dir,f'history/{args.reasoning}_history_{args.agent}_{args.model}.jsonl')
    else:
        score_file_path = os.path.join(args.output_dir,f'score_{args.agent}_{args.model}.json')
        history_file_path = os.path.join(args.output_dir,f'history/history_{args.agent}_{args.model}.jsonl')
    os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

    ## validate (expansion vs summary) - mutually exclusive
    assert args.document_expansion is None or args.global_summary is None, "Cannot use expansion and summary together!"
    if args.global_summary:
        assert not args.long_context, "Global summary is supposed to enhance short-context retrieval!"

    ## 3) set dataset source based on expansion/summary
    document_postfix = ''
    if args.document_expansion == 'gold':
        document_postfix = '_expanded_gold_only'
        dataset_source = 'rulins/bright-expanded'
    elif args.document_expansion == 'full':
        document_postfix = '_expanded'
        dataset_source = 'rulins/bright-expanded'
    elif args.global_summary == 'concat':
        document_postfix = '_concat_with_summary'
        dataset_source = 'rulins/bright-expanded'
    else:
        dataset_source = 'xlangai/BRIGHT'
    
    if args.reasoning is not None and 'llama3-8b' in args.reasoning:
        reasoning_source = 'dreamorg/BRIGHT'
    else:
        reasoning_source = 'xlangai/BRIGHT'
    print(f"Dataset source: {dataset_source}")
    print(f"Reasoning source: {reasoning_source}")

    # 2. load examples and documents
    ## 1) load examples (queries)
    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None and args.reasoning in ['promptriever', 'cot_v2']:
        reasoning_file = f"cache/{args.reasoning}/{args.reasoning_id}_{args.task}.jsonl"
        examples = load_jsonl(reasoning_file)
        for e in examples:
            e['query'] = e['query'] + '\n' + e['rewritten_query']
    elif args.reasoning is not None and args.separate_reasoning:
        examples = load_dataset(dataset_source, 'examples', cache_dir=args.cache_dir)[args.task]
        reasoning_examples = load_dataset(reasoning_source, f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    elif args.reasoning is not None and args.reasoning_length_limit is None:
        examples = load_dataset(reasoning_source, f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    elif args.reasoning is not None and args.reasoning_length_limit:
        reasoning_file = f"cache/reasoning/{args.task}_{args.reasoning}_{args.reasoning_length_limit}"
        with open(reasoning_file, 'r') as f:
            examples = json.load(f)
    else:
        examples = load_dataset(dataset_source, 'examples',cache_dir=args.cache_dir)[args.task]

    ## 2) load documents
    if args.long_context:
        doc_pairs = load_dataset(dataset_source, 'long_documents'+document_postfix, cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset(dataset_source, 'documents'+document_postfix, cache_dir=args.cache_dir)[args.task]
    
    doc_ids = []
    documents = []
    doc_dict = {}
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
        doc_dict[dp['id']] = dp['content']

    # 3. run retrieval
    if not os.path.isfile(score_file_path):
        print("The scores file does not exist, start retrieving...")

        with open(os.path.join(args.config_dir,args.model.split('_ckpt')[0].split('_bilevel')[0],f"{args.task}.json")) as f:
            config = json.load(f)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
        for qid, e in enumerate(examples):
            if args.separate_reasoning:
                new_query = f"{e['query']}\n<REASON>\n{reasoning_examples[qid]['query']}"
                queries.append(new_query)
            else:
                queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap)==0

        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"
        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")):
            try:
                with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")) as f:
                    cached_doc_ids = json.load(f)
                for id1,id2 in zip(cached_doc_ids,doc_ids):
                    assert id2 in cached_doc_ids
            except:
                print("Document IDs mismatche with the cached version!")
        else:
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json"),'w') as f:
                json.dump(doc_ids,f,indent=2)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})
        if args.skip_doc_emb:
            kwargs.update({'skip_doc_emb': args.skip_doc_emb})
        if args.store_all_scores:
            kwargs.update({'store_all_scores': args.store_all_scores})
        
        model_id = args.model_id if args.model_id is not None else args.model
        smr_machine = SMR(queries=queries,query_ids=query_ids,documents=documents,excluded_ids=excluded_ids,
                            instructions=config['instructions_long'] if args.long_context else config['instructions'],
                            doc_ids=doc_ids,task=args.task,cache_dir=args.cache_dir,long_context=args.long_context,
                            model_id=model_id,checkpoint=args.checkpoint,model=args.model,tokenizer=args.agent_tokenizer,doc_dict=doc_dict,agent=args.agent, 
                            max_qlen=args.max_qlen, max_iteration=args.max_iteration,prompt_path=args.prompt_path,topk=args.topk,retrieved_budget=args.retrieved_budget,**kwargs)

        scores = {}
        for q_idx, query in enumerate(tqdm(queries)):
            qid = query_ids[q_idx]
            qid, retrieved, history, history_raw = smr_machine.process_query(qid, query)
            scores[qid] = {}
            for docid, score in retrieved.items():
                scores[qid][docid] = score
            with open(history_file_path, 'a') as f_history:
                dumps = json.dumps({
                    'qid': qid,
                    'results': retrieved,
                    'history': history
                })
                f_history.write(f'{dumps}\n')
                f_history.flush()
        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    

    # track successful completion of the run
    if args.sweep_output_dir:
        with open(os.path.join(args.sweep_output_dir, 'done'), 'w') as f:
            f.write('done')