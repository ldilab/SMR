
import json
import logging
import os
import torch

from smr.retrievers import RETRIEVAL_FUNCS
from smr.utils.llm import LLM

class SMR:
    def __init__(self, queries, query_ids, documents, excluded_ids, 
                instructions, doc_ids, task, cache_dir, long_context, 
                model_id, checkpoint, model, tokenizer, doc_dict, agent, 
                max_qlen, max_iteration, prompt_path, topk=10, retrieved_budget=25, **kwargs):
        self.queries = queries
        self.query_ids = query_ids
        self.documents = documents
        self.excluded_ids = excluded_ids
        self.instructions = instructions
        self.doc_ids = doc_ids
        self.task = task
        self.cache_dir = cache_dir
        self.long_context = long_context
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.model = model
        self.tokenizer = tokenizer
        self.doc_dict = doc_dict
        self.max_qlen = max_qlen
        self.max_iteration = max_iteration
        self.prompt_path = prompt_path
        self.topk = topk
        self.retrieved_budget = retrieved_budget
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = LLM(agent, tokenizer=tokenizer)
        self.kwargs = kwargs

        with open(self.prompt_path, 'r', encoding='utf-8') as fp:
            self.system_prompt = fp.read().strip()

    def process_query(self, qid: str, query: str) -> tuple:
        '''
        Process a single query with iterative refinement.
        Returns a tuple of query ID and a dictionary mapping document IDs to scores.
        '''
        retrieved = []
        curr_query = query
        iteration, output_length = 0, 0
        queries, query_ids = [query], [qid]
        scores = RETRIEVAL_FUNCS[self.model](queries=queries,query_ids=query_ids,documents=self.documents,excluded_ids=self.excluded_ids,
                                             instructions=self.instructions,doc_ids=self.doc_ids,task=self.task,cache_dir=self.cache_dir,
                                             long_context=self.long_context,model_id=self.model_id,checkpoint=self.checkpoint,**self.kwargs)

        for idx, (doc_id, score) in enumerate(scores[qid].items()):
            if idx >= self.topk:
                break
            retrieved.append(doc_id)

        history = [{
            'iteration': 0,
            'query': curr_query,
            'retrieved': retrieved.copy(),
            'action': 'start',
            'reason': None,
            'response': None,
            'output_length': output_length
        }]
        history_raw = [{'iteration': 0,'query': curr_query,'retrieved': retrieved.copy(),'action': 'start','reason': None,'response_raw': None,'output_length': output_length}]
        last_query, last_retrieved = curr_query, retrieved.copy()
        max_attempts, attempts, temperature = 3, 0, 0.0
        while iteration < self.max_iteration:
            user_prompt = json.dumps({
                'query': curr_query,
                'retrieved': [(did, self.doc_dict[did]) for did in retrieved]
            })
            action = None
            try:
                response, output_length, raw_response = self.agent.generate(user_prompt, self.system_prompt, temperature=temperature)
                if '```' in response:
                    response = response.split('```')[1]
                if 'json {' in response:
                    response = response.split('json')[1]
                response = response.strip()
                response = json.loads(response)
                action = response['action']
                if action == 'refine query':
                    curr_query = response['refined_query']
                    reason = response['reason']
                    scores = RETRIEVAL_FUNCS[self.model](queries=[curr_query],query_ids=[qid],documents=self.documents,excluded_ids=self.excluded_ids,
                                                         instructions=self.instructions,doc_ids=self.doc_ids,task=self.task,cache_dir=self.cache_dir,
                                                         long_context=self.long_context,model_id=self.model_id,checkpoint=self.checkpoint,**self.kwargs)
                    for idx, (doc_id, score) in enumerate(scores[qid].items()):
                        if idx >= self.topk or len(retrieved) >= self.retrieved_budget:
                            break
                        if doc_id not in retrieved:
                            retrieved.append(doc_id)
                elif action == 're-rank':
                    new_reranked = [did for did in response['reranked'] if did in self.doc_dict]
                    prev_retrieved = retrieved.copy()
                    retrieved = new_reranked.copy()
                    for did in prev_retrieved:
                        if len(retrieved) >= len(prev_retrieved):
                            break
                        if did not in retrieved:
                            retrieved.append(did)
                    reason = response['reason']
                elif action == 'stop':
                    break
                else:
                    raise ValueError(f'Invalid action: [{action}]')
            except Exception as e:
                logging.error(f'Query: [{qid} | {curr_query}] | Error: {e} | Response : {response}')
                attempts += 1
                temperature += 0.1
                if attempts >= max_attempts:
                    reason = f'Error during processing, fallback to last query and retrieved'
                    logging.error(f'Query: [{qid} | {curr_query}] | attempt: {attempts} | Error: {e} | Response: {response}')
                    temperature = 0.0
                    break
                continue
            if curr_query == last_query and retrieved == last_retrieved:
                break

            last_query = curr_query
            last_retrieved = retrieved.copy()
            iteration += 1
            history.append({
                'iteration': iteration,
                'query': curr_query,
                'retrieved': retrieved.copy(),
                'action': action,
                'reason': reason,
                'response': response,
                'output_length': output_length
            })
            history_raw.append({'iteration': iteration,'query': curr_query,'retrieved': retrieved.copy(),'action': action,'reason': reason,'response_raw': raw_response,'output_length': output_length})
        return qid, {did: int(360360 / (i + 1)) for i, did in enumerate(retrieved)}, history, history_raw