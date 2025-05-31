MODEL=bm25 # reasonir
cache_dir=cache
agent=qwen2.5:32b-instruct-q4_K_M
tokenizer=Qwen/Qwen2.5-32B-Instruct

for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
    python main.py \
        --task $TASK \
        --model $MODEL \
        --output_dir output/${MODEL} \
        --cache_dir ${cache_dir} \
        --agent $agent \
        --agent_tokenizer $tokenizer \
done