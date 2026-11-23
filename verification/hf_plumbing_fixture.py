"""Create a tiny random local Qwen3 fixture to exercise real HF/PEFT entry points."""
from pathlib import Path
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
p=Path('verification/random_hf_model');p.mkdir(parents=True,exist_ok=True)
vocab={'[PAD]':0,'[EOS]':1,'[UNK]':2,'[BOS]':3}
for i in range(4,64):vocab[f't{i}']=i
raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
t=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]',bos_token='[BOS]')
t.chat_template="{% for message in messages %}{{ message['role'] }} {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
t.save_pretrained(p)
m=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=2048,bos_token_id=3,eos_token_id=1,pad_token_id=0,tie_word_embeddings=True));m.save_pretrained(p)
print('Saved a randomly initialized local Qwen3 fixture.')
