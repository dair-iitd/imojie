#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'code/utils/bertviz'))
	print(os.getcwd())
except:
	pass

#%%
from bertviz import head_view
from transformers import BertTokenizer, BertModel


#%%
get_ipython().run_cell_magic('javascript', '', "require.config({\n  paths: {\n      d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',\n      jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',\n  }\n});")


#%%
import ipdb
def show_head_view(model, tokenizer, sentence_a, sentence_b=None):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    if sentence_b:
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        sentence_b_start = None
    print(attention[0].shape, attention[1].shape)
    head_view(attention, tokens, sentence_b_start)


#%%
import sys

sys.path.insert(0,"/home/keshav/neural_oie")
sys.path.insert(0,"/home/keshav/neural_oie/code")
from transformers import BertTokenizer, BertModel
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive

import_submodules('noie')
best = load_archive('/home/keshav/neural_oie/models/oie4/bert_append/base', cuda_device=0)
best.model._source_embedder.token_embedder_tokens.transformer_model


#%%
model = best.model._source_embedder.token_embedder_tokens.transformer_model
for module in model.modules():
    try:
        module.output_attentions = True
    except:
        continue

model_version = 'bert-base-uncased'
# do_lower_case = True
# model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
sentence_a = "He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia"
sentence_b = "He served as the first Prime Minister of Australia"
show_head_view(model.cpu(), tokenizer, sentence_a, sentence_b)


#%%



