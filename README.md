
<!---

### Notes to self:
###1. Gabi's oie-benchmark repo and benchmark in supervised oie repo have differences. One major difference is that oie-benchmark does not enforce one to one match which is required. We have added that from his supervised oie repo and are using one to one match with oie16.

###2. In mil_trainer, we had ignored the 2% sentences that had extractions more than the maximum number of tokens that BERT can handle. This might hurt recall since we are removing sentences with high number of extractions but only 2% of the sentences are like that so it shouldn't make much of a difference.

###3. RNNOIE has no output on the sentence:
A second factor is resource dependence ; there must be a perceptible threat of resource depletion , and it must be difficult to find substitutes .

-->

# Install dependencies
pip install -r requirements.txt
cd code/allennlp
pip install --editable .

All reported results are on pytorch-1.2 with TeslaV100 GPU having CUDA 10.0

# Download the (train,dev,test) data
bash download_data.sh
 
# Command to run the code
IMoJIE (on OpenIE-4, ClausIE, RnnOIE bootstrapping data with QPBO filtering)
python allennlp_script.py --param_path code/noie/configs/imojie.json --s models/imojie --mode train_test 


Arguments:
- param_path: file containing all the parameters for the model
- s:  path of the directory where the model will be saved
- mode: train, test, train_test

Important baselines:
IMoJIE (on OpenIE-4 bootstrapping)
python allennlp_script.py --param_path code/noie/configs/ba.json --s models/ba --mode train_test 

CopyAttn+BERT (on OpenIE-4 bootstrapping)
python allennlp_script.py --param_path code/noie/configs/be.json --s models/be --mode train_test 

## Generating aggregated data

Score using bert_encoder trained on oie4: 
```
python imojie/aggregate/score.py --model_dir models/oie4/bert_encoder/base --inp_fp data/train/comb_4cr/extractions.tsv --out_fp data/train/comb_4cr/extractions.tsv.be
```

Score using bert_append trained on comb_4cr (random): 
```            
python imojie/aggregate/score.py --model_dir models/comb_4cr/bert_append/rand --inp_fp data/train/comb_4cr/extractions.tsv.be --out_fp data/train/comb_4cr/extractions.tsv.ba
```

Filter using QPBO:
```
python imojie/aggregate/filter.py --inp_fp data/train/comb_4cr/extractions.tsv.ba --out_fp data/train/comb_4cr/extractions.tsv.filter
```

# Expected Results
vim models/imojie/test/carb_1/best_results.txt
(Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last)
(65.20/45.60/53.60, 33.50, 64.90/45.60/53.50)

vim models/ba/test/carb_1/best_results.txt
(Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last)
(60.80/47.20/53.10, 33.50, 59.00/47.50/52.60)

vim models/be/test/carb_3/best_results.txt
(Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last)
(59.50/45.50/51.60, 32.80, 52.90/46.70/49.60)
