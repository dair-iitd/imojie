# IMoJIE

Iterative Memory based Joint OpenIE (please don't ask us where the "Joint" came from ;))

A BERT-based OpenIE system that generates extraction using an iterative Seq2Seq model, as described in the following publication, in ACL 2020, [insert link here](https://www.google.com)

## Installation Instructions
Create a new conda environment and install the dependencies using,
```
pip install -r requirements.txt
```
This will install custom versions of allennlp and pytorch_transformers based on the code in the folder.

All reported results are based on pytorch-1.2 run on a TeslaV100 GPU (CUDA 10.0). Results may vary slightly with change in environment.

## Execution Instructions
### Data Download
bash download_data.sh 

This downloads the (train, dev, test) data

### Running the code
IMoJIE (on OpenIE-4, ClausIE, RnnOIE bootstrapping data with QPBO filtering)
```
python allennlp_script.py --param_path imojie/configs/imojie.json --s models/imojie --mode train_test 
```

Arguments:
- param_path: file containing all the parameters for the model
- s:  path of the directory where the model will be saved
- mode: train, test, train_test

Important baselines:

IMoJIE (on OpenIE-4 bootstrapping)
```
python allennlp_script.py --param_path imojie/configs/ba.json --s models/ba --mode train_test 
```

CopyAttn+BERT (on OpenIE-4 bootstrapping)
```
python allennlp_script.py --param_path imojie/configs/be.json --s models/be --mode train_test --type single --beam_size 3
```

### Generating aggregated data

Score using bert_encoder trained on oie4: 
```
python imojie/aggregate/score.py --model_dir models/score/be --inp_fp data/train/4cr_comb_extractions.tsv --out_fp data/train/4cr_comb_extractions.tsv.be 
```

Score using bert_append trained on comb_4cr (random): 
```            
python imojie/aggregate/score.py --model_dir models/score/4cr_rand --inp_fp data/train/4cr_comb_extractions.tsv.be --out_fp data/train/4cr_comb_extractions.tsv.ba
```

Filter using QPBO:
```
python imojie/aggregate/filter.py --inp_fp data/train/4cr_comb_extractions.tsv.ba --out_fp data/train/4cr_qpbo_extractions.tsv
```

### Note on the notation
We have been internally calling our model as "bert-append" (ba) until the day of submission of the paper and CopyAttention + BERT as "bert-encoder" (be). So you will find similar references throughout the code-base. In this context, IMoJIE is bert-append trained on qpbo filtered data.

### Expected Results
Format: (Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last)

models/imojie/test/carb_1/best_results.txt \
(64.70/45.60/53.50, 33.30, 63.80/45.80/53.30)

models/ba/test/carb_1/best_results.txt \
(Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last) \
(63.50/45.80/53.20, 33.10, 60.40/46.30/52.40)

models/be/test/carb_3/best_results.txt \
(Prec/Rec/F1-Optimal, AUC, Prec/Rec/F1-Last) \
(59.50/45.50/51.60, 32.80, 52.90/46.70/49.60)

## Resources

Downloading the pre-trained models:
```
zenodo_get 3779954
```

Downloading the data:
```
zenodo_get 3775983
```

Downloading the results:
```
zenodo_get 3780045
```

### Citing
If you use this code, please cite:

## !Update this!
```
@inproceedings{kolluru-etal-2020-imojie,
    title = "{IM}o{JIE}: {I}terative {M}emory-{B}ased {J}oint {O}pen {I}nformation {E}xtraction",
    author = "Kolluru, Keshav  and
      Aggarwal, Samarth  and
      Rathore, Vipul and
      Mausam, Mausam and
      Chakrabarti, Soumen",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = july,
    year = "2020",
    address = "Seattle, U.S.A",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1651",
    doi = "10.18653/v1/D19-1651",
    pages = "6263--6268",
    abstract = "While traditional systems for Open Information Extraction were statistical and rule-based, recently neural models have been introduced for the task. Our work builds upon CopyAttention, a sequence generation Open IE model. Our analysis reveals that CopyAttention produces a constant number of extractions per sentence, and its extracted tuples often express redundant information.
    We present IMoJIE, an extension to CopyAttention, which produces the next extraction conditioned on all previously extracted tuples. This approach overcomes both shortcomings of CopyAttention, resulting in a variable number of diverse extractions per sentence. We train IMoJIE on training data bootstrapped from extractions of several non-neural systems, which have been automatically filtered to reduce redundancy and noise.  IMoJIE outperforms CopyAttention by about 18 F1 pts, and a BERT-based strong baseline by 2 F1 pts, establishing a new state of the art for the task. 
",
}
```


## Contact
In case of any issues, please send a mail to
```keshav.kolluru (at) gmail (dot) com``` 


