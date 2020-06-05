import ipdb
import pdb
import sys
import os
import regex as re
import argparse
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common.util import import_submodules
from distutils.util import strtobool

sys.path.insert(0,"code")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True) 
    parser.add_argument('--type', type=str) # single or append
    parser.add_argument('--inp_fp', type=str) 
    parser.add_argument('--out_fp', type=str)
    parser.add_argument('--out_ext', type=str, default='prob')
    parser.add_argument('--inp_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--epoch_num', type=int, default=-1)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ext_ratio', type=float, default=1)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--overwrite', default=True, type=lambda x:bool(strtobool(x)))

    return parser

def generate_probs(model_dir, inp_fp, weights_fp, type_, out_fp, out_ext, cuda_device, overwrite, batch_size, extraction_ratio, hparams):
    import_submodules('imojie') 

    if out_fp == None:
        inp_base = os.path.basename(inp_fp)
        inp_dir = os.path.basename(os.path.dirname(inp_fp))
        eval_dir = os.path.basename(os.path.dirname(os.path.dirname(inp_fp))) # train, dev or test
        inp_base = inp_base.replace('extractions',eval_dir)
        prob_dir = model_dir+'/prob' 
        os.makedirs(prob_dir, exist_ok=True)
        out_fp = prob_dir+'/'+inp_base+'.'+inp_dir+'.'+out_ext
    else:
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    if os.path.exists(out_fp) and not overwrite:
        print('found ',out_fp)
        return

    args = argparse.Namespace()
    args.archive_file = model_dir
    args.cuda_device = cuda_device
    args.embedding_sources_mapping = {}
    args.extend_vocab = None
    args.batch_weight_key = ''
    args.output_file = ''
    args.overrides = "{'model': {'token_based_metric': null}, 'iterator': {'batch_size': "+str(batch_size)+", \
        'instances_per_epoch': null}, 'trainer':{'num_epochs':1}, 'dataset_reader': {'max_tokens': 10000, \
            'gradients': false, 'max_extractions': 30, 'extraction_ratio': "+str(extraction_ratio)+", 'probability': true \
                 }, 'validation_dataset_reader': null}"
    args.weights_file = weights_fp
    args.input_file = inp_fp
    probs = evaluate_from_args(args)
    probsD = dict()
    # For some reason the last batch results are repeated in the probs 
    # Not an issue as they are just overwritten while forming the probsD
    for i in range(len(probs['example_ids'])):
        probsD[probs['example_ids'][i]] = probs['probs'][i]
    lines = open(inp_fp).readlines()

    all_fields = []
    for line_number, line in enumerate(lines):
        line = line.strip('\n')
        fields = line.split('\t')
        if line_number not in probsD: # the example is too large and rejected by dataloader ('max_tokens' argument)
            continue
        # Removing appended extractions after reranking
        fields[0] = fields[0].split('[SEP]')[0].strip()
        fields[2] = str(probsD[line_number])
        all_fields.append('\t'.join(fields))

    # if type_ == 'single':
    #     all_fields = []
    #     for line_number, line in enumerate(lines):
    #         line = line.strip('\n')
    #         fields = line.split('\t')
    #         if line_number not in probsD: # the example is too large and rejected by dataloader ('max_tokens' argument)
    #             continue
    #         # Removing appended extractions after reranking
    #         fields[0] = fields[0].split('[SEP]')[0].strip()
    #         fields[2] = str(probsD[line_number])
    #         all_fields.append('\t'.join(fields))
    # elif type_ == 'append':
    #     all_fields = []
    #     all_examples = dict()
    #     extractions = []
    #     lines = lines + ['']
    #     for line_number, line in enumerate(lines):
    #         line = line.strip('\n')
    #         if line_number != len(lines)-1:
    #             sentence, extraction, confidence = line.split('\t')
    #         else:
    #             sentence, extraction, confidence = '', '', 1
    #         if line_number == 0:
    #             old_sentence = sentence
    #         if line_number == len(lines)-1 or sentence != old_sentence:
    #             # if line_number == len(lines)-1:
    #                 # extractions.append(extraction)
    #                 # old_sentence = sentence
    #             all_examples[line_number-1] = [old_sentence, extractions]

    #             old_sentence = sentence
    #             extractions = []
    #         extractions.append(extraction)

    #     for line_number in probsD:
    #         # assert line_number in all_examples
    #         if line_number not in all_examples:
    #             continue
    #         sentence, extractions = all_examples[line_number]
    #         for ext_num, extraction in enumerate(extractions):
    #             confidence = probsD[line_number][ext_num].item()
    #             out = sentence+'\t'+extraction+'\t'+str(confidence)
    #             all_fields.append(out)

    # sorting all_fields according to the confidences assigned by bert_encoder
    all_fields_sorted = []
    prev_sent=None
    exts=[]
    for f in all_fields:
        sent = f.split('\t')[0]
        if sent!=prev_sent:
            if prev_sent!=None:
                exts = sorted(exts, reverse=True, key= lambda x: float(x.split('\t')[2]) )
                if hparams.topk != None:
                    exts = exts[:hparams.topk]
                all_fields_sorted.extend(exts)
            prev_sent=sent
            exts=[f]
        else:
            exts.append(f)
    exts = sorted(exts, reverse=True, key= lambda x: float(x.split('\t')[2]) )
    all_fields_sorted.extend(exts)

    open(out_fp,'w').write('\n'.join(all_fields_sorted))

    print('Probabilities written to: ',out_fp)
    return


def main():
    parser = parse_args()
    args = parser.parse_args()

    if args.epoch_num == -1:
        weights_fp = args.model_dir + '/best.th'
    else:
        weights_fp = args.model_dir + '/model_state_epoch_' + str(args.epoch_num) + '.th'
    if args.inp_dir != None:
        for fp in os.listdir(args.inp_dir):
            if re.search('^pro_output_.*.txt$', fp) != None:
                inp_fp = args.inp_dir + '/' + fp
                out_fp = args.out_dir + '/' + fp
                generate_probs(args.model_dir, inp_fp, weights_fp, args.type, out_fp, args.out_ext, args.cuda_device, overwrite=args.overwrite, batch_size=args.batch_size)
    else:
        generate_probs(args.model_dir, args.inp_fp, weights_fp, args.type, args.out_fp, args.out_ext, args.cuda_device, overwrite=args.overwrite, extraction_ratio=args.ext_ratio, batch_size=args.batch_size, hparams=args)

    

if __name__ == '__main__':
    main()
