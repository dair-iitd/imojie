from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules, JsonDict, sanitize
import argparse
import_submodules('imojie')

def process(token_ids):
    temp=" ".join(token_ids)
    temp = temp.replace(" ##","")
    temp = temp.replace("[unused1]","( ")
    temp = temp.replace("[unused2]"," ; ")
    temp = temp.replace("[unused3]","")
    temp = temp.replace("[unused4]"," ; ")
    temp = temp.replace("[unused5]","")
    temp = temp.replace("[unused6]"," )")
    temp = temp.strip()
    temp = temp.split("[SEP]")
    ans=[]
    for x in temp:
        if x!="":
            ans.append(x)
    return ans

def main(args):
    archive = load_archive(
        "models/imojie",
        weights_file="models/imojie/model_state_epoch_7.th",
        cuda_device=-1)

    predictor = Predictor.from_archive(archive, "noie_seq2seq")
    out = open(args.out, 'w')
    input_instances = []
    sentences = []
    for line in open(args.inp, 'r'):
        instance = predictor._dataset_reader.text_to_instance(line)
        input_instances.append(instance)
        sentences.append(line)
    output_instances = predictor._model.forward_on_instances(input_instances)
    for output in output_instances:
        output = sanitize(output)
        output = process(output["predicted_tokens"][0])
        out.write(line)
        out.write('\n'.join(output)+'\n\n')
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--out')
    args = parser.parse_args()
    main(args)
