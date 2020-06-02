# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import sys
import argparse
import subprocess
import numpy as np


def compute_bleus(args, epochs, split):
    all_scores = {}
    lgs = args.lgs.split("-")
    ref_folder = args.ref_folder
    for e in epochs:
        fd = os.path.join(args.save_dir, 
                          "decodes/{}/checkpoint{}/{}".format(args.exp, e, split)) 
        scores = {}
        for lg in lgs:
            fs = os.path.join(fd, "{}_tgt.{}.hyp".format(lg, split))
            ref = os.path.join(ref_folder, "{}.tgt.{}".format(lg, split))
            if os.path.isfile(fs):
                hp = open(fs, 'r')
                hps = hp.readlines()
                hp.close()
                rf = open(ref, 'r')
                rfs = rf.readlines()
                rf.close()
                if len(hps) == len(rfs):
                    cmd = "python -m sacrebleu --force -lc -l {}-{} {} < {}".format(lg, lg, ref, fs)
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                    result = p.communicate()[0].decode("utf-8")
                    x = result.split("=")[1].split()[0]
                    print(fd, lg, split) 
                    print(x)
                    scores[lg] = float(x)
        if len(scores) > 0:
            all_scores[e] = scores
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language transfer")
    parser.add_argument("--exp", type=str, required=True,
                        help="experiment tag")
    parser.add_argument("--dataset", type=str, default="NTG",
                        help="dataset type")
    parser.add_argument("--split", type=str, default="valid",
                        help="dataset split to decode")
    parser.add_argument("--epoch", type=int, required=True,
                        help="total number of epochs")
    parser.add_argument("--lgs", type=str, default="en-de-es-fr-ru",
                        help="languages to evaluation")
    parser.add_argument("--supervised_lg", type=str, default="en",
                        help="language used for supervision during finetuning")
    parser.add_argument("--multilingual", action="store_true",
                        help="multilingual training, select best epoch with the average bleu scores over all languages")

    parser.add_argument("--ref_folder", type=str,
                        help="the folder that contains reference files")
    parser.add_argument("--task", type=str, default="generation_from_pretrained_bart",
                        help="dataset type NTG/QG")
    parser.add_argument("--ngpu", type=int, default=4,
                        help="number of gpus")
    parser.add_argument("--start", type=int, default=1,
                        help="start epochs")
    parser.add_argument("--spe", type=str,
                        help="SPE")
    parser.add_argument("--data_path", type=str,
                        help="path to binary data")
    parser.add_argument("--test_beam", type=int, default=10,
                        help="beam size") 
    parser.add_argument("--valid_split", type=str, default="valid",
                        help="validation split name")
    parser.add_argument("--test_split", type=str, default="test",
                        help="test split name")
    parser.add_argument("--save_dir", type=str,
                        help="dir to save all checkpoints and decoded results")
    parser.add_argument("--code_root", type=str,
                        help="path to code root")

    args = parser.parse_args()
    all_scores = compute_bleus(args, [e+1 for e in range(args.epoch)], args.valid_split) 

    decode_script = os.path.join(args.code_root, "evaluation/decode_all.py")
    lgs = args.lgs.split("-")

    if not args.multilingual:
        LG = args.supervised_lg
        all_LG = []
        all_others = []
        for e in all_scores:
            ss = all_scores[e]
            if LG in ss:
                all_LG.append((e, ss[LG]))
            bleus = []
            for lg in lgs:
                if lg != LG and lg in ss:
                    bleus.append(ss[lg])
            if len(bleus) == len(lgs) -1:
                all_others.append((e, np.mean(bleus)))

        all_LG.sort(key=lambda x: x[1], reverse=True)
        all_others.sort(key=lambda x: x[1], reverse=True)
        
        max_LG_ep, max_LG_valid = all_LG[0]
        max_other_ep, max_other_valid = all_others[0]

        print("largest {} at Epoch {} : {}".format(LG, max_LG_ep, max_LG_valid))
        print("largest others at Epoch {} : avg - {}; {}".format(max_other_ep, max_other_valid, all_scores[max_other_ep]))

        print("best others test start") 
        os.system("python {0} --beam {1} --start {2} --ngpu {3} --epoch {4} --split {5} --exp {6} " \
                  "--data_path {7} --dataset {8} --code_root {9} --task {10} --spe {11} --save_dir {12}".format(decode_script, 
                    args.test_beam, max_other_ep, args.ngpu, max_other_ep, args.test_split, args.exp, 
                    args.data_path, args.dataset, args.code_root, args.task, args.spe, args.save_dir))
        print("best {} test start".format(LG)) 
        os.system("python {0} --beam {1} --start {2} --ngpu {3} --epoch {4} --split {5} --exp {6} " \
                  "--data_path {7} --dataset {8} --code_root {9} --task {10} --spe {11} --save_dir {12}".format(decode_script, 
                    args.test_beam, max_LG_ep, args.ngpu, max_LG_ep, args.test_split, args.exp, 
                   args.data_path, args.dataset, args.code_root, args.task, args.spe, args.save_dir))
        
        best_test = {lg : max_other_ep for lg in lgs}
        best_test[LG] = max_LG_ep
    else:
        all_lgs = []
        for e in all_scores:
            ss = all_scores[e]
            bleus = []
            for lg in lgs:
                if lg in ss:
                    bleus.append(ss[lg])
            if len(bleus) == len(lgs):
                all_lgs.append((e, np.mean(bleus)))

        all_lgs.sort(key=lambda x: x[1], reverse=True)
        max_lgs_ep, max_lgs_valid = all_lgs[0]

        print("largest average validation bleus at Epoch {} : avg - {}; {}".format(max_lgs_ep, max_lgs_valid, all_scores[max_lgs_ep]))
        print("test start") 
        os.system("python {0} --beam {1} --start {2} --ngpu {3} --epoch {4} --split {5} --exp {6} " \
                  "--data_path {7} --dataset {8} --code_root {9} --task {10} --spe {11} --save_dir {12}".format(decode_script, 
                    args.test_beam, max_lgs_ep, args.ngpu, max_lgs_ep, args.test_split, args.exp, 
                    args.data_path, args.dataset, args.code_root, args.task, args.spe, args.save_dir))

        best_test = {lg :  max_lgs_ep for lg in lgs}

    fd = os.path.join(args.save_dir, "decodes/{}/results".format(args.exp))
    if not os.path.exists(fd):
        os.makedirs(fd)
    print()
    for lg in lgs:
        pred_f = os.path.join(args.save_dir, 
                "decodes/{}/checkpoint{}/{}".format(args.exp, best_test[lg], args.test_split),
                "{}_tgt.{}.hyp".format(lg, args.test_split)) 
        dest_f = os.path.join(fd, "{}.prediction".format(lg))
        os.system("cp {} {}".format(pred_f, dest_f))
        print("Save {}.prediction to {}".format(lg, dest_f))

    print("Done!")
