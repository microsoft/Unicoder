# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Language transfer")
    parser.add_argument("--task", type=str, default="generation_from_pretrained_bart", 
                        help="dataset type")

    parser.add_argument("--ngpu", type=int, default=4, 
                        help="number of gpus")
    parser.add_argument("--epoch", type=int, default=20, 
                        help="total number of epochs")
    parser.add_argument("--start", type=int, default=1, 
                        help="start epochs")
    parser.add_argument("--spe", type=str, 
                        help="path to SPE model")
    parser.add_argument("--exp", type=str, required=True, 
                        help="experiment tag")
    parser.add_argument("--dataset", type=str, 
                        help="dataset name, NTG/QG")
    parser.add_argument("--split", type=str, default="dev", 
                        help="dataset split to decode")
    parser.add_argument("--beam", type=int, default=5, 
                        help="beam size")
    parser.add_argument("--data_path", type=str,
                        help="path to binary data")
    parser.add_argument("--code_root", type=str,
                        help="path to code root")
    parser.add_argument("--save_dir", type=str,
                        help="path to the dir to save checkpoints and decoded results")


    args = parser.parse_args()
    generate_script = os.path.join(args.code_root, "evaluation/generate.sh")
    cmd = "bash {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}".format(generate_script,
                                                           args.task,
                                                           args.ngpu,
                                                           args.exp,
                                                           args.epoch, 
                                                           args.dataset,
                                                           args.split,
                                                           args.spe,
                                                           args.data_path,
                                                           args.beam,
                                                           args.start,
                                                           args.save_dir,
                                                           args.code_root)
    print(cmd)
    os.system(cmd)
