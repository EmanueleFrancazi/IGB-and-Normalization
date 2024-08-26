# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#avoid apex
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

import sys
# Dynamically add the project root (two levels up) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from RunsCode.ViT.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from RunsCode.ViT.utils.data_utils import get_loader
from RunsCode.ViT.utils.dist_util import get_world_size

from RunsCode.ViT.models.modeling import VisionTransformer, CONFIGS




print(project_root)
print(sys.path)

from utils.IGB_utils import *



logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10  
    elif args.dataset == "CatsVsDogs":
        num_classes = 2
    else:
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    
    model.StoringVariablesCreation()  # Initialize storage variables 
    
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model, params):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    
    #compute the real number of steps as the number of batches in the train loader times the number of epochs
    print('number of epochs for the simulation is', args.num_steps)
    args.num_steps = args.num_steps*len(params['train_loader'])
    print('number of steps for the simulation is', args.num_steps)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    #avoid apex
    """
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    """
    # With the following, which simply skips the amp initialization but keeps the optimizer and model unchanged:
    if args.fp16:
        model.half()  # Use PyTorch's native support for half-precision
        

    # Distributed training
    #avoid apex
    """
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    """
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(params['train_loader'],
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        train_losses = []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            #print('BATCH', batch)
            
            x, y = batch
            #print(f"Batch shape: {x.shape}")
            #loss = model(x, y)
            
            Res= model.training_step(batch, params['num_trdata_points'], params)
            loss = Res['loss']
            #print('mean loss :', loss)
            #print(loss)
            #TrRes.append(Res['train_f'])
            train_losses.append(loss)
            
            

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            #avoid apex
            """
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            """
            
            if args.fp16:
                loss.backward()
            else:
                loss.backward()

            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                #avoid apex
                """
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                """
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                """
                #substituted by the evaluation block from IGB_utils
                
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()
                """
                if global_step % t_total == 0:
                    break       
                if (global_step) in params['TimeValSteps']:      # Validation phase
                    
                    #first we save the norm of the gradient used for the step and the corresponding step size
                    total_norm = 0
                    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5  
                    
                    model.GradNorm.append(total_norm)
                    #we then multiply the gradnorm by the learning rate to get the step size
                    model.StepSize.append(total_norm*optimizer.param_groups[-1]['lr']) #this works because in our caase all params group have the same learning rate
                    
                    #test eval
                    test_result = evaluate(model, params['test_loader'], 'Eval', params)
                    test_result['train_loss'] = torch.stack(train_losses).mean().item()
                    model.time.append(global_step+1)
    
                    
                    #train eval
                    train_result = evaluate(model, params['train_loader'], 'Train', params)
                    WandB_logs(global_step+1, model) #log on wandb 
                    save_on_file(model, params)
                
                

            
            #step+=1
            #optimizer.zero_grad() #reset gradient before next step
                
                
                

        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", 'CatsVsDogs'], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=2000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    #avoid apex
    """
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    """
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")

    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    
    
    #args = parser.parse_args()
    
    # Parse known arguments
    args, unknown = parser.parse_known_args()
    
    
    # Merge args0 (from IGB_utils) and args
    for key, value in vars(args0).items():
        setattr(args, key, value)

    # Now args contains all arguments from both IGB_utils.py and train.py
    print(args)
    
    # creating folder to store results
    
    FolderPath = './RunsResults/SimulationResult'
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath, exist_ok=True)  
        
    #then we create the specific sample folder
    FolderPath = FolderPath +'/Data_Shift' + str(shift_const) + '/Sample' + str(args.SampleIndex)
    print('La cartella creata per il sample ha come path: ', FolderPath)
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath, exist_ok=True) 
        
    # Prepare dataset
    dataset = get_loader(args)
    train_loader  = dataset['train_loader']
    test_loader =  dataset['test_loader']
    num_trdata_points = dataset['num_trdata_points']
    num_valdata_points = dataset['num_valdata_points']
    input_size = dataset['input_size']
    
    train_classes = per_class_counting(dataset['train_ds'])
    valid_classes = per_class_counting(dataset['test_ds'])
    
    if Loss_function=='Hinge':
        label_list=[-1, 1] #for the hinge we have to map the labels
    else:
        label_list = list(train_classes.keys())
    print('the classes for train and valid are: ', train_classes, valid_classes, label_list)    
    
    num_classes = len(train_classes)



    # fix the steps for eval measures
    N_ValidSteps=30
    num_tr_batches = len(train_loader)
    TimeValSteps= ValidTimes(args.num_steps, num_tr_batches, N_ValidSteps)
    print('epochs with validation: ', TimeValSteps)
        
    # wrapping flags/variables from IGB_utils.py
    params = {'NormMode': NormMode,  'hidden_sizes': hidden_sizes, 'n_outputs': n_outputs, 'input_size': input_size, 'NormPos': NormPos
              , 'Architecture': Architecture
              ,'ks':ks, 'ReLU_Slope': ReLU_Slope, 'Loss_function': Loss_function, 'IGB_Mode':IGB_Mode
              , 'train_classes': train_classes, 'valid_classes':valid_classes, 'num_data_points': {'Train': num_trdata_points, 'Eval':num_valdata_points}, 'label_list':label_list
              ,'epochs':epochs, 'num_tr_batches':num_tr_batches
              , 'GradNormMode': GradNormMode,
              'FolderPath': FolderPath,
              'TimeValSteps':TimeValSteps,
              'train_loader': train_loader, 'test_loader': test_loader, 'num_trdata_points': num_trdata_points, 'num_valdata_points':num_valdata_points,
              'SampleIndex': args.SampleIndex}


    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:1", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    params['device'] = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    
    #init wandb
    WandbInit(params)

    # Training
    train(args, model, params)


if __name__ == "__main__":
    main()
