import argparse
import os
import random
import time
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_n_classes
from libs.class_weight import get_class_weight, get_pos_weight
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import train, validate
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss, KLLoss
from libs.optimizer import get_optimizer
from libs.transformer import TempDownSamp, ToTensor
from libs.loss_fn.ECloss import EnergyConsistencyLoss

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action segmentation"
    )
    parser.add_argument("--dataset", type=str, default="TCG-15", help="name of the dataset")
    parser.add_argument("--result_path", type=str, default="./result", help="path of a result")
    parser.add_argument(
        "--seed", type=int, default=42, help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument("--cuda", type=int, default=5, help="cuda id")
    parser.add_argument(
        "--resume", action="store_true", help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def change_label_score(best_test, train_loss, epoch, cls_acc, edit_score, f1s):

    best_test['train_loss'] = train_loss
    best_test['epoch'] = epoch
    best_test['cls_acc'] = cls_acc
    best_test['edit'] = edit_score
    best_test['f1s@0.1'] = f1s[0]
    best_test['f1s@0.25'] = f1s[1]
    best_test['f1s@0.5'] = f1s[2]
    best_test['f1s@0.75'] = f1s[3]
    best_test['f1s@0.9'] = f1s[4]

def main() -> None:
    start_start = time.time()

    # argparser
    args = get_arguments()
    dataset_name = args.dataset
    device_num = args.cuda
    embedding_type = 'pool'
    # configuration
    config = get_config(f"config/{dataset_name}/config.yaml")  # get config.yaml

    # './result/LARA/split1'
    result_path = os.path.join(args.result_path, config.dataset, 'split' + str(config.split))

    print('\n---------------------------result_path---------------------------\n')
    print('result_path:',result_path) #'./result/LARA/DeST_tcn/split1'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(f'{result_path}/scores.txt', "w") as file:
        file.write(f'The result printed:\n')

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        device = device_num  # 0
        output_device = device_num[0] if type(device_num) is list else device_num
        torch.cuda.set_device(output_device)
        if type(device) is list:
            # CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_num))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = f'{device_num}'

        current_device = torch.cuda.current_device()
        print(f"Currently using GPU {current_device}")

    # Dataloader
    # Temporal downsampling is applied to only videos in LARA
    downsamp_rate = 4 if config.dataset == "LARA" else 1

    train_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="trainval" if not config.param_search else "training",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
        augmentations=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size, #8
        shuffle=True,
        num_workers=config.num_workers, #4
        drop_last=True if config.batch_size > 1 else False,
        collate_fn=collate_fn,
    )

    # if you do validation to determine hyperparams
    if config.param_search: #validation
        val_data = ActionSegmentationDataset(
            config.dataset,
            transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
            mode="test",
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
            augmentations=False,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    # load model
    print("---------- Loading Model ----------")

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir) #class_num

    action_embeddings = np.load(f'text_embeddings/{dataset_name}_actions_{embedding_type}.npy')
    action_embeddings = torch.from_numpy(action_embeddings).to(device)
    action_embedding_differences = (action_embeddings.unsqueeze(1) - action_embeddings.unsqueeze(0))**2
    action_embedding_distance = torch.sqrt(action_embedding_differences.sum(dim=-1))
    # action_embedding_difference = action_embeddings.unsqueeze(1) - action_embeddings.unsqueeze(0)
    # action_embedding_distance2 = torch.norm(action_embedding_difference,p=2,dim=-1)
    action_embeddings_distance_normalized = (action_embedding_distance - action_embedding_distance.min())/(action_embedding_distance.max() - action_embedding_distance.min())
    action_embeddings_graph = 1 - action_embeddings_distance_normalized

    joint_embeddings = np.load(f'text_embeddings/{dataset_name}_joints_{embedding_type}.npy')
    joint_embeddings = torch.from_numpy(joint_embeddings).to(device)
    joint_embedding_differences = (joint_embeddings.unsqueeze(1) - joint_embeddings.unsqueeze(0))**2
    joint_embedding_distance = torch.sqrt(joint_embedding_differences.sum(dim=-1))
    joint_embeddings_distance_normalized = (joint_embedding_distance - joint_embedding_distance.min())/(joint_embedding_distance.max() - joint_embedding_distance.min())
    joint_embeddings_graph = 1 - joint_embeddings_distance_normalized

    Model = import_class(config.model) #'libs.models.LaDy.Model'

    model = Model(
        in_channel=config.in_channel,
        n_features=config.n_features, #64
        n_classes=n_classes,
        n_stages=config.n_stages,
        n_layers=config.n_layers, #10
        n_refine_layers=config.n_refine_layers, #10
        n_stages_asb=config.n_stages_asb, #2
        n_stages_brb=config.n_stages_brb, #3
        SFI_layer=config.SFI_layer, #{1,2,3,4,5,6,7,8,9}
        dataset=config.dataset,
        node=config.node,
        dof=config.dof,
        num_people=config.num_people,
        use_Friction=config.use_Friction,
    )

    # send the model to cuda/cpu
    model.to(device)

    optimizer = get_optimizer(
        config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    ) #Adam or SGD

    # resume if you want
    columns = ["epoch", "lr", "train_loss"]

    # if you do validation to determine hyperparams
    if config.param_search:
        columns += ["val_loss", "cls_acc", "edit"]
        columns += [
            "f1s@{}".format(config.iou_thresholds[i])
            for i in range(len(config.iou_thresholds))
        ] #验证集的acc和edit以及f1值
        columns += ["bound_acc", "precision", "recall", "bound_f1s"]

    begin_epoch = 0
    best_loss = float("inf")

    # the best epoch's evaluation scores
    best_test_acc =  {'epoch':0,'train_loss':0,'cls_acc':0,'edit':0,'f1s@0.1':0,'f1s@0.25':0,'f1s@0.5':0,'f1s@0.75':0,'f1s@0.9':0}
    best_test_F1_10 =  best_test_acc.copy()
    best_test_F1_50 =  best_test_acc.copy()

    log = pd.DataFrame(columns=columns)
    # ['epoch', 'lr', 'train_loss', 'val_loss', 'cls_acc', 'edit', 'f1s@0.1', 'f1s@0.25', 'f1s@0.5', 'f1s@0.75', 'f1s@0.9', 'bound_acc', 'precision', 'recall', 'bound_f1s'] [Columns: [epoch, lr, train_loss, val_loss, cls_acc, edit, f1s@0.1, f1s@0.25, f1s@0.5, f1
    if args.resume: # checkpoint
        if os.path.exists(os.path.join(result_path, "checkpoint.pth")):
            checkpoint = resume(result_path, model, optimizer)
            begin_epoch, model, optimizer, best_loss = checkpoint
            log = pd.read_csv(os.path.join(result_path, "log.csv"))
            print("training will start from {} epoch".format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")

    #get_class_weight
    if config.class_weight:
        class_weight = get_class_weight(
            config.dataset,
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
            mode="training" if config.param_search else "trainval",
        )
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion_cls = ActionSegmentationLoss(
        ce=config.ce,
        focal=config.focal,
        tmse=config.tmse,
        gstmse=config.gstmse,
        weight=class_weight,
        ignore_index=255,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
        tmse_weight=config.tmse_weight,
        gstmse_weight=config.gstmse,
    ) #ce-loss and smooth-loss for action segmentation

    pos_weight = get_pos_weight(
        dataset=config.dataset,
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
        mode="training" if config.param_search else "trainval",
    ).to(device) #weight = boundary_frame_num/all_frame_num

    criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight) #bce-loss for boundary regression
    criterion_contrast = KLLoss().cuda(device)  # action-text contrastive loss
    criterion_mse = torch.nn.MSELoss()
    criterion_energy = EnergyConsistencyLoss(warmup_steps=300)       # ECloss (work-energy theorem)

    # train and validate model
    print("---------- Start training ----------")
    avg_cls_acc=0
    avg_edit_score=0
    avg_segment_f1s=[0,0,0,0,0]
    avg_bound_acc=0
    avg_precision=0
    avg_recall=0
    avg_bound_f1s=0

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()

        train_loss = train(
            train_loader,
            model,
            action_embeddings,
            action_embeddings_graph,
            joint_embeddings_graph,
            criterion_cls,
            criterion_bound,
            criterion_contrast,
            criterion_mse,
            criterion_energy,
            config.lambda_b,
            optimizer,
            device,
            dataset_name,
            epoch,
            config.ECloss_start_epoch,
            config.ECloss_weight,
        ) #读取的函数
        train_time = (time.time() - start) / 60

        # if you do validation to determine hyperparams
        if config.param_search:
            start = time.time()
            (
                val_loss,
                cls_acc,
                edit_score,
                segment_f1s,
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ) = validate(
                val_loader,
                model,
                joint_embeddings_graph,
                criterion_cls,
                criterion_bound,
                config.lambda_b,
                device,
                config.dataset,
                config.dataset_dir,
                config.iou_thresholds,
                config.boundary_th,
                config.tolerance,
                config.refinement_method,
            )
            if (epoch>=config.max_epoch-20):
                avg_cls_acc += cls_acc/20
                avg_edit_score += edit_score/20
                avg_segment_f1s = [a + b/20 for a, b in zip(avg_segment_f1s,segment_f1s)]
                avg_bound_acc += bound_acc/20
                avg_precision += precision/20
                avg_recall += recall/20
                avg_bound_f1s += bound_f1s/20

            if (epoch >0):
                # save a model if top1 cls_acc is higher than ever
                if best_loss > val_loss:
                    best_loss = val_loss

                if cls_acc > best_test_acc['cls_acc']:
                    change_label_score(best_test_acc, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_acc_model.prm')
                    )

                if segment_f1s[0] > best_test_F1_10['f1s@0.1']:
                    change_label_score(best_test_F1_10, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_F1_0.1_model.prm')
                    )

                if segment_f1s[2] > best_test_F1_50['f1s@0.5']:
                    change_label_score(best_test_F1_50, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_F1_0.5_model.prm')
                    )
 
        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss) #save .pth（contains epoch, model, optimizer, best_loss）

        # write logs to dataframe and csv file
        tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

        # if you do validation to determine hyperparams
        if config.param_search:
            tmp += [
                val_loss,
                cls_acc,
                edit_score,
            ]
            tmp += segment_f1s
            tmp += [
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ]

        tmp_df = pd.DataFrame(tmp, index=log.columns).T
        log = pd.concat([log, tmp_df], ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"))

        val_time = (time.time() - start) / 60


        eta_time = (config.max_epoch-epoch)*(train_time+val_time) #last time
        if config.param_search:
            # if you do validation to determine hyperparams
            print(
                'epoch: {}, lr: {:.4f}, train_time: {:.2f}min, train loss: {:.4f}, val_time: {:.2f}min, eta_time: {:.2f}min, \nval_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}, bound_acc: {:.2f}, bound_f1: {:.2f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss, val_time, eta_time, val_loss, cls_acc, \
                edit_score, segment_f1s[0],segment_f1s[1], segment_f1s[2],bound_acc,bound_f1s)
            )
            with open(f'{result_path}/scores.txt', "a+") as file:
                file.write(
                    'epoch: {}, lr: {:.4f}, train_time: {:.2f}min, train loss: {:.4f}, val_time: {:.2f}min, eta_time: {:.2f}min, \nval_loss: {:.4f}, acc: {:.2f}, edit: {:.2f}, F1@0.1: {:.2f}, F1@0.25: {:.2f}, F1@0.5: {:.2f}, bound_acc: {:.2f}, bound_f1: {:.2f}\n'
                    .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss, val_time, eta_time, val_loss, cls_acc, \
                    edit_score, segment_f1s[0],segment_f1s[1], segment_f1s[2],bound_acc,bound_f1s)
                )
        else:
            print(
                "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], train_loss
                )
            )



    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    print('\n---------------------------best_test_acc---------------------------\n')
    print('{}'.format(best_test_acc))
    print('\n---------------------------best_test_F1_10---------------------------\n')
    print('{}'.format(best_test_F1_10))
    print('\n---------------------------best_test_F1_50---------------------------\n')
    print('{}'.format(best_test_F1_50))
    print('\n---------------------------all_train_time---------------------------\n')
    print('all_train_time: {:.2f}min'.format((time.time() - start_start) / 60))

    with open(f'{result_path}/scores.txt', "a+") as file:
        file.write('\n---------------------------best_test_acc---------------------------\n')
        file.write('{}'.format(best_test_acc))
        file.write('\n---------------------------best_test_F1_10---------------------------\n')
        file.write('{}'.format(best_test_F1_10))
        file.write('\n---------------------------best_test_F1_50---------------------------\n')
        file.write('{}'.format(best_test_F1_50))
        file.write('\n---------------------------all_train_time---------------------------\n')
        file.write('all_train_time: {:.2f}min'.format((time.time() - start_start) / 60))

    print('avg_acc: {:.2f}, avg_edit: {:.2f}, avg_f1@10: {:.2f}, avg_f1@25: {:.2f}, avg_f1@50: {:.2f}, avg_bound_acc: {:.2f}, avg_precision: {:.2f}, avg_recall: {:.2f}, avg_bound_f1s: {:.2f}'
            .format(avg_cls_acc, avg_edit_score, avg_segment_f1s[0],avg_segment_f1s[1],avg_segment_f1s[2], avg_bound_acc, avg_precision, avg_recall, avg_bound_f1s)
         )

    with open(f'{result_path}/scores.txt', "a+") as file:
        file.write(
            'avg_acc: {:.2f}, avg_edit: {:.2f}, avg_f1@10: {:.2f}, avg_f1@25: {:.2f}, avg_f1@50: {:.2f}, avg_bound_acc: {:.2f}, avg_precision: {:.2f}, avg_recall: {:.2f}, avg_bound_f1s: {:.2f}\n'
            .format(avg_cls_acc, avg_edit_score, avg_segment_f1s[0],avg_segment_f1s[1],avg_segment_f1s[2], avg_bound_acc, avg_precision, avg_recall, avg_bound_f1s)
        )

    best_test_acc = pd.DataFrame.from_dict(best_test_acc, orient='index').T
    best_test_F1_10 = pd.DataFrame.from_dict(best_test_F1_10, orient='index').T
    best_test_F1_50 = pd.DataFrame.from_dict(best_test_F1_50, orient='index').T
    log = pd.concat([log, best_test_acc], ignore_index=True)
    log = pd.concat([log, best_test_F1_10], ignore_index=True)
    log = pd.concat([log, best_test_F1_50], ignore_index=True)
    log.to_csv(os.path.join(result_path, 'log.csv'), index=False)

    print("Done!")


if __name__ == "__main__":
    main()
