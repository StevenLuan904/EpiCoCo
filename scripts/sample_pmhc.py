# This version uses MHC as mhc condition
import os
import sys
import shutil
import argparse

sys.path.append('.')

import torch
import numpy as np
import torch.utils.tensorboard
from easydict import EasyDict
from Bio.PDB import PDBParser, PDBIO

from models.model import MolDiff
from models.bond_predictor import BondPredictor
from utils.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.train import *
from utils.dataset import get_dataset
from datetime import datetime
from torch_geometric.loader import DataLoader
from utils.reorder_peptide_chains import reorder_peptide_chains
from utils.get_peptide_chain_head import get_peptide_chain_head

AMINO_ACID = "ACDEFGHIKLMNPQRSTVWY"


def peptide_indices_to_sequence(indices):
    """Convert peptide indices to amino acid sequence.

    Accepts a 1-D sequence of indices (list/np.array/torch.Tensor) and returns
    the corresponding single-letter amino-acid string. Filters out invalid
    indices and the mask token 'X' if present in AMINO_ACID mapping.
    """
    sequence = ""
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    for idx in indices:
        # ensure idx is an int-like and in valid range
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(AMINO_ACID):
            sequence += AMINO_ACID[idx]
    return sequence


def save_csv(sample, log_dir, logger):
    """Save allele name and peptide sequence to CSV."""
    import csv
    csv_path = os.path.join(log_dir, 'gen_pep_seq.csv')

    file_exists = os.path.exists(csv_path)
    if not file_exists:
        logger.info(f'Creating peptide sequences file at {csv_path}')

    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='') as csvfile:
        fieldnames = ['allele', 'peptide']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        peptide_sequence = peptide_indices_to_sequence(sample['element'])
        writer.writerow({'allele': sample['allele_name'], 'peptide': peptide_sequence})


def save_pdb(sample, out_dir, logger, postfix='pdb'):
    # sample is a single peptide molecule
    import os
    from datetime import datetime
    import torch

    # pdb都存到./pdb/文件夹中
    pdb_out_dir = os.path.join(out_dir, postfix)
    os.makedirs(pdb_out_dir, exist_ok=True)

    # 单字母氨基酸 → 三字母氨基酸
    aa_map = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }

    # Ensure tensors are on the CPU before processing
    entity = sample['entity']
    element = sample['element']
    atom_pos = sample['atom_pos']

    if isinstance(entity, torch.Tensor):
        entity = entity.cpu().numpy()
    if isinstance(element, torch.Tensor):
        element = element.cpu().numpy()
    if isinstance(atom_pos, torch.Tensor):
        atom_pos = atom_pos.cpu().numpy()

    # Convert indices to sequence
    # 假设你外部已经定义了 peptide_indices_to_sequence 函数
    sequence = peptide_indices_to_sequence(element)

    # Sanitize allele_name to remove special characters
    if 'allele_name' in sample:
        sanitized_allele_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in sample['allele_name'])
    else:
        sanitized_allele_name = "unknown_allele"

    # 文件名：sanitized_allele_name+当前时间（到秒）+唯一标识
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{sanitized_allele_name}_{timestamp}"
    pdb_name = f"{base_name}.pdb"
    pdb_path = os.path.join(pdb_out_dir, pdb_name)

    # 确保文件名唯一
    counter = 1
    while os.path.exists(pdb_path):
        pdb_name = f"{base_name}_{counter}.pdb"
        pdb_path = os.path.join(pdb_out_dir, pdb_name)
        counter += 1

    # 初始化每个链的残基计数器，这样 Chain P 可以从 1 开始编号
    chain_res_counters = {'M': 0, 'P': 0}

    with open(pdb_path, "w") as f:
        # [修改点]: zip 中加入 entity，同时获取氨基酸、坐标和实体类型
        for j, (aa, pos, ent) in enumerate(zip(sequence, atom_pos, entity)):
            x, y, z = pos
            ent_val = int(ent)  # 确保转为整数

            # [修改点]: 根据 entity 判断链 ID
            if ent_val == 0:
                chain_id = 'M'
            elif ent_val == 1:
                chain_id = 'P'
            else:
                chain_id = 'A'  # 默认兜底

            # 更新该链的残基编号
            if chain_id not in chain_res_counters:
                chain_res_counters[chain_id] = 0
            chain_res_counters[chain_id] += 1
            res_seq_num = chain_res_counters[chain_id]

            resname = aa_map.get(aa, "UNK")  # 转三字母残基名

            # 标准 PDB 行：
            # 列 22: Chain identifier
            # 列 23-26: Residue sequence number
            # 这里 j+1 是原子序号(serial)，res_seq_num 是残基序号
            f.write(f"ATOM  {j+1:5d}  CA  {resname:>3s} {chain_id}{res_seq_num:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.00:6.2f}{20.00:6.2f}           C\n")
        f.write("END\n")

    logger.info(f"Saved PDB to {pdb_path}")


# keep only for pep-pep bonds
# print bond connections
# sample and entity are both single molecule
def filter_bond_connections(sample, logger, entity):
    if 'bond_index' in sample:
        bond_index = sample['bond_index']
        if isinstance(bond_index, torch.Tensor):
            bond_index = bond_index.cpu().numpy()
        if isinstance(entity, torch.Tensor):
            entity = entity.cpu().numpy()
        start_nodes = bond_index[0]
        end_nodes = bond_index[1]
        is_pep_pep_bond = (entity[start_nodes] == 1) & (entity[end_nodes] == 1)
        # Filter the bond_index
        filtered_bond_index = bond_index[:, is_pep_pep_bond]
        sample['bond_index'] = filtered_bond_index
        logger.info("Peptide-Peptide Bonds:")
        for start, end in zip(filtered_bond_index[0], filtered_bond_index[1]):
            logger.info(f"  Bond between atom {start+1} and atom {end+1}")


def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (len(pool.finished), len(pool.failed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_pmhc.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # # Loggingtrain_config
    log_root = args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    # Transforms
    logger.info('Loading data placeholder...')
    featurizer = FeaturizeMol(
        train_config.chem.atomic_numbers,
        train_config.chem.mol_bond_types,
        use_mask_node=train_config.transform.use_mask_node,
        use_mask_edge=train_config.transform.use_mask_edge,
    )
    transform = Compose([
        featurizer,
    ])
    # # Load dataset for allele names
    logger.info('Loading dataset...')
    dataset = get_dataset(
        config=config.dataset,
        transform=transform,
    )

    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
        model = MolDiff(config=train_config.model,
                        num_node_types=featurizer.num_node_types,
                        num_edge_types=featurizer.num_edge_types).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()

    pool = EasyDict({
        'failed': [],
        'finished': [],
    })

    batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
    # 创建DataLoader实例
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             follow_batch=featurizer.follow_batch,
                             exclude_keys=featurizer.exclude_keys)

    # 遍历一个epoch的所有batch
    logger.info("开始遍历一个epoch的所有batch...")
    for i, batch in enumerate(data_loader):
        # 将batch移动到GPU
        batch = batch.to(args.device)

        # 处理batch逻辑
        logger.info(f"正在处理第 {i+1} 个batch, 包含 {batch.num_graphs} 个图。")

        # 示例：调用make_data_placeholder处理batch
        batch_holder = make_data_placeholder(loader=[batch], device=args.device)

        # Unpack batch_holder
        allele_name = batch_holder['allele_name']
        entity = batch_holder['entity']
        halfedge_entity = batch_holder['halfedge_entity']
        batch_node = batch_holder['node_type_batch']
        halfedge_index = batch_holder['halfedge_index']
        halfedge_type = batch_holder['halfedge_type']
        batch_halfedge = batch_holder['halfedge_type_batch']
        node_index = batch_holder['node_type']
        node_pos = batch_holder['node_pos']
        actual_n_graphs = batch_holder['actual_n_graphs']
        mhc_node_mask = (entity == 0)
        mhc_halfedge_mask = (halfedge_entity == 0)

        mhc = EasyDict({
            'node_pos': node_pos[mhc_node_mask],
            'node_type': node_index[mhc_node_mask],
            'batch_node': batch_node[mhc_node_mask],
            'halfedge_type': halfedge_type[mhc_halfedge_mask],
            'batch_halfedge': batch_halfedge[mhc_halfedge_mask],
            'num_atoms': mhc_node_mask.sum().item(),
            'mhc_node_mask': mhc_node_mask,
            'mhc_halfedge_mask': mhc_halfedge_mask,
        })

        # inference
        outputs = model.sample(
            n_graphs=actual_n_graphs,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            halfedge_type=halfedge_type,
            batch_halfedge=batch_halfedge,
            node_index=node_index,
            node_pos=node_pos,
            entity=entity,
            halfedge_entity=halfedge_entity,
            # No ic50 input during sampling
            esm=batch.esm,
            esm_avg=batch.esm_avg,
            is_resample=config.sample.resample,
            scaffold=mhc,
            guidance_scale=config.sample.guidance_scale)

        outputs = {key: [v.cpu().numpy() for v in value] for key, value in outputs.items()}

        # decode outputs to molecules
        batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(
        ), batch_halfedge.cpu().numpy()
        output_list = seperate_outputs(outputs, actual_n_graphs, batch_node, halfedge_index, batch_halfedge)
        for i_mol, output_mol in enumerate(output_list):
            # mol_info is the current pmhc object
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )  # note: traj is not used
            mol_info['allele_name'] = allele_name[i_mol]
            # entity is in whole batch
            mol_info['entity'] = entity[batch_node == i_mol]

            save_pdb(mol_info, log_dir, logger, postfix='raw_pdb')

            filter_bond_connections(mol_info, logger, mol_info['entity'])

            mol_info = get_peptide_chain_head(mol_info, logger)

            # Reorder peptide chains based on edge_index
            mol_info = reorder_peptide_chains(mol_info)

            save_csv(mol_info, log_dir, logger)
            save_pdb(mol_info, log_dir, logger)
            pool.finished.append(mol_info)

        print_pool_status(pool, logger)
    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
