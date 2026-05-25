import os
import numpy as np
import torch

def parse_conf_list(pmhc, config=None, esm=None, esm_avg=None, ic50=None):
    """
    Parses a pMHC object into a dictionary containing structural and feature information.
    Merged functionality of original parse_conf_list and parse_drug3d_mol.
    """
    
    # 1. Extract and convert basic attributes to numpy arrays
    # Using np.array initially to allow for easy boolean masking if filtering is required
    resi = np.array(pmhc.resi)
    pos = np.array(pmhc.pos, dtype=np.float32)
    entity = np.array(pmhc.entity, dtype=np.int64)
    
    # 2. Filter for peptide only if specified in config
    if config is not None and config.get('use_data') == 'peptide':
        peptide_mask = entity == 1
        resi = resi[peptide_mask]
        pos = pos[peptide_mask]
        entity = entity[peptide_mask]
        
    num_atoms = len(resi)

    # 3. Map residue types to integers
    residue2id = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    # Map residue types, assuming all residues are in the dictionary keys
    element = np.array([residue2id[r] for r in resi], dtype=np.int64)

    # 4. Build Bonds
    # Logic: Connect adjacent residues (i, i+1) only if they are both part of the Peptide chain (entity == 1).
    # Identify indices i where entity[i] == 1 AND entity[i+1] == 1
    if num_atoms > 1:
        bond_mask = (entity[:-1] == 1) & (entity[1:] == 1)
        peptide_start_indices = np.where(bond_mask)[0]
    else:
        peptide_start_indices = np.array([], dtype=np.int64)

    if len(peptide_start_indices) > 0:
        # Create bidirectional edges for undirected graph
        # row: i -> i+1, col: i+1 -> i
        row = np.repeat(peptide_start_indices, 2) + np.tile([0, 1], len(peptide_start_indices))
        col = np.repeat(peptide_start_indices, 2) + np.tile([1, 0], len(peptide_start_indices))
        bond_type_list = np.ones(len(row), dtype=np.int64)
    else:
        row = np.array([], dtype=np.int64)
        col = np.array([], dtype=np.int64)
        bond_type_list = np.array([], dtype=np.int64)

    bond_index = np.array([row, col], dtype=np.int64)
    bond_type = np.array(bond_type_list, dtype=np.int64)

    # 5. Sort Bonds
    # Sort based on source node then target node for canonical representation
    if bond_index.shape[1] > 0:
        perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
        bond_index = bond_index[:, perm]
        bond_type = bond_type[perm]

    # Calculate number of unique bonds (undirected count)
    num_bonds = bond_index.shape[1] // 2 

    # 6. Process ESM embeddings
    esm_out = np.array(esm, dtype=np.float32) if esm is not None else None
    esm_avg = esm_avg.unsqueeze(0) if esm_avg is not None else None
    esm_avg_out = np.array(esm_avg, dtype=np.float32) if esm_avg is not None else None

    # 7. Construct and return final dictionary
    # pos_all_confs is expected to be shape [num_confs, num_atoms, 3]
    return {
        'element': element,
        'bond_index': bond_index,
        'bond_type': bond_type,
        'pos_all_confs': np.array([pos], dtype=np.float32),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'i_conf_list': [0],
        'num_confs': 1,
        'entity': entity,
        'esm': esm_out,
        'esm_avg': esm_avg_out,
        'ic50': ic50  # [新增] 添加 IC50 测量值
    }