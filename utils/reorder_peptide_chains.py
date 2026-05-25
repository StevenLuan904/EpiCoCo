import torch
from builtins import enumerate, range, print, set, isinstance, len, ValueError, list
import logging

logger = logging.getLogger(__name__)

def reorder_peptide_chains(mol_info):
    """
    Reorder atoms in mol_info to form a single chain using the given bond_index.
    This function now operates on a compact peptide_mol_info.

    Args:
        mol_info (dict): A dictionary containing 'bond_index' and other molecular information.
            - bond_index: torch.Tensor of shape [2, num_bonds], representing bonds between atoms.

    Returns:
        dict: Updated mol_info with reordered atoms.
    """
    # 1. Ensure all relevant inputs are torch tensors
    for key in ['entity', 'element', 'atom_pos', 'atom_prob', 'bond_index']:
        if not isinstance(mol_info[key], torch.Tensor):
            mol_info[key] = torch.from_numpy(mol_info[key]).cpu()
        else:
            mol_info[key] = mol_info[key].cpu()

    # 2. Identify peptide atoms
    entity_mask = (mol_info['entity'] == 1)
    if not entity_mask.any():
        msg = "Warning: No peptide atoms found (entity == 1). Returning original mol_info."
        print(msg)
        logger.warning(msg)
        return mol_info

    # 3. Filter bonds to keep only those between peptide atoms
    bond_index = mol_info['bond_index']
    bond_entities = mol_info['entity'][bond_index]
    peptide_bond_mask = (bond_entities == 1).all(dim=0)
    peptide_bond_index = bond_index[:, peptide_bond_mask]

    # 4. Create a mapping from old indices to new, compact indices for peptide atoms
    peptide_indices = torch.where(entity_mask)[0]
    old_to_new_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(peptide_indices)}

    # 5. Remap the filtered bond indices to the new compact indices
    remapped_bond_index_list = []
    for i in range(peptide_bond_index.shape[1]):
        u, v = peptide_bond_index[0, i].item(), peptide_bond_index[1, i].item()
        if u in old_to_new_map and v in old_to_new_map:
            remapped_bond_index_list.append([old_to_new_map[u], old_to_new_map[v]])
    
    if not remapped_bond_index_list:
        remapped_bond_index = torch.empty((2, 0), dtype=torch.long)
    else:
        remapped_bond_index = torch.tensor(remapped_bond_index_list, dtype=torch.long).t()

    # 6. Create a new, compact mol_info containing only peptide information
    peptide_mol_info = {
        'allele_name': mol_info['allele_name'],
        'element': mol_info['element'][entity_mask],
        'atom_pos': mol_info['atom_pos'][entity_mask],
        'atom_prob': mol_info['atom_prob'][entity_mask],
        'entity': mol_info['entity'][entity_mask],
        'bond_index': remapped_bond_index,
        'pep_head_pos': mol_info.get('pep_head_pos') # Carry over the head position
    }

    # The rest of the function now operates on the compact peptide_mol_info
    bond_index = peptide_mol_info['bond_index']
    num_bonds = bond_index.shape[1]

    # Create adjacency list from bond_index
    adjacency_list = {}
    for i in range(num_bonds):
        start, end = bond_index[0, i].item(), bond_index[1, i].item()
        if start not in adjacency_list:
            adjacency_list[start] = []
        if end not in adjacency_list:
            adjacency_list[end] = []
        adjacency_list[start].append(end)
        adjacency_list[end].append(start)

    # Ensure all adjacency lists are unique
    for node in adjacency_list:
        adjacency_list[node] = list(set(adjacency_list[node]))

    # Find potential starting points (nodes with only one neighbor)
    endpoints = [node for node, neighbors in adjacency_list.items() if len(neighbors) == 1]
    
    start_node = None
    if len(endpoints) == 1:
        start_node = endpoints[0]
    elif len(endpoints) > 1:
        # If multiple endpoints, choose the one closest to the expected peptide head position
        pep_head_pos = peptide_mol_info.get('pep_head_pos')
        if pep_head_pos is not None:
            pep_head_pos = torch.from_numpy(pep_head_pos).float()
            endpoint_pos = peptide_mol_info['atom_pos'][endpoints]
            distances = torch.norm(endpoint_pos - pep_head_pos, dim=1)
            closest_endpoint_idx = torch.argmin(distances)
            start_node = endpoints[closest_endpoint_idx]
            msg = f"Info: Multiple endpoints found. Selected node {start_node} as start point based on proximity to pep_head_pos."
            print(msg)
            logger.info(msg)
        else:
            # Fallback if pep_head_pos is not available
            start_node = endpoints[0]
            msg = "Warning: Multiple endpoints found, but pep_head_pos not available. Arbitrarily choosing a start point."
            print(msg)
            logger.warning(msg)

    chain = []
    if start_node is not None:
        # Construct the chain by traversing the bonds
        visited = set()
        def construct_chain(node, prev_node):
            chain.append(node)
            visited.add(node)
            if node in adjacency_list:
                for neighbor in adjacency_list[node]:
                    if neighbor != prev_node and neighbor not in visited:
                        construct_chain(neighbor, node)
        construct_chain(start_node, None)

    # If bond-based traversal fails, use a distance-based greedy approach
    if len(chain) != len(peptide_mol_info['element']):
        num_atoms = len(peptide_mol_info['element'])
        if num_atoms > 1:
            msg = (f"Warning: Could not form a natural chain from bonds (found {len(chain)}/{num_atoms} atoms). "
                   f"Forcing a chain based on spatial distance.")
            print(msg)
            logger.warning(msg)

            atom_pos = peptide_mol_info['atom_pos']
            pep_head_pos = peptide_mol_info.get('pep_head_pos')

            if pep_head_pos is not None:
                # Start from the atom closest to the expected head position
                pep_head_pos = torch.from_numpy(pep_head_pos).float()
                distances_to_head = torch.norm(atom_pos - pep_head_pos, dim=1)
                start_node_dist = torch.argmin(distances_to_head).item()
                msg = f"Info: Using distance-based fallback. Starting from node {start_node_dist} closest to pep_head_pos."
                print(msg)
                logger.info(msg)
            else:
                # Fallback: find the two atoms that are furthest apart to serve as endpoints
                msg = "Warning: pep_head_pos not available for distance-based fallback. Starting from one of the two furthest atoms."
                print(msg)
                logger.warning(msg)
                dist_matrix_full = torch.cdist(atom_pos, atom_pos)
                max_dist_idx = torch.argmax(dist_matrix_full).item()
                start_node_dist = max_dist_idx // num_atoms

            # Calculate pairwise distances for greedy traversal
            dist_matrix = torch.cdist(atom_pos, atom_pos)

            # Greedy chain construction based on distance
            chain = [start_node_dist]
            visited = {start_node_dist}
            current_node = start_node_dist
            
            while len(chain) < num_atoms:
                # Find the nearest unvisited neighbor
                distances = dist_matrix[current_node].clone()
                # Mask visited nodes
                distances[list(visited)] = float('inf')
                
                next_node = torch.argmin(distances).item()
                
                if next_node in visited: # Should not happen with proper masking
                    break

                chain.append(next_node)
                visited.add(next_node)
                current_node = next_node
        else: # Only one atom, chain is just that atom
            chain = [0] if num_atoms == 1 else []

    if len(chain) != len(peptide_mol_info['element']):
        msg = f"Warning: Chain construction failed. Expected {len(peptide_mol_info['element'])}, got {len(chain)}. Skipping reordering."
        print(msg)
        logger.warning(msg)
        return peptide_mol_info

    # Reorder all relevant fields in peptide_mol_info based on the constructed chain
    chain = torch.tensor(chain, dtype=torch.long)
    peptide_mol_info['element'] = peptide_mol_info['element'][chain]
    peptide_mol_info['atom_pos'] = peptide_mol_info['atom_pos'][chain]
    peptide_mol_info['atom_prob'] = peptide_mol_info['atom_prob'][chain]

    # Update bond_index to reflect the new order
    chain_list = chain.tolist()
    old_to_new_reorder_map = {old: new for new, old in enumerate(chain_list)}
    
    final_bond_index_list = []
    for i in range(peptide_mol_info['bond_index'].shape[1]):
        u, v = peptide_mol_info['bond_index'][0, i].item(), peptide_mol_info['bond_index'][1, i].item()
        if u in old_to_new_reorder_map and v in old_to_new_reorder_map:
            new_u, new_v = old_to_new_reorder_map[u], old_to_new_reorder_map[v]
            final_bond_index_list.append([new_u, new_v])

    if not final_bond_index_list:
        peptide_mol_info['bond_index'] = torch.empty((2, 0), dtype=torch.long)
    else:
        peptide_mol_info['bond_index'] = torch.tensor(final_bond_index_list, dtype=torch.long).t()

    return peptide_mol_info