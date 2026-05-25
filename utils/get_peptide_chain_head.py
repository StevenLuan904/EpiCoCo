import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm


def _create_pep_head_pos_csv(pdb_dir, summary_path, output_csv_path, logger):
    """
    Scans PDB files, calculates the average starting position of peptide chains (chain 'P')
    for each allele, and saves the mapping to a CSV file.
    """
    logger.info("Creating peptide head position cache...")

    # 1. Read summary and create ID -> allele mapping
    try:
        summary_df = pd.read_csv(summary_path)
        id_to_allele = pd.Series(summary_df.allele.values, index=summary_df.ID).to_dict()
        valid_ids = set(summary_df['ID'])
    except FileNotFoundError:
        logger.error(f"Summary file not found at {summary_path}")
        return

    parser = PDBParser(QUIET=True)
    allele_positions = {}

    # 2. Iterate through PDB files and extract head positions
    if not os.path.exists(pdb_dir):
        logger.error(f"PDB directory not found: {pdb_dir}")
        return

    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    for filename in tqdm(pdb_files, desc="Processing PDBs"):
        pdb_id = os.path.splitext(filename)[0]
        if pdb_id in valid_ids:
            allele_name = id_to_allele[pdb_id]
            pdb_path = os.path.join(pdb_dir, filename)

            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                # Ensure chain P exists
                chains = {chain.id: chain for chain in structure.get_chains()}
                if 'P' in chains:
                    p_chain = chains['P']
                    # Get first residue safely
                    first_residue = next(p_chain.get_residues(), None)
                    if first_residue and 'CA' in first_residue:
                        pos = first_residue['CA'].get_coord()
                        if allele_name not in allele_positions:
                            allele_positions[allele_name] = []
                        allele_positions[allele_name].append(pos)
            except Exception as e:
                logger.warning(f"Could not process {filename}. Error: {e}")

    # 3. Calculate average positions and prepare for saving
    avg_positions_data = []
    for allele, positions in allele_positions.items():
        if positions:
            avg_pos = np.mean(positions, axis=0)
            avg_positions_data.append({
                'allele_name': allele,
                'pep_head_pos_x': avg_pos[0],
                'pep_head_pos_y': avg_pos[1],
                'pep_head_pos_z': avg_pos[2]
            })

    # 4. Save to CSV
    if avg_positions_data:
        avg_df = pd.DataFrame(avg_positions_data)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        avg_df.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully created peptide head position cache at {output_csv_path}")
    else:
        logger.warning("No valid peptide head positions found. Cache file not created.")


def get_peptide_chain_head(mol_info, logger):
    """
    Finds the average starting position for the peptide chain of a given allele.
    If the cache file does not exist, it will be created first.
    If the allele is missing in the cache, it attempts to calculate it from PDBs and update the cache.
    The position is then added to the mol_info dictionary.
    """
    # 定义项目根目录
    PDB_ROOT = "/data0/luanhaoyang/FlexStruct/EpiCoCo_v1"
    PROJECT_ROOT = "/data0/luanhaoyang/FlexStruct/EpiCoCo_v2"

    # 使用 os.path.join 拼接成绝对路径
    pdb_dir = os.path.join(PDB_ROOT, 'pdb_preprocess/pdb/')
    summary_path = os.path.join(PROJECT_ROOT, 'data/flex_pmhc/full_dataset.csv')
    cache_csv_path = os.path.join(PROJECT_ROOT, 'data/flex_pmhc/pep_head_pos.csv')

    # 1. Check for cache file and create if it doesn't exist (Bulk creation)
    if not os.path.exists(cache_csv_path):
        logger.info(f"Cache file not found at {cache_csv_path}. Creating from scratch...")
        _create_pep_head_pos_csv(pdb_dir, summary_path, cache_csv_path, logger)

    # 2. Read cache and find the position for the given allele
    try:
        # Load cache
        cache_df = pd.read_csv(cache_csv_path)
        # Create a quick lookup dict or use index
        cache_df.set_index('allele_name', inplace=True)

        allele_name = mol_info.get('allele_name', 'Unknown')

        if allele_name in cache_df.index:
            # Hit in cache
            pos_series = cache_df.loc[allele_name]
            pos = np.array([pos_series['pep_head_pos_x'], pos_series['pep_head_pos_y'], pos_series['pep_head_pos_z']])
            mol_info['pep_head_pos'] = pos
            # logger.info(f"Found peptide head position for {allele_name} in cache.") # Optional: reduce log spam
        else:
            # Miss in cache -> Attempt incremental update
            logger.warning(f"Allele '{allele_name}' not found in cache. Attempting to calculate from PDBs...")

            found_pos = None
            try:
                # A. Read summary to find PDB IDs for this allele
                full_summary = pd.read_csv(summary_path)
                if 'allele' in full_summary.columns and 'ID' in full_summary.columns:
                    target_pdbs = full_summary[full_summary['allele'] == allele_name]['ID'].values
                else:
                    target_pdbs = []

                # B. Scan specific PDBs
                parser = PDBParser(QUIET=True)
                positions = []

                for pdb_id in target_pdbs:
                    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
                    if os.path.exists(pdb_path):
                        try:
                            structure = parser.get_structure(pdb_id, pdb_path)
                            # Safe chain access
                            chains = {c.id: c for c in structure.get_chains()}
                            if 'P' in chains:
                                residue = next(chains['P'].get_residues(), None)
                                if residue and 'CA' in residue:
                                    positions.append(residue['CA'].get_coord())
                        except Exception:
                            continue  # Skip bad PDBs

                # C. Calculate mean if positions found
                if positions:
                    avg_pos = np.mean(positions, axis=0)
                    found_pos = avg_pos

                    # D. Update CSV (Append mode)
                    new_entry = pd.DataFrame([{
                        'allele_name': allele_name,
                        'pep_head_pos_x': avg_pos[0],
                        'pep_head_pos_y': avg_pos[1],
                        'pep_head_pos_z': avg_pos[2]
                    }])
                    # mode='a' adds to end, header=False avoids writing header again
                    new_entry.to_csv(cache_csv_path, mode='a', header=False, index=False)
                    logger.info(f"Calculated and cached new position for {allele_name}: {avg_pos}")

            except Exception as e:
                logger.error(f"Error during incremental cache update for {allele_name}: {e}")

            # E. Assign found position or default
            if found_pos is not None:
                mol_info['pep_head_pos'] = found_pos
            else:
                logger.warning(f"Could not calculate position for {allele_name}. Using default [0,0,0].")
                mol_info['pep_head_pos'] = np.array([0.0, 0.0, 0.0])

    except Exception as e:
        logger.error(f"Critical error accessing peptide head cache: {e}. Using default.")
        mol_info['pep_head_pos'] = np.array([0.0, 0.0, 0.0])

    return mol_info
