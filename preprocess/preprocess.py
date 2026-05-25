import os
import sys
import pickle
import argparse
import logging
import torch
import numpy as np
import pandas as pd  # [新增] 引入pandas读取csv
from Bio.PDB import PDBParser
from tqdm import tqdm
import esm

# ================= 1. 日志与基础类定义 =================


def setup_logger():
    """配置带时间戳和层级的日志记录器"""
    logger = logging.getLogger("pMHC_Pipeline")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


class pMHC_type:
    """
    pMHC结构数据类
    attributes:
        resi: List[str], 残基单字母列表 (Filtered M + All P)
        pos: List[List[float]], CA原子坐标列表
        entity: List[int], 链标识 (0 for M, 1 for P)
    """

    def __init__(self, resi, pos, entity):
        self.resi = resi
        self.pos = pos
        self.entity = entity

    def __repr__(self):
        return f"<pMHC_type: {len(self.resi)} residues (M:{self.entity.count(0)}, P:{self.entity.count(1)})>"


def three_to_one(resname):
    """三字母转单字母氨基酸"""
    mapping = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y"
    }
    return mapping.get(resname.upper(), "X")


# ================= 2. 核心处理逻辑 =================


def extract_structural_info(pdb_path, threshold):
    """
    解析PDB，执行距离筛选逻辑。
    
    更改说明：
    - 支持将链 'A' 识别为 M 链（若 'M' 不存在则尝试 'A'）。
    - 当 P 链缺失时仍然返回结构，但跳过基于距离的筛选（等效于保留全部 M 残基）。
    - 如果 P 链缺失，会附加一个合成 P 链（9 个 CA 原点坐标）以保持下游数据结构一致性；注意合成 P 不会用于恢复距离筛选（p_present 仍为 False）。

    返回：combined_obj, full_m_seq, pocket_indices, p_present（bool）
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", pdb_path)
    except Exception as e:
        logger.error(f"解析错误 {os.path.basename(pdb_path)}: {e}")
        return None, None, None, False

    model = structure[0]

    # --- 步骤 A: 提取 M 链（支持 'M' 或 'A'）---
    m_residues = []
    m_chain_name = None
    for name in ('M', 'A'):
        if name in model:
            m_chain = sorted([r for r in model[name] if 'CA' in r], key=lambda r: r.get_id()[1])
            m_chain_name = name
            break
    if m_chain_name:
        for res in m_chain:
            code = three_to_one(res.get_resname())
            coord = res['CA'].get_coord()
            m_residues.append({'code': code, 'coord': coord})
    else:
        # 没有 M 或 A 链，无法继续处理
        return None, None, None, False

    # --- 步骤 B: 提取 P 链（可缺失）---
    p_residues = []
    p_present = False
    if 'P' in model:
        p_chain = sorted([r for r in model['P'] if 'CA' in r], key=lambda r: r.get_id()[1])
        for res in p_chain:
            code = three_to_one(res.get_resname())
            coord = res['CA'].get_coord()
            p_residues.append({'code': code, 'coord': coord})
        if p_residues:
            p_present = True
    else:
        logger.warning(
            f"{os.path.basename(pdb_path)}: P chain not found, skipping distance-based pocket selection; keeping all M residues."
        )
        # 添加合成的 P 链以便下游流程（例如构建 combined_obj），但保留 p_present=False
        p_residues = []
        for i in range(9):
            # 使用任意残基类型并将 CA 坐标设为原点
            p_residues.append({'code': 'A', 'coord': np.array([0.0, 0.0, 0.0])})
        logger.info(f"{os.path.basename(pdb_path)}: Added synthetic P chain with 9 CA atoms at origin.")

    # --- 步骤 C: 计算 pocket_indices ---
    if not p_present:
        # 原始 P 链缺失 -> 按设计不进行基于距离的筛选，保留全部 M 残基
        pocket_indices = list(range(len(m_residues)))
    else:
        if threshold == -1.0:
            pocket_indices = list(range(len(m_residues)))
        else:
            m_coords = np.array([r['coord'] for r in m_residues])
            p_coords = np.array([r['coord'] for r in p_residues])

            # 计算 M 中每个原子到 P 中所有原子的最小距离
            dists = np.linalg.norm(m_coords[:, None, :] - p_coords[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)  # (Num_M,)

            pocket_indices = np.where(min_dists < threshold)[0].tolist()

    # --- 步骤 D: 构建输出数据 ---
    # 筛选后的 M 链部分
    m_resi_filt = [m_residues[i]['code'] for i in pocket_indices]
    m_pos_filt = [m_residues[i]['coord'].tolist() for i in pocket_indices]
    m_entity_filt = [0] * len(pocket_indices)

    # 完整的 P 链部分
    p_resi_all = [r['code'] for r in p_residues]
    p_pos_all = [r['coord'].tolist() for r in p_residues]
    p_entity_all = [1] * len(p_residues)

    combined_obj = pMHC_type(resi=m_resi_filt + p_resi_all,
                             pos=m_pos_filt + p_pos_all,
                             entity=m_entity_filt + p_entity_all)

    full_m_seq = "".join([r['code'] for r in m_residues])

    return combined_obj, full_m_seq, pocket_indices, p_present


# ================= 3. 主程序 =================


def main():
    parser = argparse.ArgumentParser(description="Process PDBs: Filter Structure & Extract Pocket Embeddings")
    parser.add_argument("--input_dir",
                        type=str,
                        default="/data0/luanhaoyang/FlexStruct/pMHCDiff_v1/pdb_preprocess/pdb",
                        help="PDB目录")
    parser.add_argument("--output_dir",
                        type=str,
                        default="/data0/luanhaoyang/FlexStruct/pMHCDiff_v2/data/flex_pmhc/",
                        help="输出目录")
    parser.add_argument("--csv_path",
                        type=str,
                        default="/data0/luanhaoyang/FlexStruct/pMHCDiff_v2/data/flex_pmhc/full_dataset.csv",
                        help="Metadata CSV 路径")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/data0/luanhaoyang/FlexStruct/pMHCDiff_v2/preprocess/logs/",
                        help="日志目录")
    parser.add_argument("--threshold", type=float, default=-1.0, help="原子距离阈值 (Angstrom)，-1.0表示跳过筛选保留全部")
    parser.add_argument("--num", type=int, default=-1, help="处理文件数量限制 (-1为全部)")
    parser.add_argument("--batch_size", type=int, default=8, help="ESM推理Batch Size")
    args = parser.parse_args()

    # 路径检查
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志文件，添加时间戳到文件名
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(args.log_dir, f"pdb_esm_pipeline_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    logger.info(f"日志文件已配置: {log_file_path}")

    # ================= [新增] 步骤 0: 读取 CSV 并筛选正样本 =================
    logger.info(">>> Phase 0: 读取 CSV 并筛选样本 ID...")
    valid_ids = set()
    try:
        if os.path.exists(args.csv_path):
            df_summary = pd.read_csv(args.csv_path)
            # 不再筛选正样本，直接提取所有 ID
            valid_ids = set(df_summary['ID'].astype(str).values)
            logger.info(f"CSV 加载成功。样本总数: {len(valid_ids)}")
        else:
            logger.error(f"CSV 文件未找到: {args.csv_path}")
            return
    except Exception as e:
        logger.error(f"读取 CSV 失败: {e}")
        return

    # 获取 PDB 文件列表并进行过滤
    all_pdb_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pdb")]
    pdb_files = []
    skipped_files = 0

    for f in all_pdb_files:
        file_id = f.split('.')[0]
        if file_id in valid_ids:
            pdb_files.append(f)
        else:
            skipped_files += 1

    if args.num > 0:
        pdb_files = pdb_files[:args.num]

    logger.info(f"PDB 筛选结果: 总文件 {len(all_pdb_files)} -> 匹配样本 {len(pdb_files)} (跳过 {skipped_files})")
    threshold_str = "All atoms" if args.threshold == -1.0 else f"{args.threshold}Å"
    logger.info(f"配置: Threshold={threshold_str}, BatchSize={args.batch_size}")

    # ------ Phase 1: 结构处理与筛选 ------
    structure_data = {}  # 结果字典
    sequence_batch_list = []  # 待推理列表
    indices_map = {}  # 索引映射

    logger.info(">>> Phase 1: 读取PDB并执行几何筛选...")

    p_present_count = 0
    for fname in tqdm(pdb_files, desc="Parsing Structure"):
        pdb_id = fname.split(".")[0]
        fpath = os.path.join(args.input_dir, fname)

        pmhc_obj, m_seq, indices, p_present = extract_structural_info(fpath, args.threshold)

        if pmhc_obj:
            structure_data[pdb_id] = pmhc_obj
            sequence_batch_list.append((pdb_id, m_seq))
            indices_map[pdb_id] = indices
            if p_present:
                p_present_count += 1

    valid_count = len(structure_data)
    logger.info(f"结构提取完成: {valid_count}/{len(pdb_files)} 个有效文件。")
    if p_present_count == 0:
        logger.info("注意: 在处理的 PDB 中未检测到任何 P 链，已退回为仅 M-chain 模式 (不执行原子距离筛选)。")

    # ------ Phase 2: ESM Embedding 推理与切片 ------
    logger.info(">>> Phase 2: ESM模型推理与Embedding筛选...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ESM-2 model on {device}...")

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embedding_data = {}  # 结果字典
    sequence_embedding_data = {}  # 存储每序列的 averaged 表示

    num_batches = (valid_count + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches), desc="ESM Inference"):
        batch = sequence_batch_list[i * args.batch_size:(i + 1) * args.batch_size]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        tokens = batch_tokens.to(device)

        if i == 0:
            logger.info(f"[Shape Check] Input Tokens: {tokens.shape} (Batch, SeqLen+Pad)")

        with torch.no_grad():
            results = model(tokens, repr_layers=[6], return_contacts=False)

        token_reps = results["representations"][6]

        # token_reps: (batch, seq_len_padded, dim)
        # Compute per-sequence representations by averaging residue token embeddings
        # Note: ESM tokenization adds BOS and EOS tokens, so actual token length = len(seq) + 2
        token_representations = token_reps
        sequence_representations = []
        for bi, tokens_len in enumerate(batch_lens):
            # slice tokens 1 .. tokens_len-2 (exclude BOS and EOS), then average
            seq_repr = token_representations[bi, 1:tokens_len - 1].mean(0)
            sequence_representations.append(seq_repr)
        # stack into a single tensor (B, D)
        sequence_representations = torch.stack(sequence_representations, dim=0)

        if i == 0:
            logger.info(f"[Shape Check] Raw Embeddings: {token_reps.shape} (Batch, SeqLen+Pad, Dim)")
            logger.info(
                f"[Shape Check] Sequence representations (averaged): {sequence_representations.shape} (Batch, Dim)")

        for j, (pid, seq) in enumerate(batch):
            seq_len = len(seq)
            # 1. 提取有效序列 Embedding
            full_emb = token_reps[j, 1:seq_len + 1]

            # 2. 根据结构筛选步骤得到的索引进行切片
            inds = indices_map[pid]

            if len(inds) > 0:
                inds_tensor = torch.tensor(inds, device=device)
                pocket_emb = full_emb.index_select(0, inds_tensor)
            else:
                pocket_emb = torch.empty((0, 320), device=device)

            if i == 0 and j == 0:
                logger.info(f"[Shape Check] Filtered Embedding ({pid}): {pocket_emb.shape}")
                logger.info(f"  -> Should match M-atom count in structure: {structure_data[pid].entity.count(0)}")
            embedding_data[pid] = pocket_emb.cpu().half()
            # store averaged sequence representation for pid (consistent storage format)
            sequence_embedding_data[pid] = sequence_representations[j].cpu().half()

    # ------ Phase 3: 保存文件与详细验证 ------
    logger.info(">>> Phase 3: 保存 Pickle 文件与数据验证...")

    # If no P chains were present in any processed PDB, use the mhc_all suffix (no distance-based filtering possible)
    if 'p_present_count' in locals() and p_present_count == 0:
        suffix = f"mhc_all_num{valid_count}"
    elif args.threshold == -1.0:
        suffix = f"all_num{valid_count}"
    else:
        suffix = f"radius{args.threshold}_num{valid_count}"

    struct_out = os.path.join(args.output_dir, f"pdb_pmhc_{suffix}.pickle")
    with open(struct_out, "wb") as f:
        pickle.dump(structure_data, f)
    logger.info(f"Structure Data Saved: {struct_out}")

    embed_out = os.path.join(args.output_dir, f"esm_pmhc_{suffix}.pickle")
    with open(embed_out, "wb") as f:
        pickle.dump(embedding_data, f)
    logger.info(f"Embedding Data Saved: {embed_out}")

    # 额外保存每序列的 averaged 表示，使用相同的路径/后缀规则
    seqavg_out = os.path.join(args.output_dir, f"esm_pmhc_{suffix}_avg.pickle")
    with open(seqavg_out, "wb") as f:
        pickle.dump(sequence_embedding_data, f)
    logger.info(f"Averaged sequence representations saved: {seqavg_out}")

    # 3. 输出真实的保存形状和示例内容 (满足Check 1)
    logger.info("\n" + "=" * 60)
    logger.info("               OUTPUT DATA VERIFICATION               ")
    logger.info("=" * 60)
    logger.info(f"Total Entries Processed: {len(structure_data)}")

    example_keys = list(structure_data.keys())[:2]

    for idx, key in enumerate(example_keys):
        struct_obj = structure_data[key]
        embed_tensor = embedding_data[key]

        num_m_atoms = struct_obj.entity.count(0)
        num_p_atoms = struct_obj.entity.count(1)

        logger.info(f"\n[Example {idx+1} | PDB_ID: {key}]")

        logger.info(f"  [Structure Data]")
        logger.info(f"    Composition:    M-chain (Pocket) = {num_m_atoms}, P-chain (All) = {num_p_atoms}")

        logger.info(f"  [Embedding Data]")
        logger.info(f"    Shape: {embed_tensor.shape}  <-- [Num_Pocket_Residues, Hidden_Dim]")

        if embed_tensor.shape[0] == num_m_atoms:
            logger.info(f"    Status: \033[92m[MATCH]\033[0m Embedding rows align with Structure M-atoms.")
        else:
            logger.warning(
                f"    Status: \033[91m[MISMATCH]\033[0m Embed rows {embed_tensor.shape[0]} != Struct M-atoms {num_m_atoms}"
            )

    # Print averaged sequence representation summary (no validation)
    if sequence_embedding_data:
        try:
            n_seq = len(sequence_embedding_data)
            # get dimension from first item
            first_vec = next(iter(sequence_embedding_data.values()))
            dim = tuple(first_vec.shape)
            # compute mean vector safely (use float for accumulation)
            acc = None
            for v in sequence_embedding_data.values():
                vf = v.float()
                if acc is None:
                    acc = vf.clone()
                else:
                    acc += vf
            mean_vec = (acc / float(n_seq)).cpu().numpy()
            logger.info(f"\n[Average Sequence Embeddings] Count: {n_seq} | Dim: {dim}")
            # print a short prefix of mean vector for quick check
            prefix = mean_vec.ravel()[:8]
            logger.info(f"  Mean vector (first 8 values): {prefix}")
        except Exception as e:
            logger.warning(f"Could not compute averaged embedding summary: {e}")

    logger.info("=" * 60)
    logger.info("Pipeline Finished.")


if __name__ == "__main__":
    main()
