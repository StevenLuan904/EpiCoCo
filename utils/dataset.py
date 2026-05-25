import pickle
import os

import torch
import numpy as np
import pandas as pd
import lmdb
from tqdm import tqdm
import copy
from collections import defaultdict

from torch.utils.data import Subset, Dataset
from .parser import parse_conf_list
from .data import Drug3DData, torchify_dict
from utils.pMHC_type import pMHC_type


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.root
    force_reload = config.get('force_reload', False)

    if name == 'pmhc':
        # 这里 config 实际上是 config.dataset 部分
        dataset = Drug3DDataset(root, config.path_dict, force_reload=force_reload, config=config, *args, **kwargs)
    elif name == 'pmhc_sample':
        dataset = Drug3DDataset(root, config.path_dict, force_reload=force_reload, config=config, *args, **kwargs)
        dataset = modify_dataset(dataset)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    return dataset


class Drug3DDataset(Dataset):

    def __init__(self, root, path_dict, force_reload=True, config=None, transform=None):
        super().__init__()
        self.root = root
        self.summary_path = os.path.join(root, path_dict['summary'])
        self.struct_path = os.path.join(root, path_dict['struct'])
        self.esm_path = os.path.join(root, path_dict['esm'])
        self.esm_avg_path = os.path.join(root, path_dict['esm_avg'])

        self.processed_path = os.path.join(root, path_dict['processed'])
        self.id2idx_path = self.processed_path[:self.processed_path.find('.lmdb')] + '_id2idx.pt'

        self.config = config
        self.force_reload = force_reload
        self.transform = transform
        self.db = None
        self.keys = None
        self.is_sample = config.name == 'pmhc_sample'
        if force_reload:
            if os.path.exists(self.processed_path):
                os.remove(self.processed_path)
            if os.path.exists(self.id2idx_path):
                os.remove(self.id2idx_path)
        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.id2idx_path)) or force_reload:
            self._process()
            self._precompute_id2idx()
        self.id2idx = torch.load(self.id2idx_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        # read summary
        df_summary = pd.read_csv(self.summary_path)

        # Load struct data
        struct_data = pickle.load(open(self.struct_path, 'rb'))

        # [新增] 3. 加载 ESM Embedding 数据
        print(f"Loading ESM embeddings from {self.esm_path}...")
        esm_data = pickle.load(open(self.esm_path, 'rb'))

        # [新增] 4. 加载 ESM_AVG Embedding 数据
        print(f"Loading ESM_AVG embeddings from {self.esm_avg_path}...")
        esm_avg_data = pickle.load(open(self.esm_avg_path, 'rb'))

        # 辅助函数：查找数据 (增加了 esm_avg_db 和 ic50)
        def find_data(entry_id, struct_db, esm_db, esm_avg_db):
            s = struct_db.get(entry_id, None)
            e = esm_db.get(entry_id, None)
            ea = esm_avg_db.get(entry_id, None)  # [新增]
            return s, e, ea

        valid_entries = []
        num_skipped = 0
        for _, line in tqdm(df_summary.iterrows(), total=len(df_summary), desc='Finding structures'):
            allele = line['allele']
            if allele != "HLA-A*02:01":
                continue
            entry_id = line['ID']
            ic50 = line['measurement_value']  # [新增] 获取 IC50 测量值
            # [修改] 同时获取结构、IC50、 ESM 和 ESM_AVG 数据
            current_struct, current_esm, current_esm_avg = find_data(entry_id, struct_data, esm_data, esm_avg_data)

            # [修改] 必须四者都存在才算有效数据
            if current_struct is None or current_esm is None or current_esm_avg is None or ic50 is None:
                num_skipped += 1
                # 可选：打印具体缺失了哪部分
                # if current_struct is None: print(f"Missing struct: {entry_id}")
                # if current_esm is None: print(f"Missing ESM: {entry_id}")
                # if current_esm_avg is None: print(f"Missing ESM_AVG: {entry_id}")
                continue

            # [修改] 将 esm、esm_avg 和 ic50 也加入到 entry 字典中
            valid_entries.append({
                'line': line,
                'struct': current_struct,
                'esm': current_esm,
                'esm_avg': current_esm_avg,  # [新增]
                'ic50': ic50  # [新增] 添加 IC50 测量值
            })

        print(f"Found {len(valid_entries)} valid entries (Structure + ESM + ESM_AVG + IC50).")

        # sample 模式，每个 MHC 只保留固定数量
        if self.is_sample:
            sample_num_per_mhc = self.config.get('sample_num_per_mhc', 32)

            # 1. Grouping: 按 allele 分组
            entries_by_allele = defaultdict(list)
            for entry in valid_entries:
                entries_by_allele[entry['line']['allele']].append(entry)

            all_sampled_entries = []
            # 使用固定种子保证可复现性
            rng = np.random.RandomState(42)

            print(f"Sampling strategy: Target {sample_num_per_mhc} per allele.")

            for allele, group in entries_by_allele.items():
                n_samples = len(group)

                # 安全检查：防止空数据
                if n_samples == 0:
                    continue

                if n_samples < sample_num_per_mhc:
                    # Case 1: 欠采样 (样本数 < 目标数) -> 需要重复采样 (replace=True)
                    # print(f"  [Upsampling] Allele {allele}: {n_samples} -> {sample_num_per_mhc}")

                    # 关键修复：replace=True 允许重复被选中
                    chosen_idx = rng.choice(n_samples, size=sample_num_per_mhc, replace=True)

                    for idx in chosen_idx:
                        # [重要] 使用 deepcopy！
                        # 确保复制出来的样本在内存中是独立的对象。
                        # 防止后续 transform 或处理时，修改一个样本影响到所有重复的副本。
                        all_sampled_entries.append(copy.deepcopy(group[idx]))

                else:
                    # Case 2: 过采样 (样本数 >= 目标数) -> 随机抽取不重复 (replace=False)
                    chosen_idx = rng.choice(n_samples, size=sample_num_per_mhc, replace=False)
                    for idx in chosen_idx:
                        # 这里如果是只读，浅拷贝通常够用，但为了统一行为建议也用 deepcopy
                        # 如果为了性能，且确定后续只读，可以直接 append(group[idx])
                        all_sampled_entries.append(group[idx])
            valid_entries = all_sampled_entries
            print(f'Sampled {len(valid_entries)} entries for {len(entries_by_allele)} alleles.')

        # 建立一个列表来存储最终所有的 keys，方便 Dataset 类读取
        lmdb_keys = []

        with db.begin(write=True, buffers=True) as txn:
            # [修改 1] 使用 enumerate 获取循环索引 i，用于生成唯一 Key
            for i, entry in tqdm(enumerate(valid_entries), total=len(valid_entries), desc='Parsing structures'):
                line = entry['line']
                current_struct = entry['struct']
                current_esm = entry['esm']
                current_esm_avg = entry['esm_avg']
                current_ic50 = entry['ic50']

                # 原始 ID (用于数据内部记录，保持原样)
                entry_id = line['ID']
                try:
                    # 解析结构
                    pmhc_dict = parse_conf_list(current_struct, self.config, current_esm, current_esm_avg, current_ic50)
                    pmhc_dict = torchify_dict(pmhc_dict)
                    data = Drug3DData.from_drug3d_dicts(pmhc_dict)

                    # 数据内部保留原始 PDB ID，方便科研溯源
                    data.ID = entry_id
                    data.allele_name = line['allele']

                    # [修改 2] 构造 LMDB 的 Unique Key
                    # 格式: "OriginalID_Index" (例如: "3UTQ_0", "3UTQ_1", ...)
                    # 这样既保证了唯一性，又能看出来它是哪个 PDB 的副本
                    unique_key_str = f'{entry_id}_{i}'
                    unique_key_bytes = unique_key_str.encode('ascii')

                    txn.put(key=unique_key_bytes, value=pickle.dumps(data))

                    # 记录这个 key
                    lmdb_keys.append(unique_key_str)

                except Exception as e:
                    num_skipped += 1
                    # 建议打印 i 以便定位是第几个样本出问题
                    print(f'Skipping entry {i} (ID: {entry_id}): {e}')
                    continue

        db.close()
        print(f'Processed {len(valid_entries)} entries, Skipped {num_skipped}')

    def _precompute_id2idx(self):
        id2idx = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            id2idx[data.ID] = i
        torch.save(id2idx, self.id2idx_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.idx = idx
        if self.transform is not None:
            data = self.transform(data)
        return data


def modify_dataset(dataset):
    """
    Modify the dataset by removing peptide segments and adding new random noise and positions.
    """
    # Sample lengths based on peptide length distribution
    pep_len = sample_peptide_lengths(dataset)
    # 由于continuous，这里的噪声要改成从高斯分布采样，而不是randint
    # peptide内部edge也要初始加噪
    from collections import Counter
    allele_counter = Counter()
    for data in dataset:
        allele_counter[data.allele_name] += 1
        peptide_mask = (data.entity == 1)
        if peptide_mask.sum() > 0:
            # Remove peptide nodes
            non_peptide_mask = ~peptide_mask
            data.element = data.element[non_peptide_mask]
            data.pos_all_confs = data.pos_all_confs[:, non_peptide_mask, :]
            data.entity = data.entity[non_peptide_mask]
            data.num_atoms = non_peptide_mask.sum().item()

            # Sample a random length for the new peptide
            sampled_length = np.random.choice(pep_len)

            # Add new peptide nodes with Gaussian noise
            # types 由于还不是 one-hot，所以直接设为0，会后续在models/model/sample()中变成 one-hot 之后再加噪。
            random_types = torch.zeros(sampled_length)
            random_positions = torch.zeros((sampled_length, 3))
            random_positions.normal_()
            random_entities = torch.ones(sampled_length, dtype=torch.long)

            # Append new peptide nodes
            data.element = torch.cat([data.element, random_types], dim=0)
            data.pos_all_confs = torch.cat([data.pos_all_confs, random_positions.unsqueeze(0)], dim=1)
            data.entity = torch.cat([data.entity, random_entities], dim=0)
            data.num_atoms += sampled_length

            # 因为重新添加了peptide，所以 bond 和 halfedge 要重新生成
            entity_mask = data.entity == 1
            # build bond
            row, col = [], []
            bond_type_list = []

            peptide_indices = np.where((data.entity[:-1] == 1) & (data.entity[1:] == 1))[0]
            row = np.repeat(peptide_indices, 2) + np.tile([0, 1], len(peptide_indices))
            col = np.repeat(peptide_indices, 2) + np.tile([1, 0], len(peptide_indices))
            bond_type_list = np.ones(len(row), dtype=np.int64)

            bond_type = np.array(bond_type_list, dtype=np.int64)
            bond_index = np.array([row, col], dtype=np.int64)

            perm = (bond_index[0] * data.num_atoms + bond_index[1]).argsort()
            bond_index = bond_index[:, perm]
            bond_type = bond_type[perm]
            data.num_bonds = bond_index.shape[1] // 2  # 保留真实键数，而不是双向的
            data.bond_type = bond_type
            data.bond_index = bond_index

            # build halfedge (not full because perturb for edge_ij should be the same as edge_ji)
            edge_type_mat = torch.zeros([data.num_atoms, data.num_atoms], dtype=torch.long)
            for i in range(data.num_bonds * 2):  # multiplication by two is for symmetric of bond index
                edge_type_mat[data.bond_index[0, i], data.bond_index[1, i]] = data.bond_type[i]
            data.halfedge_index = torch.triu_indices(data.num_atoms, data.num_atoms, offset=1)
            data.halfedge_type = edge_type_mat[data.halfedge_index[0], data.halfedge_index[1]]
            assert len(data.halfedge_type) == len(data.halfedge_index[0])
            data.halfedge_entity = (
                (entity_mask[data.halfedge_index[0]] & entity_mask[data.halfedge_index[1]]).long() * 1) + (
                    (entity_mask[data.halfedge_index[0]] ^ entity_mask[data.halfedge_index[1]]).long() * 2)

            # halfedge_type 也没有加噪

    print("Allele 出现次数统计:")
    for name, count in allele_counter.most_common():  # most_common 会按次数从大到小排序
        print(f"{name}: {count}")

    # 如果只想看总共有多少种
    print(f"总共有 {len(allele_counter)} 种不同的 Allele。")
    return dataset


def sample_peptide_lengths(dataset):
    """
    Sample peptide lengths based on the distribution in the dataset.
    """
    peptide_lengths = []

    for data in dataset:
        peptide_mask = (data.entity == 1) & (data.element != 20)  # Exclude mask tokens
        if peptide_mask.sum() > 0:
            peptide_lengths.append(peptide_mask.sum().item())

    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    return peptide_lengths
