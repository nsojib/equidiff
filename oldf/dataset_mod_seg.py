"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

"""
modified of /home/ns1254/robomimic/robomimic/utils/dataset.py

Changes:
    - remove filter_by_attribute
    - add segment_map
"""

class MultiSegmentSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        segments_map,       #NEW: demo_no: [ [start,end,gb],...] list of segments for each demo_no
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        # filter_by_attribute=None,
        load_next_obs=True,
    ):
        """ 
        toadd
        """
        super(MultiSegmentSequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        self.last_asked_index=-1

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        # self.filter_by_attribute = filter_by_attribute

        self.segments_map = segments_map

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(segments_map=self.segments_map)

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("MultiSegmentSequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def load_demo_info(self, segments_map):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        self.demos=list(segments_map.keys()) 
        # self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._index_to_index_in_demo=dict()  # maps every index to the actual index in demo 
        self._index_to_start_segment=dict()  # maps every index to the start of the segment
        self._index_to_end_segment=dict()    # maps every index to the end of the segment

        #TODO: remove the below
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()



        # # determine index mapping
        # self.total_num_sequences = 0
        # for ep in self.demos:
        #     demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        #     self._demo_id_to_start_indices[ep] = self.total_num_sequences
        #     self._demo_id_to_demo_length[ep] = demo_length

        #     num_sequences = demo_length
        #     # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
        #     if not self.pad_frame_stack:
        #         num_sequences -= (self.n_frame_stack - 1)
        #     if not self.pad_seq_length:
        #         num_sequences -= (self.seq_length - 1)

        #     if self.pad_seq_length:
        #         assert demo_length >= 1  # sequence needs to have at least one sample
        #         num_sequences = max(num_sequences, 1)
        #     else:
        #         assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

        #     for _ in range(num_sequences):
        #         self._index_to_demo_id[self.total_num_sequences] = ep
        #         self.total_num_sequences += 1

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:  
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            demo_length=0  
            for seg in self.segments_map[ep]: 
                demo_length += seg[1]-seg[0]       #sum of all segments length

                for i in range(seg[0], seg[1]+1):  #valid indeices from seg[0] to seg[1] inclusive
                    self._index_to_start_segment[self.total_num_sequences] = seg[0]
                    self._index_to_end_segment[self.total_num_sequences] = seg[1] 
                    
                    self._index_to_index_in_demo[self.total_num_sequences] = i 
                    self._index_to_demo_id[self.total_num_sequences] = ep
                    self.total_num_sequences += 1

            self._demo_id_to_demo_length[ep] = demo_length 
            # print(f"ep={ep} demo_length={demo_length} segments={self.segments_map[ep]} org_demo_length={demo_length1}")
 

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str =  "none"
        goal_mode_str = "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("MultiSegmentSequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            # get obs
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()] for k in obs_keys}
            if load_next_obs:
                all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()] for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations 
        (per dimension and per obs key) and returns it.
        """
        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("MultiSegmentSequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
            obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        # print('__getitem__ called with index=',index)

        self.last_asked_index=index 
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    # def get_item(self, index):
    #     """
    #     Main implementation of getitem when not using cache.
    #     """

        # demo_id = self._index_to_demo_id[index]
        # demo_start_index = self._demo_id_to_start_indices[demo_id]
        # demo_length = self._demo_id_to_demo_length[demo_id]

        # # start at offset index if not padding for frame stacking
        # demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        # index_in_demo = index - demo_start_index + demo_index_offset

        # # end at offset index if not padding for seq length
        # demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        # end_index_in_demo = demo_length - demo_length_offset

        # meta = self.get_dataset_sequence_from_demo(
        #     demo_id,
        #     index_in_demo=index_in_demo,
        #     keys=self.dataset_keys,
        #     num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
        #     seq_length=self.seq_length
        # )

        # # determine goal index
        # goal_index = None
        # if self.goal_mode == "last":
        #     goal_index = end_index_in_demo - 1

        # meta["obs"] = self.get_obs_sequence_from_demo(
        #     demo_id,
        #     index_in_demo=index_in_demo,
        #     keys=self.obs_keys,
        #     num_frames_to_stack=self.n_frame_stack - 1,
        #     seq_length=self.seq_length,
        #     prefix="obs"
        # )
        # if self.hdf5_normalize_obs:
        #     meta["obs"] = ObsUtils.normalize_obs(meta["obs"], obs_normalization_stats=self.obs_normalization_stats)


    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """ 

        # print('get_item called with index=',index)

        demo_id=self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        index_in_demo = self._index_to_index_in_demo[index] 
        
        start_segment=self._index_to_start_segment[index]
        end_segment=self._index_to_end_segment[index]

        meta=self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys= self.dataset_keys,
            num_frames_to_stack= self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length= self.seq_length,
            start_segment=start_segment,
            end_segment=end_segment
        )

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys= self.obs_keys,
            num_frames_to_stack= self.n_frame_stack - 1,
            seq_length= self.seq_length,
            prefix="obs",
            start_segment=start_segment,
            end_segment=end_segment
        )
        # meta["ep"] = demo_id

        # print(f"index={index} demo_id={demo_id} index_in_demo={index_in_demo} start_segment={start_segment} end_segment={end_segment} meta={meta['actions'].shape}")
        return meta  

    # def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
    #     """
    #     Extract a (sub)sequence of data items from a demo given the @keys of the items.

    #     Args:
    #         demo_id (str): id of the demo, e.g., demo_0
    #         index_in_demo (int): beginning index of the sequence wrt the demo
    #         keys (tuple): list of keys to extract
    #         num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
    #         seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

    #     Returns:
    #         a dictionary of extracted items.
    #     """
    #     assert num_frames_to_stack >= 0
    #     assert seq_length >= 1

    #     demo_length = self._demo_id_to_demo_length[demo_id]
    #     assert index_in_demo < demo_length

    #     # determine begin and end of sequence
    #     seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
    #     seq_end_index = min(demo_length, index_in_demo + seq_length)

    #     # determine sequence padding
    #     seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
    #     seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

    #     # make sure we are not padding if specified.
    #     if not self.pad_frame_stack:
    #         assert seq_begin_pad == 0
    #     if not self.pad_seq_length:
    #         assert seq_end_pad == 0

    #     # fetch observation from the dataset file
    #     seq = dict()
    #     for k in keys:
    #         data = self.get_dataset_for_ep(demo_id, k)
    #         seq[k] = data[seq_begin_index: seq_end_index]

    #     seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
    #     pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
    #     pad_mask = pad_mask[:, None].astype(bool)

    #     return seq, pad_mask

    # def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
    #     """
    #     Extract a (sub)sequence of observation items from a demo given the @keys of the items.

    #     Args:
    #         demo_id (str): id of the demo, e.g., demo_0
    #         index_in_demo (int): beginning index of the sequence wrt the demo
    #         keys (tuple): list of keys to extract
    #         num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
    #         seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
    #         prefix (str): one of "obs", "next_obs"

    #     Returns:
    #         a dictionary of extracted items.
    #     """
    #     obs, pad_mask = self.get_sequence_from_demo(
    #         demo_id,
    #         index_in_demo=index_in_demo,
    #         keys=tuple('{}/{}'.format(prefix, k) for k in keys),
    #         num_frames_to_stack=num_frames_to_stack,
    #         seq_length=seq_length,
    #     )
    #     obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
    #     if self.get_pad_mask:
    #         obs["pad_mask"] = pad_mask

    #     return obs

    # def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
    #     """
    #     Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
    #     Args:
    #         demo_id (str): id of the demo, e.g., demo_0
    #         index_in_demo (int): beginning index of the sequence wrt the demo
    #         keys (tuple): list of keys to extract
    #         num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
    #         seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

    #     Returns:
    #         a dictionary of extracted items.
    #     """
    #     data, pad_mask = self.get_sequence_from_demo(
    #         demo_id,
    #         index_in_demo=index_in_demo,
    #         keys=keys,
    #         num_frames_to_stack=num_frames_to_stack,
    #         seq_length=seq_length,
    #     )
    #     if self.get_pad_mask:
    #         data["pad_mask"] = pad_mask
    #     return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        return None

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, start_segment, end_segment, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        # demo_length = self._demo_id_to_demo_length[demo_id]
        # demo_length = end_segment-start_segment
        # assert index_in_demo < demo_length

        end_segment=end_segment+1 #TODO: double check, why +1 is needed

        # determine begin and end of sequence  
        seq_begin_index = max(start_segment, index_in_demo - num_frames_to_stack)
        seq_end_index = min(end_segment, index_in_demo + seq_length)  

        # determine sequence padding
        seq_begin_pad = max(0, start_segment+num_frames_to_stack - index_in_demo)  # pad for frame stacking
        # seq_end_pad = max(0, index_in_demo + seq_length - demo_length-1)  # pad for sequence length  #TODO double check -1
        seq_end_pad = max(0, index_in_demo + seq_length - end_segment)

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict() 
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)  
            seq[k] = data[seq_begin_index: seq_end_index]
 

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, start_segment, end_segment, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """ 
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
            start_segment=start_segment,
            end_segment=end_segment
        ) 


        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs


    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, start_segment, end_segment, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
            start_segment=start_segment,
            end_segment=end_segment
        ) 
        
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data
    

