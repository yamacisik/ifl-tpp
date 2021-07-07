
import torch
from typing import Any, List, Optional, Tuple, Union
from  util import _size_repr,clamp_preserve_gradients,pad_sequence
import numpy as np


class DotDict:
    """Dictionary where elements can be accessed as dict.entry."""

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        info = [_size_repr(key, item) for key, item in self]
        return f"{self.__class__.__name__}({', '.join(info)})"

class Sequence(DotDict):
    """
    A sequence of events with corresponding marks.

    IMPORTANT: last entry of inter_times must correspond to the survival time
    until the end of the observed interval. Because of this len(inter_times) == len(marks) + 1.

    Args:
        inter_times: Inter-event times. Last entry corresponds to the survival time
            until the end of the observed interval, shape (seq_len,)
        marks: Mark corresponding to each event. Note that the length is 1 shorter than
            for inter_times, shape (seq_len - 1,)
    """
    def __init__(self, inter_times: torch.Tensor, marks: Optional[torch.Tensor] = None,device = 'cpu', **kwargs):
        if not isinstance(inter_times, torch.Tensor):
            inter_times = torch.tensor(inter_times,device = device)
        # The inter-event times should be at least 1e-10 to avoid numerical issues
        self.inter_times = inter_times.float().clamp(min=1e-10)

        if marks is not None:
            if not isinstance(marks, torch.Tensor):
                marks = torch.tensor(marks,device = device)
            self.marks = marks.long()
        else:
            self.marks = None

        for key, value in kwargs.items():
            self[key] = value

        self._validate_args()

    def __len__(self):
        return len(self.inter_times)

    def _validate_args(self):
        """Check if all tensors have correct shapes."""
        if self.inter_times.ndim != 1:
            raise ValueError(
                f"inter_times must be a 1-d tensor (got {self.inter_times.ndim}-d)"
            )
        if self.marks is not None:
            expected_marks_length = len(self.inter_times) - 1
            if self.marks.shape != (expected_marks_length,):
                raise ValueError(
                    f"marks must be of shape (seq_len - 1 = {expected_marks_length},)"
                    f"(got {self.marks.shape})"
                )

    def to(self, device: str):
        """Move the underlying data to the specified device."""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)


class Batch(DotDict):
    """
    A batch consisting of padded sequences.

    Usually constructed using the from_list method.

    Args:
        inter_times: Padded inter-event times, shape (batch_size, seq_len)
        mask: Mask indicating which inter_times correspond to observed events
            (and not to padding), shape (batch_size, seq_len)
        marks: Padded marks associated with each event, shape (batch_size, seq_len)
    """
    def __init__(self, inter_times: torch.Tensor, mask: torch.Tensor, marks: Optional[torch.Tensor] = None, **kwargs):
        self.inter_times = inter_times
        self.mask = mask
        self.marks = marks

        for key, value in kwargs.items():
            self[key] = value

        self._validate_args()

    @property
    def size(self):
        """Number of sequences in the batch."""
        return self.inter_times.shape[0]

    @property
    def max_seq_len(self):
        """Length of the padded sequences."""
        return self.inter_times.shape[1]

    def _validate_args(self):
        """Check if all tensors have correct shapes."""
        if self.inter_times.ndim != 2:
            raise ValueError(
                f"inter_times must be a 2-d tensor (got {self.inter_times.ndim}-d)"
            )
        if self.mask.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"mask must be of shape (batch_size={self.size}, "
                f" max_seq_len={self.max_seq_len}), got {self.mask.shape}"
            )
        if self.marks is not None and self.marks.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"marks must be of shape (batch_size={self.size},"
                f" max_seq_len={self.max_seq_len}), got {self.marks.shape}"
            )

    @staticmethod
    def from_list(sequences: List[Sequence]):
        batch_size = len(sequences)
        # Remember that len(seq) = len(seq.inter_times) = len(seq.marks) + 1
        # since seq.inter_times also includes the survival time until t_end
        max_seq_len = max(len(seq) for seq in sequences)
        inter_times = pad_sequence([seq.inter_times for seq in sequences], max_len=max_seq_len)

        dtype = sequences[0].inter_times.dtype
        device = sequences[0].inter_times.device
        mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)

        for i, seq in enumerate(sequences):
            mask[i, :len(seq) - 1] = 1

        if sequences[0].marks is not None:
            marks = pad_sequence([seq.marks for seq in sequences], max_len=max_seq_len)
        else:
            marks = None

        return Batch(inter_times, mask, marks)

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.mask[idx].sum(-1)) + 1
        inter_times = self.inter_times[idx, :length]
        if self.marks is not None:
            marks = self.marks[idx, :length - 1]
        else:
            marks = None
        # TODO: recover additional attributes (passed through kwargs) from the batch
        return Sequence(inter_times=inter_times, marks=marks)

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.size)]

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: List[Sequence], num_marks=1):
        self.sequences = sequences
        self.num_marks = num_marks

    def __getitem__(self, item):
        return self.sequences[item]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "SequenceDataset") -> "SequenceDataset":
        if not isinstance(other, SequenceDataset):
            raise ValueError(f"other must be a SequenceDataset (got {type(other)})")
        new_num_marks = max(self.num_marks, other.num_marks)
        new_sequences = self.sequences + other.sequences
        return SequenceDataset(new_sequences, num_marks=new_num_marks)

    def get_dataloader(
            self, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=Batch.from_list
        )

    def train_val_test_split(
            self, train_size=0.8, val_size=0.1, test_size=0.1, seed=None, shuffle=True,
    ) -> Tuple["SequenceDataset", "SequenceDataset", "SequenceDataset"]:
        """Split the sequences into train, validation and test subsets."""
        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("train_size, val_size and test_size must be >= 0.")
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size and test_size must add up to 1.")

        if seed is not None:
            np.random.seed(seed)

        all_idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(all_idx)

        train_end = int(train_size * len(self))  # idx of the last train sequence
        val_end = int((train_size + val_size) * len(self))  # idx of the last val seq

        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]

        train_sequences = [self.sequences[idx] for idx in train_idx]
        val_sequences = [self.sequences[idx] for idx in val_idx]
        test_sequences = [self.sequences[idx] for idx in test_idx]

        return (
            SequenceDataset(train_sequences, num_marks=self.num_marks),
            SequenceDataset(val_sequences, num_marks=self.num_marks),
            SequenceDataset(test_sequences, num_marks=self.num_marks),
        )

    def get_inter_time_statistics(self):
        """Get the mean and std of log(inter_time)."""
        all_inter_times = torch.cat([seq.inter_times[:-1] for seq in self.sequences])
        mean_log_inter_time = all_inter_times.log().mean()
        std_log_inter_time = all_inter_times.log().std()
        return mean_log_inter_time, std_log_inter_time

    @property
    def total_num_events(self):
        return sum(len(seq) - 1 for seq in self.sequences)


def get_inter_times(seq: dict):
    """Get inter-event times from a sequence."""
    return np.ediff1d(np.concatenate([[seq["t_start"]], seq["arrival_times"], [seq["t_end"]]]))


def create_seq_data_set(dataset, num_marks,device):
    sequences = [
        Sequence(
            inter_times=get_inter_times(seq),
            marks=seq.get("marks"),
            t_start=seq.get("t_start"),
            t_end=seq.get("t_end"),device = device
        )
        for seq in dataset["sequences"]
    ]
    dataset = SequenceDataset(sequences=sequences, num_marks=num_marks)

    return dataset