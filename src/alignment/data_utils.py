import os
import sys
import logging
from typing import List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def check_collator_and_pad_token(tokenizer: PreTrainedTokenizer, 
                                 packing: bool,
                                 force_pad_token: bool = False) -> None:
    """
    Verifies that EOS tokens will not be removed from the labels by `DataCollatorForLanguageModeling`.
    If True, will raise an exception, unless force_pad_token=True in the config
    """
    condition_pad_token = True if tokenizer.pad_token_id == tokenizer.eos_token_id else False
    condition_packing = True if packing else False
    
    if (condition_pad_token and not condition_packing) and not force_pad_token:
        raise ValueError('SFT trainer will set `DataCollatorForLanguageModeling` as collator. Since pad_token_id == eos_token_id the collator will remove EOS tokens from the data labels, and this will produce a model that generates indefinitely. You should set a collator or opt to packing. If you would like to force this behaviour set `force_pad_token=True` in the config.')
    

def list_depth(lst: list) -> int:
    """
    Checks the depth of a list
    """
    if not (isinstance(lst, list) or isinstance(lst, tuple)):
        # If the element is not a list, it's at depth 0
        return 0
    else:
        # If the element is a list, find the maximum depth among its elements and add 1
        return 1 + max(list_depth(item) for item in lst)


def concatenate_lists(input_list: list, max_item_length: int, sep: str = '') -> list:
    """
    Concatenate lists within the input list, following a two-level hierarchy (master and secondary).
    The concatenation is performed in a greedy manner, considering the maximum item length for 
    master concatenation and an optional separator for secondary concatenation.

    Args:
    - input_list (list): A list of nested lists, each containing a master list and 
      secondary items to concatenate.
    - max_item_length (int): The maximum allowed length for concatenated master lists.
    - sep (str, optional): Separator to use when concatenating secondary items.
      Defaults to an empty string.

    Returns:
    - list: A new list containing concatenated nested lists, maintaining the two-level hierarchy.
    """
    # Base case: if there's only one list or no lists left, return the list as is
    if len(input_list) <= 1:
        return input_list
    
    # Check if input_list also has "metadata", so that we know the required level of execution
    depth = list_depth(input_list)
    # If depth == 2, synthetically add another level so the function can run. This will be "flattened" eventually
    if depth == 2:
        input_list = [[item] for item in input_list]
    
    # Input list must be sorted by the length of L1 items
    input_list = sorted(input_list, key=lambda x: len(x[0]), reverse=True)

    new_list = []
    used_indices = set()
    last_idx_checked = -1

    # Iterate over the list of L1 items - starting from the bottom
    for j in range(len(input_list) - 1, -1, -1):
        for i in range(last_idx_checked + 1, j):
            # Check if the current pair of L2 lists can be concatenated
            if len(input_list[i][0]) + len(input_list[j][0]) <= max_item_length:
                # Concatenate "master" L2 lists in the current L1 item
                new_l1_item = [input_list[i][0] + input_list[j][0]]
                # Concatenate the L2 lists according to L1 concatenation <-> iff depth == 3
                if depth == 3:
                    other_l1_concat = [l2_i + sep + l2_j for l2_i, l2_j in zip(input_list[i][1], input_list[j][1])]
                    new_l1_item.append(other_l1_concat)
                
                new_list.append(new_l1_item)
                used_indices.add(i)
                used_indices.add(j)
                last_idx_checked = i
                break
        
        # Check if we iterated over all the list. If so, there are no more possible matches
        if j == i + 1:
            break
    
    # Add any L1 items that weren't used
    indices_not_used = set(range(len(input_list))) - used_indices
    [new_list.append(input_list[k]) for k in indices_not_used]
    
    # If original depth == 2, we transform the result back to this depth
    if depth == 2:
        new_list = [[item for sublist in sublist_lst for item in sublist] for sublist_lst in new_list]

    return new_list


def tokenize_to_list(text_data: List[str],
                    tokenizer: PreTrainedTokenizer,
                    metadata: Optional[List[list]] = None,
                    **tokenizer_kwargs) -> list:
    
    tokenized_data = tokenizer.batch_encode_plus(text_data, return_attention_mask=False, **tokenizer_kwargs)['input_ids']
    if metadata:
        metadata = list(zip(*metadata))
        res = list(zip(tokenized_data, metadata))
        return res
    else:
        return tokenized_data
    
    
def decode_list_to_dataset(input_data: list, tokenizer: PreTrainedTokenizer, metadata_column_names: Optional[List[str]] = None) -> Dataset:
    depth = list_depth(input_data)
    if depth == 3:
        input_data, metadata = zip(*input_data)
    else:
        metadata = None
    
    decode_data = tokenizer.batch_decode(input_data)
    # For some reason decoding adds a whitespace between BOS token (usually <s>) and the next token, so we remove it
    # Also some chat templates (Zephyr) add " \n" in the last message after </s> which is redundant
    decode_data = [txt.replace(tokenizer.bos_token + ' ', tokenizer.bos_token).rstrip() for txt in decode_data]
    res = {'text': decode_data}
    if metadata:
        metadata_dict = {k: v for k, v in zip(metadata_column_names, zip(*metadata))}
        res = {**res, **metadata_dict}
    dataset = Dataset.from_dict(res)
    return dataset
    
    
def dataset_bin_packing(dataset: Union[Dataset, DatasetDict], 
                        tokenizer: PreTrainedTokenizer, 
                        text_column: str, 
                        max_seq_length: int, 
                        metadata_columns: Optional[List[str]] = None, 
                        sep: Optional[str] = ' | ', 
                        **tokenizer_kwargs) -> Union[Dataset, DatasetDict]:
    """
    This function takes in a Dataset or DatasetDict and concats rows on a greedy basis
    It returns Dataset / DatasetDict with concatenated rows for both the text column and any user-defined metadata columns
    """
    dataset_flag = False
    if not (isinstance(dataset, Dataset) or isinstance(dataset, DatasetDict)):
        raise ValueError('input dataset must be a Dataset or DatasetDict object')
    elif isinstance(dataset, Dataset):
        # If input is a Dataset, convert it to DatasetDict in order to run the function
        dataset = DatasetDict({'ALL': dataset})
        dataset_flag = True  # For reversing to Dataset later
    
    dataset_dict = {}
    for name, split in dataset.items():
        text_data = split[text_column]
        if metadata_columns:
            metadata = [split[col] for col in metadata_columns]
        else:
            metadata = None
        tokenized_data = tokenize_to_list(text_data=text_data, tokenizer=tokenizer, metadata=metadata, **tokenizer_kwargs)
        idx = 0
        original_len = len(tokenized_data)
        data_len = len(tokenized_data)
        while data_len < len(tokenized_data) or idx == 0:
            tokenized_data = concatenate_lists(input_list=tokenized_data, max_item_length=max_seq_length, sep=sep)
            data_len = len(tokenized_data)
            idx += 1
        logger.info(f"Dataset packing algorithm for {name.upper()} split ran {idx} times. Row count was reduced from {original_len} to {data_len}")
        new_dataset = decode_list_to_dataset(input_data=tokenized_data, tokenizer=tokenizer, metadata_column_names=metadata_columns)
        
        dataset_dict[name] = new_dataset
        
    if dataset_flag:
        return dataset_dict['ALL']
    else:
        return DatasetDict(dataset_dict)
