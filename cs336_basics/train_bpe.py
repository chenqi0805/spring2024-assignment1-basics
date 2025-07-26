from collections import Counter, defaultdict
import os
import regex as re
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer(object):
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def train(self, input_path: str | os.PathLike):
        vocabs = set([bytes([i]) for i in range(256)])
        vocabs.update([s.encode('utf-8') for s in self.special_tokens])
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        if self.special_tokens:
            special_pattern = f"(?:{'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_parts = [part for part in re.split(special_pattern, text) if part]
        else:
            text_parts = [text]
        
        pretokenized = self._pretokenize(text_parts)
        token_tuple_count = {tuple([char.encode('utf-8') for char in token]): count for token, count in pretokenized.items()}
        token_tuples = list(token_tuple_count.keys())

        inverted_indices = defaultdict(list)
        for i in range(len(token_tuples)):
            token_tuple = token_tuples[i]
            for j in range(len(token_tuple)-1):
                for k in range(j+1, len(token_tuple)):
                    inverted_indices[b''.join(token_tuple[j:k+1])].append(i)
        
        byte_pair_count = Counter()
        for token_tuple, count in token_tuple_count.items():
            for i in range(len(token_tuple)-1):
                byte_pair_count[token_tuple[i:i+2]] += count

        byte_pair_count_max_heap = [(-count, pair) for pair, count in byte_pair_count.items()]

        remaining = self.vocab_size - len(vocabs)
        merges = []
        while remaining > 0:
            print(f"remaining {remaining}")
            negCount, tuple_to_merge = heapq.heappop(byte_pair_count_max_heap)
            mergedBytes = b''.join(tuple_to_merge)
            while mergedBytes in vocabs or -negCount != byte_pair_count[tuple_to_merge]:
                negCount, tuple_to_merge = heapq.heappop(byte_pair_count_max_heap)
                mergedBytes = b''.join(tuple_to_merge)
            merges.append(tuple_to_merge)
            vocabs.add(mergedBytes)
            remaining -= 1
            for idx in inverted_indices[mergedBytes]:
                old_token_tuple = token_tuples[idx]
                word = b''.join(old_token_tuple).decode('utf-8')
                word_count = pretokenized[word]
                new_token_tuple = self.merge_token_tuple(old_token_tuple, tuple_to_merge)
                token_tuples[idx] = new_token_tuple
                for i in range(len(old_token_tuple)-1):
                    byte_pair_count[old_token_tuple[i:i+2]] -= word_count
                for i in range(len(new_token_tuple)-1):
                    byte_pair_count[new_token_tuple[i:i+2]] += word_count
                byte_pairs_to_push = set()
                for i in range(len(new_token_tuple)-1):
                    if new_token_tuple[i] == mergedBytes or new_token_tuple[i+1] == mergedBytes:
                        byte_pairs_to_push.add(new_token_tuple[i:i+2])
                for byte_pair in byte_pairs_to_push:
                    heapq.heappush(byte_pair_count_max_heap, (-byte_pair_count[byte_pair], byte_pair))
        vocab_dict = {i: v for i, v in enumerate(vocabs)}
        return vocab_dict, merges

    def merge_token_tuple(self, token_tuple, tuple_to_merge):
        new_token_tuple = []
        idx = 0
        while idx < len(token_tuple):
            if idx < len(token_tuple) - 1 and token_tuple[idx:idx+2] == tuple_to_merge:
                new_token_tuple.append(b''.join(tuple_to_merge))
                idx += 2
            else:
                new_token_tuple.append(token_tuple[idx])
                idx += 1
        return tuple(new_token_tuple)
        
    def _pretokenize(self, text_parts: list[str]):
        pretokenized = Counter()
        for text_part in text_parts:
            pretokenized.update(re.findall(PAT, text_part))
        return pretokenized
    
obj = BPETokenizer(10000, ["<|endoftext|>"])
vocab_dict, merges = obj.train("data/TinyStoriesV2-GPT4-train.txt")
print(merges)
# print((1, 2, 3)[1:3])
# print(tuple([s.encode('utf-8') for s in 'ac']))