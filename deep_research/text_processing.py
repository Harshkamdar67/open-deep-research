import tiktoken
from typing import List

class TextSplitter:
    """
    A base class for splitting text into smaller chunks for processing.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextSplitter with chunking parameters.
        
        Parameters:
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.
        
        Parameters:
            text (str): The input text to be split.
            
        Returns:
            List[str]: A list of text chunks.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

class RecursiveCharacterTextSplitter(TextSplitter):
    """
    A recursive text splitter that intelligently splits text based on different separators.
    """

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None
    ):
        """
        Initialize the RecursiveCharacterTextSplitter.
        
        Parameters:
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (List[str]): List of separators to use for splitting.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ".", ",", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using the best separator.
        
        Parameters:
            text (str): The input text.
            
        Returns:
            List[str]: A list of optimally split text chunks.
        """
        final_chunks = []
        separator = next((s for s in self.separators if s in text), self.separators[-1])
        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self.merge_splits(good_splits, separator))
                    good_splits = []
                final_chunks.extend(self.split_text(split))
        
        if good_splits:
            final_chunks.extend(self.merge_splits(good_splits, separator))
        
        return final_chunks

    def merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge split chunks while ensuring they fit within the chunk size.
        
        Parameters:
            splits (List[str]): List of split text pieces.
            separator (str): The separator used for splitting.
            
        Returns:
            List[str]: A list of merged chunks.
        """
        docs, current_chunk = [], []
        total_length = 0

        for part in splits:
            part_length = len(part)
            if total_length + part_length >= self.chunk_size:
                if total_length > self.chunk_size:
                    print(f"Warning: Chunk exceeds max size ({self.chunk_size})")

                if current_chunk:
                    docs.append(separator.join(current_chunk).strip())
                    while total_length > self.chunk_overlap and current_chunk:
                        total_length -= len(current_chunk.pop(0))

            current_chunk.append(part)
            total_length += part_length
        
        if current_chunk:
            docs.append(separator.join(current_chunk).strip())

        return docs

def trim_prompt(prompt: str, context_size: int = 120000, min_chunk_size: int = 140) -> str:
    """
    Trim a prompt to fit within the maximum context size.
    
    Parameters:
        prompt (str): The input prompt.
        context_size (int): The maximum allowed token length.
        min_chunk_size (int): The minimum allowable chunk size.
        
    Returns:
        str: A trimmed prompt that fits within the allowed size.
    """
    if not prompt:
        return ""

    encoder = tiktoken.get_encoding("o200k_base")
    length = len(encoder.encode(prompt))

    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    estimated_chars = overflow_tokens * 3
    chunk_size = max(len(prompt) - estimated_chars, min_chunk_size)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    trimmed_prompt = splitter.split_text(prompt)[0] if splitter.split_text(prompt) else ""

    return trim_prompt(trimmed_prompt, context_size) if len(trimmed_prompt) == len(prompt) else trimmed_prompt
