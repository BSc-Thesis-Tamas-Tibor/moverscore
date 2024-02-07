"""
Text Embedding Module with Transformer Models

Key Features:
- Easy-to-use interface for generating embeddings from text data.
- Supports batch processing of texts for efficient computation.
- Utilizes pre-trained transformer models such as BERT, DistilBERT, and others, 
making it versatile for different NLP tasks.
- Handles tokenization, padding, and application of Inverse Document Frequency (IDF) 
weights internally.
- Optimized to run on GPU devices for accelerated computation, 
with automatic fallback to CPU if GPU is not available.

Example Usage:s
    from text_embedder import TextEmbedder
    
    # Initialize the embedder with a specific transformer model
    embedder = TextEmbedder(model_name='bert-base-uncased')
    
    # Prepare a list of sentences for embedding
    sentences = [["The big blue whale in the ocean, The ocean blue car passed the corner'"]
    
    # Generate embeddings
    embeddings, seq_lengths, attention_mask, padded_idf, tokens = embedder(sentences)

This module requires the transformers library and PyTorch, along with their dependencies,
to be installed in your environment.
"""


from collections import defaultdict, Counter
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModel
from torch.cuda import is_available



class TextEmbedder:
    """
    A comprehensive class for generating text embeddings using pre-trained transformer models.

    This class abstracts the complexity involved in text preprocessing, tokenization, embedding generation,
    and post-processing for NLP tasks. It leverages Hugging Face's transformers library to utilize pre-trained
    models such as BERT, DistilBERT, etc., for embedding generation. The class is designed to be flexible and 
    efficient, allowing for batch processing of text data and utilization of GPU resources if available.

    Attributes:
        model_name (str | None): Name of the pre-trained transformer model. Defaults to 'distilbert-base-uncased'.
        device (str): Computation device ('cuda:0' for GPU or 'cpu'). Automatically selected based on availability.
        tokenizer: Instance of the tokenizer corresponding to the specified model for text tokenization.
        model: Instance of the transformer model for generating embeddings.

    Args:
        model_name (str | None, optional): Specifies the transformer model to be used for embedding generation.
                                            If not provided, 'distilbert-base-uncased' is used by default.

    Methods:
        __call__(self, sentences: list, batch_size: int = -1) -> tuple:
            Acts as a callable for the class to generate embeddings for a list of sentences.
            Supports dynamic batch processing and automatically handles padding and attention mask creation.

        truncate(self, tokens: list) -> list:
            Truncates a list of tokens to fit within the model's maximum input size.

        tokenize_text(self, input_text: str) -> list:
            Tokenizes input text, adding special tokens ([CLS] and [SEP]).

        process_text(self, input_text: str) -> set:
            Processes input text into a set of unique token IDs, ready for embedding generation.

        create_idf_dict(self, input_texts: list, n_threads: int = 4) -> defaultdict:
            Creates an IDF dictionary for a given list of input texts.

        padding(self, sequences: list, pad_token: int, dtype: torch.dtype = torch.long) -> tuple:
            Pads sequences to uniform length, creating necessary tensors for model input and attention handling.

        encode(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor, last_hidden_state: int = 0) -> tuple:
            Encodes input tensors to generate embeddings.

        collate_idf(self, sequences: list, pad: str = "[PAD]") -> tuple:
            Prepares input sequences for embedding generation, applying tokenization, padding, and IDF weighting.

    Usage:
        >>> embedder = TextEmbedder(model_name='bert-base-uncased')
        >>> sentences = ["The big blue whale in the ocean, The ocean blue car passed the corner'"]
        >>> embeddings, seq_lengths, attention_mask, padded_idf, tokens = embedder(sentences)
    """


    def __init__(self, model_name: Union[str, None] = None) -> None:
        # Set the model name to the provided value or to a default if not provided
        self.model_name = model_name if model_name else 'distilbert-base-uncased'

        # Determine the computing device based on the availability of CUDA (GPU support)
        self.device = 'cuda' if is_available() else 'cpu'

        # Load the tokenizer corresponding to the specified (or default) model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the pre-trained transformer model from the Hugging Face's transformers library
        self.model = AutoModel.from_pretrained(self.model_name)

        # Set the model to evaluation mode.
        self.model.eval()

        # Move the model to the appropriate computing device (GPU or CPU)
        self.model.to(self.device)

    def __call__(self, sentences: list, batch_size: int = -1) -> tuple:
        """
        Generates embeddings for a list of sentences, processing them in batches.

        This method acts as a callable interface for the class, allowing the user to 
        directly generate embeddings for a list of input sentences. It preprocesses the 
        sentences by tokenizing, applying IDF weights, padding to uniform length, and 
        then computes the embeddings using the BERT model or a similar transformer model. 

        Args:
            sentences (list): A list of sentences to be encoded into embeddings.
            batch_size (int, optional): The number of sentences to process in each batch.
                                        If set to -1, all sentences are processed in a single batch.
                                        Defaults to -1.

        Returns:
            tuple: Contains the following elements:
                - total_embedding (torch.Tensor): The concatenated embeddings for all input sentences.
                - seq_lengths (torch.Tensor): The lengths of each sentence before padding.
                - attention_mask (torch.Tensor): A binary mask indicating the presence of tokens versus padding.
                - padded_idf (torch.Tensor): The IDF weights for each token in the padded sentences.
                - tokens (list of list): Tokenized representation of the input sentences including special tokens.
        """

        # Prepare input sentences for embedding generation
        padded_ids, padded_idf, seq_lengths, attention_mask, tokens = self.collate_idf(
            sentences)

        # Set batch size to total number of sentences if batch_size is -1, otherwise use specified batch_size
        batch_size = len(sentences) if batch_size == -1 else batch_size

        # Initialize list to store batch embeddings
        embeddings = []

        # Generate embeddings without computing gradients for efficiency
        with torch.no_grad():

            # Iterate through sentences in batches
            for i in range(0, len(sentences), batch_size):

                # Encode the current batch of sentences to get embeddings
                batch_embedding = self.encode(padded_ids[i:i+batch_size],
                                              attention_mask=attention_mask[i:i+batch_size])

                # Stack the embeddings from the current batch
                # batch_embedding = torch.stack(batch_embedding)

                # Append the batch embeddings to the list
                embeddings.append(batch_embedding)

                # Free up memory
                del batch_embedding

        # Concatenate embeddings from all batches into a single tensor
        total_embedding = torch.cat(embeddings, dim=-3)

        # Return the final embeddings and additional useful information
        return total_embedding, seq_lengths, attention_mask, padded_idf, tokens

    def collate_idf(self,
                    sequences: list,
                    pad: str = "[PAD]"):
        """
        Prepares input sequences for model processing, including padding and IDF weighting.

        This method takes a list of text sequences, tokenizes them, converts the tokens to
        numerical IDs, applies Inverse Document Frequency (IDF) weighting, and then pads the
        sequences to ensure they are of uniform length.

        Args:
            sequences (list): A list of raw text sequences to be processed.
            pad (str, optional): The token used for padding shorter sequences to match the
                                longest sequence in the batch. Defaults to "[PAD]".

        Returns:
            tuple: A tuple containing the following elements:
                - padded_ids (torch.Tensor): A tensor containing the padded numerical IDs of the input sequences.
                - padded_idf (torch.Tensor): A tensor of the same shape as `padded_ids`, 
                containing IDF weights for each token.
                - seq_lengths (torch.Tensor): A tensor containing the original lengths of the sequences before padding.
                - attention_mask (torch.Tensor): A binary tensor indicating which elements are tokens and padding.
                - tokens (list of list): A nested list of tokenized texts, including special tokens.

        """

        # Create the IDF dictionary for the inpust sequences of texts
        idf_dict = self.create_idf_dict(sequences)

        # Tokenize each sequence
        tokens = [self.tokenize_text(seq) for seq in sequences]

        # Convert each text sequence into a set of unique token IDs
        id_sequences = [self.process_text(text) for text in sequences]

        # Apply IDF weights to each token ID in the sequences
        idf_weights = [[idf_dict[i] for i in seq] for seq in id_sequences]

        # Convert the padding token to its numerical ID
        pad_token = self.tokenizer.convert_tokens_to_ids([pad])[0]

        # Pad the sequences of token IDs to uniform length and generate attention masks
        padded_ids, seq_lengths, attention_mask = self.padding(
            id_sequences, pad_token)

        # Similarly, pad the sequences of IDF weights to match the padded token IDs
        padded_idf, _, _ = self.padding(
            idf_weights, pad_token, dtype=torch.float)

        # Return the prepared data structures for model processing
        return padded_ids, padded_idf, seq_lengths, attention_mask, tokens

    def encode(self,
               input_tensor: torch.Tensor,
               attention_mask: torch.Tensor,
               last_hidden_state: int = 0) -> tuple:
        """
        Generates embeddings from the model for the given input tensor and attention mask.

        This method sets the model to evaluation mode and passes the input tensor along
        with an attention mask through the model to obtain embeddings. 

        Args:
            input_tensor (torch.Tensor): The input tensor to the model.
            attention_mask (torch.Tensor): A binary tensor of the same shape as `input_tensor`
                                        indicating which elements are padding and should not
                                        be attended to by the model.
            last_hidden_state (int, optional): The index of the desired output in the model's
                                            return tuple. Defaults to 1.

        Returns:
            tuple: The embeddings from the model, typically represented by the last hidden states.
        """

        # Set the model to evaluation mode.
        self.model.eval()

        # Disable gradient calculation for efficiency and to reduce memory usage during inference
        with torch.no_grad():
            # Forward pass through the model with the input tensor and attention mask
            result = self.model(input_tensor, attention_mask=attention_mask)

        return result[last_hidden_state]

    def truncate(self, tokens: list) -> list:
        """
        Truncates a list of tokens to a maximum length allowed by the tokenizer.

        This method ensures that the token list does not exceed the maximum sequence
        length that the tokenizer can handle. It accounts for the addition of special
        tokens ([CLS] and [SEP]) which are added in a seperate method.

        Args:
            tokens (list): A list of tokens representing a tokenized text.

        Returns:
            list: A truncated list of tokens if the original list exceeded the
                maximum length; otherwise, the original list of tokens.
        """

        # Check if the length of the token list exceeds the maximum model input length minus 2
        if len(tokens) > self.tokenizer.model_max_length - 2:
            # Truncate the token list to fit within the maximum input length,
            # leaving space for the [CLS] and [SEP] tokens
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]

        return tokens

    def tokenize_text(self, input_text: str) -> list:
        """
        Tokenizes the input text, adding special tokens at the beginning and end.

        This method tokenizes the given text, prepends the '[CLS]' token at the beginning, 
        appends the '[SEP]' token at the end,
        and ensures the tokenized text does not exceed the maximum sequence length allowed
        by the model, truncating it if necessary.

        Args:
            input_text (str): The text to tokenize.

        Returns:
            list: A list of tokens representing the input text, including '[CLS]' at the
                beginning, '[SEP]' at the end.
        """

        # Tokenize the input text
        tokens = ["[CLS]"] + \
            self.truncate(self.tokenizer.tokenize(input_text)) + ["[SEP]"]

        return tokens

    def process_text(self, input_text: str) -> set:
        """
        Processes the input text to produce a set of unique token IDs.

        This method first tokenizes the input text, including the addition of special
        tokens ('[CLS]' at the start and '[SEP]' at the end) and truncation to the model's
        maximum sequence length. 

        Args:
            input_text (str): The text to process.

        Returns:
            set: A set of unique token IDs representing the processed input text.
        """

        # Tokenize the input text, including special tokens
        tokens = self.tokenize_text(input_text)

        # Convert tokens to their corresponding IDs for model processing
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Return a set of token IDs
        return set(ids)

    def create_idf_dict(self, input_texts: list, n_threads: int = 4) -> defaultdict:
        """
        Creates an Inverse Document Frequency (IDF) dictionary for a given list of input texts.

        This method processes the input texts to calculate the frequency of each unique token
        across all documents. It then computes the IDF for each token, which measures how common
        or rare a token is across the given documents.

        Args:
            input_texts (list): A list of strings where each string is a document from which to calculate IDFs.
            n_threads (int, optional): The number of threads to use for parallel processing of texts. Defaults to 4.

        Returns:
            defaultdict: A dictionary where keys are token indices (or IDs) and values 
                        are their corresponding IDF scores.Tokens not seen in the input 
                        texts are assigned a default IDF score based on the total number of documents.

        """

        # Initialize a counter to keep track of token frequencies across documents
        idf_counter = Counter()

        # Calculate the total number of documents
        num_docs = len(input_texts)

        # Create a partial function for processing text
        process_text_partial = partial(self.process_text)

        # Use multiprocessing Pool to parallelize the processing of texts
        with Pool(n_threads) as pool:

            # Update the counter with the frequency of tokens from all documents
            # chain.from_iterable is used to flatten the list of lists of tokens into a single list
            idf_counter.update(chain.from_iterable(
                pool.map(process_text_partial, input_texts)))

        # Initialize the IDF dictionary with a default value for unseen tokens
        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))

        # Update the IDF dictionary with computed IDF values for each token
        # IDF is calculated using the formula: log((num_docs + 1) / (token_frequency + 1))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1))
                        for (idx, c) in idf_counter.items()})

        return idf_dict

    @staticmethod
    def padding(sequences: list, pad_token: int, dtype: torch.dtype = torch.long) -> tuple:
        """
        Pads sequences to the same length and creates a mask to identify padded elements.

        This function takes a list of sequences (each sequence is a list of integers),
        pads them to the length of the longest sequence using a specified padding token,
        and creates a mask to distinguish real sequence elements from padding.

        Args:
            sequences (list): A list of sequences, where each sequence is a list of integers.
            pad_token (int): An integer representing the token used for padding shorter sequences.
            dtype (torch.dtype, optional): The desired data type of the output tensors. Defaults to torch.long.

        Returns:
            tuple: A tuple containing three elements:
                - padded (torch.Tensor): A tensor of shape (len(sequences), max_seq_length) 
                containing the padded sequences.
                - sequence_lengths (torch.Tensor): A tensor containing the original lengths of each sequence.
                - mask (torch.Tensor): A tensor of the same shape as `padded`, 
                where elements are 1 if part of the original sequence and 0 if padded.
        """
        # Calculate the original lengths of all sequences
        sequence_lengths = torch.LongTensor([len(seq) for seq in sequences])

        # Find the maximum sequence length
        max_seq_length = sequence_lengths.max().item()

        # Create a tensor filled with the pad_token up to the max sequence length for all sequences
        padded = torch.full((len(sequences), max_seq_length),
                            pad_token, dtype=dtype)

        # Initialize a mask tensor with zeros, indicating all elements are initially considered padding
        mask = torch.zeros(len(sequences), max_seq_length, dtype=dtype)

        # Iterate over each sequence to set the actual values and update the mask
        for i, seq in enumerate(sequences):

            # Set the actual sequence values in the `padded` tensor up to the length of the sequence
            padded[i, :sequence_lengths[i]] = torch.tensor(
                list(seq), dtype=dtype)

            # Update the mask to 1 for positions corresponding to the actual sequence elements
            mask[i, :sequence_lengths[i]] = 1

        return padded, sequence_lengths, mask
