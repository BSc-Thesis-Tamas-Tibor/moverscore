"""Module for the MoverScore class."""

import string
from typing import List, NamedTuple

from dataclasses import dataclass
import numpy as np
from pyemd import emd_with_flow
from torch import Tensor, baddbmm, norm, cat
from torch.cuda import is_available

from src.embedder import TextEmbedder


class TextData(NamedTuple):
    """
    A simple data structure for encapsulating the data associated with text inputs.

    This NamedTuple is used to store the embedding tensor, IDF (Inverse Document Frequency) weights,
    and tokenized representation of text data.

    Attributes:
        embedding (Tensor): The tensor representation of the text's embedding.
        idf (Tensor): The tensor containing IDF weights for each token in the text. These weights
                      measure the importance of a token based on its frequency across a corpus of documents.
        tokens (List[str]): A list of tokenized text, where each element is a token extracted from the original text.
    """

    embedding: Tensor
    idf: Tensor
    tokens: List[str]


@dataclass
class MoverScoreConfig:
    """
    A data class representing the configuration settings for calculating the MoverScore
    metric.

    Attributes:
        stop_words (list): A list of words to be ignored during the calculation of
            MoverScore. These words are excluded from the embedding process.
        remove_subwords (bool): Indicates whether subwords (e.g., parts of a word
            introduced by some tokenization processes) should be removed from the
            embedding process. Set to True to remove subwords.
        batch_size (int): The number of text pairs (reference and hypothesis) to process
            in each batch. Larger batch sizes can improve processing efficiency but require
            more memory. The optimal size depends on the available computation resources.
    """

    stop_words: List[str]
    remove_subwords: bool = True
    batch_size: int = 16


class MoverScore:
    """
    A class designed to calculate the MoverScore metric, which quantifies the semantic
    similarity between reference texts and corresponding hypothesis texts. MoverScore
    utilizes embeddings from pre-trained transformer models to compute the Earth Mover's
    Distance (EMD) between the embedded representations of text pairs.

    Attributes:
        config (MoverScoreConfig): Configuration settings for the MoverScore calculation,
            including reference and hypothesis texts, batch processing size,
            and options to remove subwords and ignore specified stop words.
        device (str): The computational device ('cuda' or 'cpu') determined based on
            the availability of CUDA, optimizing for GPU acceleration if possible.
        embedder (TextEmbedder): An instance of the TextEmbedder class used for generating
            embeddings from text using a pre-trained transformer model.

    Methods:
        __call__: Computes MoverScores for all configured pairs of reference and
            hypothesis texts.
        process_batch(batch_start): Processes a single batch of texts, generating embeddings
            and calculating MoverScores for each text pair in the batch.
        embed_texts(batch_refs, batch_hyps): Generates embeddings for a batch of reference
            and hypothesis texts.
        calculate_batch_scores: Calculates MoverScores for a batch based on the embeddings
            and additional data such as IDF weights and attention masks.
        check_tokens(tokens, embeddings, idf_weights): Filters out embeddings and IDF weights
            for specified tokens, typically used for removing subwords or stop words.
        should_remove_token(token): Determines whether a token should be removed based on
            the configuration settings.
        calculate_score: Calculates the MoverScore for a single pair of texts based on their
            embeddings and the Earth Mover's Distance.
        safe_divide: Safely divides two numbers to avoid division by zero errors.
        calc_dist: Calculates the pairwise distance between two sets of embeddings, utilizing
            the squared L2 norm.
    """

    def __init__(self, config: MoverScoreConfig) -> None:
        """
        Initializes the MoverScore instance with specified configurations.

        Args:
            config (MoverScoreConfig): Configuration settings including reference and hypothesis texts,
                                       stop words, n-gram size, subword removal flag, and batch size.
        """
        # Auto-selects the computation device
        self.device = "cuda" if is_available() else "cpu"

        # Configuration settings for scoring
        self.config = config

        # Embedding generator object
        self.embedder = TextEmbedder()

    def __call__(self, references: List[str], hypothesis: List[str]) -> List[float]:
        """
        Calculates MoverScores for all pairs of reference and hypothesis texts in batches.

        Returns:
            A list of calculated MoverScores, one for each pair of reference and hypothesis texts.
        """
        all_scores = []

        # Iterates over the texts in batches as per the batch size configuration
        for batch_start in range(0, len(references), self.config.batch_size):
            scores = self.process_batch(
                batch_start, references, hypothesis
            )  # Process each batch
            all_scores.extend(scores)  # Collect scores from each batch
        return all_scores

    def process_batch(
        self, batch_start: int, references: List[str], hypothesis: List[str]
    ) -> List[float]:
        """
        Processes a single batch of texts to calculate their MoverScores.

        Args:
            batch_start (int): The starting index of the batch in the reference and hypothesis lists.
            references (List[str]): The list of the reference document strings
            hypothesis (List[str]): The list of the hypothesis document strings

        Returns:
            A list of MoverScores for the processed batch.
        """
        # Slices the reference and hypothesis lists to get the current batch
        batch_refs = references[batch_start: batch_start + self.config.batch_size]
        batch_hyps = hypothesis[batch_start: batch_start + self.config.batch_size]

        # Embed texts from the batch
        ref_data, hyp_data = self.embed_texts(batch_refs, batch_hyps)

        # Calculate and return scores
        return self.calculate_batch_scores(ref_data=ref_data, hyp_data=hyp_data)

    def embed_texts(self, batch_refs: List[str], batch_hyps: List[str]) -> tuple:
        """
        Generates embeddings for a batch of reference and hypothesis texts.

        Args:
            batch_refs (list): Reference texts for the current batch.
            batch_hyps (list): Hypothesis texts for the current batch.

        Returns:
            A tuple containing embeddings and other relevant data for both reference and hypothesis texts.
        """

        # Embed reference texts
        ref_data = TextData._make(self.embedder(sentences=batch_refs))

        # Embed hypothesis texts
        hyp_data = TextData._make(self.embedder(sentences=batch_hyps))

        return ref_data, hyp_data

    def calculate_batch_scores(
        self, ref_data: TextData, hyp_data: TextData
    ) -> List[float]:
        """
        Calculates MoverScores for a batch based on embeddings and additional data.

        Args:
            ref_embedding, hyp_embedding (Tensor): Embeddings for reference and hypothesis texts.
            ref_idf, hyp_idf (Tensor): IDF weights for reference and hypothesis texts.
            ref_tokens, hyp_tokens (list): Tokenized versions of reference and hypothesis texts.

        Returns:
            A list of MoverScores for each pair in the batch.
        """
        # Determine batch size from the number of reference tokens
        batch_size = len(ref_data.tokens)

        # Filter out specified tokens from embeddings and IDF weights
        ref_embedding, ref_idf = self.check_tokens(
            ref_data.tokens, ref_data.embedding, ref_data.idf
        )
        hyp_embedding, hyp_idf = self.check_tokens(
            hyp_data.tokens, hyp_data.embedding, hyp_data.idf
        )

        # Normalize and concatenate embeddings for distance calculation
        raw = cat([ref_embedding, hyp_embedding], 1)
        raw.div_(norm(raw, dim=-1).unsqueeze(-1) + 1e-30)
        distance_matrix = self.calc_dist(raw, raw).double().cpu().numpy()

        # Calculate scores for each text pair in the batch
        scores = [
            MoverScore.calculate_score(i, raw, ref_idf, hyp_idf, distance_matrix)
            for i in range(batch_size)
        ]

        return scores

    def check_tokens(
        self, tokens: List[str], embeddings: Tensor, idf_weights: Tensor
    ) -> tuple:
        """
        Filters out specified tokens from embeddings and IDF weights.

        Args:
            tokens (list): List of tokens for each text.
            embeddings (Tensor): Embeddings corresponding to the tokens.
            idf_weights (Tensor): IDF weights corresponding to the tokens.

        Returns:
            A tuple of filtered embeddings and IDF weights.
        """

        for i, token_list in enumerate(tokens):
            # Identify tokens to remove based on configuration
            ids = [k for k, w in enumerate(token_list) if self.should_remove_token(w)]
            embeddings[i, ids, :] = 0  # Zero out embeddings for removed tokens
            idf_weights[i, ids] = 0  # Zero out IDF weights for removed tokens
        return embeddings, idf_weights  # Return filtered embeddings and IDF weights

    def should_remove_token(self, token: str) -> bool:
        """
        Checks if a token meets the criteria for removal based on configuration settings.

        Args:
            token (str): The token to check.

        Returns:
            True if the token should be removed, False otherwise.
        """
        # Remove token if it is a stop word, a subword, or punctuation

        is_stopword = token in self.config.stop_words
        is_subword = "##" in token and self.config.remove_subwords
        is_punctuation = token in set(string.punctuation)

        return is_stopword or is_subword or is_punctuation

    @staticmethod
    def calculate_score(
        index: int,
        raw_embeddings: Tensor,
        ref_idf: Tensor,
        hyp_idf: Tensor,
        distance_matrix: np.ndarray,
    ) -> float:
        """
        Calculates the MoverScore for a single pair of texts based on their embeddings.

        Args:
            index (int): Index of the text pair within the batch.
            raw_embeddings (Tensor): Concatenated embeddings of reference and hypothesis texts.
            ref_idf (Tensor): IDF weights for the reference text.
            hyp_idf (Tensor): IDF weights for the hypothesis text.
            distance_matrix (np.ndarray): Pre-computed distance matrix for the embeddings.

        Returns:
            The calculated MoverScore for the text pair.
        """

        # Initialize cost vectors for EMD calculation
        cost_1 = np.zeros(raw_embeddings.shape[1], dtype=float)
        cost_2 = np.zeros(raw_embeddings.shape[1], dtype=float)

        # Assign IDF weights to cost vectors
        cost_1[: len(ref_idf[index])] = ref_idf[index]
        cost_2[len(ref_idf[index]):] = hyp_idf[index]

        # Normalize the cost vectors
        cost_1 = MoverScore.safe_divide(cost_1, np.sum(cost_1))
        cost_2 = MoverScore.safe_divide(cost_2, np.sum(cost_2))
        dst = distance_matrix[index]

        # Calculate EMD and return the inverse score for similarity
        _, flow = emd_with_flow(np.array(cost_1), np.array(cost_2), dst)
        flow = np.array(flow, dtype=np.float32)

        return 1.0 / (1.0 + np.sum(flow * dst))

    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: float) -> float:
        """
        Safely divides two numbers to prevent division by zero errors.

        Args:
            numerator (float): The numerator of the division.
            denominator (float): The denominator of the division.

        Returns:
            The result of the division or 0 if the denominator is 0.
        """
        return numerator / (
            denominator + 1e-30
        )  # Add a small value to avoid division by zero

    @staticmethod
    def calc_dist(x_1: Tensor, x_2: Tensor) -> Tensor:
        """
        Computes the pairwise distance between two sets of embeddings using L2 norm.

        Args:
            x_1 (Tensor): Embeddings set 1.
            x_2 (Tensor): Embeddings set 2.

        Returns:
            A tensor representing the pairwise distances.
        """
        # Compute L2 norm-based distances between embeddings
        x1_norm = x_1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x_2.pow(2).sum(dim=-1, keepdim=True)

        # Calculate pairwise distances
        dist = (
            baddbmm(x2_norm.transpose(-2, -1), x_1, x_2.transpose(-2, -1), alpha=-2)
            .add_(x1_norm)
            .clamp_min_(1e-30)
            .sqrt_()
        )

        return dist
