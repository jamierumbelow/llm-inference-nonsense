"""Fixed prompts shared by correctness runs across every experiment."""

DEFAULT_PROMPT = "The present King of France is"

_LONG_PARAGRAPHS = (
    "A language model turns token identifiers into vectors and passes those vectors through a "
    "stack of transformer layers. Each layer first mixes information between positions with "
    "causal attention, then transforms each position independently with a feed-forward network. "
    "Residual connections carry the previous representation around both operations, while "
    "normalization keeps the scale of the activations controlled.",
    "During inference, causal attention prevents a token from seeing anything to its right. The "
    "query at a position is compared with keys from the visible prefix, the resulting scores are "
    "normalized, and the values are combined using those weights. Grouped-query attention lets "
    "several query heads share one key and value head, reducing memory use during decoding.",
    "Rotary position embeddings encode position by rotating pairs of query and key coordinates. "
    "The rotation angle changes with both position and frequency. Llama adjusts some of these "
    "frequencies so that the model can operate beyond the context length used by the original "
    "schedule, while leaving the high-frequency components largely unchanged.",
    "The first call over a prompt is usually called prefill. A production engine stores the keys "
    "and values produced during prefill, then processes one new token at a time without repeating "
    "the earlier projections. This key-value cache saves computation but creates new questions "
    "about memory layout, allocation, batching, and movement between devices.",
    "Numerical comparisons are useful because a transformer contains many individually simple "
    "operations whose composition is easy to get subtly wrong. A misplaced reshape, an incorrect "
    "head grouping, or a positional frequency error quickly changes logits throughout the model. "
    "Small floating-point differences are expected, so correctness must be judged in relation to "
    "the precision and execution backend being used.",
)

# Repetition keeps this synthetic passage stable and close to 1,000 Llama tokens.
LONG_PASSAGE = "\n\n".join(_LONG_PARAGRAPHS * 3)

CORRECTNESS_PROMPTS = {
    # Llama's tokenizer adds the BOS token, giving this empty string a one-token input.
    "one_token": "",
    "sentence": DEFAULT_PROMPT,
    "code": """def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    "maths": "If 3x + 7 = 31, solve for x and then calculate x squared plus 2x.",
    "long_passage": LONG_PASSAGE,
}

PROMPT_SUITES = {"correctness": CORRECTNESS_PROMPTS}
