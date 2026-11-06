# Continuum house rules

## Synthetic keyboard boundary

All text crossing into synthetic keyboard input MUST be treated as untrusted:
functional control characters are neutralized, length is bounded, and LLM
control instructions remain structurally separate from user-controlled text.
Normal single-line punctuation and code-editing workflows remain supported
without a new confirmation step.
