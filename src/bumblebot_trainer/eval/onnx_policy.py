"""ONNX Runtime adapter for policy-only Stockfish evaluation."""

from pathlib import Path

import numpy as np


class OnnxPolicyModel:
    """Run a policy ONNX graph and return its logits for batched board inputs.

    The graph should accept board tokens and, for encodings that need it, the
    legal-move plane. Input/output names can be overridden in the evaluation
    config when the ONNX graph uses non-standard names.
    """

    _TOKEN_ALIASES = {'tokens', 'token', 'input', 'x', 'board', 'boards', 'positions'}
    _LEGAL_ALIASES = {'legal', 'legal_mask', 'legal_moves', 'legalmoves'}
    _POLICY_ALIASES = {'policy', 'policy_logits', 'logits', 'policy_output'}

    def __init__(
        self,
        checkpoint_path: str | Path,
        input_size: int,
        tokens_input: str | None = None,
        legal_input: str | None = None,
        policy_output: str | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                'Loading an ONNX checkpoint for Stockfish evaluation requires '
                'onnxruntime; install onnxruntime, onnxruntime-gpu, or an '
                'onnxruntime-rocm build appropriate for your hardware.'
            ) from exc

        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            preferred_provider = 'CUDAExecutionProvider'
        elif 'ROCMExecutionProvider' in available:
            preferred_provider = 'ROCMExecutionProvider'
        else:
            preferred_provider = 'CPUExecutionProvider'
        providers = [preferred_provider]
        if preferred_provider != 'CPUExecutionProvider' and 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        self.session = ort.InferenceSession(str(checkpoint_path), providers=providers)
        self.providers = self.session.get_providers()
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f'ONNX model has no inputs: {checkpoint_path}')

        self.input_metadata = {item.name: item for item in inputs}
        self.tokens_input = self._resolve_input(
            tokens_input,
            self._TOKEN_ALIASES,
            expected_shape=(64, input_size),
            fallback=inputs[0].name,
        )
        remaining = [name for name in self.input_metadata if name != self.tokens_input]
        self.legal_input = self._resolve_input(
            legal_input,
            self._LEGAL_ALIASES,
            expected_shape=(64, 64),
            fallback=remaining[0] if len(remaining) == 1 else None,
        )
        if self.legal_input == self.tokens_input:
            raise ValueError('ONNX token and legal inputs must be distinct')

        unhandled = set(self.input_metadata) - {self.tokens_input, self.legal_input}
        if unhandled:
            raise ValueError(
                'Unsupported extra ONNX inputs: ' + ', '.join(sorted(unhandled))
                + '. Stockfish inference supports board tokens and an optional legal plane.'
            )

        outputs = self.session.get_outputs()
        if not outputs:
            raise ValueError(f'ONNX model has no outputs: {checkpoint_path}')
        output_by_name = {item.name: item for item in outputs}
        if policy_output is not None:
            if policy_output not in output_by_name:
                raise ValueError(
                    f'ONNX output {policy_output!r} not found; outputs are '
                    f'{list(output_by_name)}'
                )
            self.policy_output = policy_output
        else:
            named_policy = [
                item for item in outputs
                if item.name.lower() in self._POLICY_ALIASES
            ]
            policy_shape = [
                item for item in outputs
                if item.shape and item.shape[-1] == 4288
            ]
            self.policy_output = (
                named_policy[0].name if named_policy
                else policy_shape[0].name if policy_shape
                else outputs[0].name
            )

        self.fixed_batch_size = self._fixed_batch_size(self.input_metadata[self.tokens_input])
        self.input_dtypes = {
            name: self._numpy_dtype(item.type)
            for name, item in self.input_metadata.items()
        }

    @staticmethod
    def _numpy_dtype(onnx_type: str):
        types = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(double)': np.float64,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': np.bool_,
        }
        try:
            return types[onnx_type]
        except KeyError as exc:
            raise TypeError(f'Unsupported ONNX input type: {onnx_type}') from exc

    @staticmethod
    def _fixed_batch_size(metadata) -> int | None:
        if metadata.shape and isinstance(metadata.shape[0], int):
            return metadata.shape[0]
        return None

    def _resolve_input(self, configured_name, aliases, expected_shape, fallback=None):
        if configured_name is not None:
            if configured_name not in self.input_metadata:
                raise ValueError(
                    f'ONNX input {configured_name!r} not found; inputs are '
                    f'{list(self.input_metadata)}'
                )
            return configured_name

        for name in self.input_metadata:
            if name.lower() in aliases:
                return name
        for name, metadata in self.input_metadata.items():
            shape = metadata.shape
            if len(shape) == 3 and tuple(shape[-2:]) == expected_shape:
                return name
        if fallback is not None:
            return fallback
        return None

    def predict(self, tokens: np.ndarray, extra: dict[str, np.ndarray]) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=self.input_dtypes[self.tokens_input])
        feed_extra = {}
        if self.legal_input is not None:
            if 'legal' not in extra:
                raise ValueError(
                    f'ONNX graph expects legal input {self.legal_input!r}, but the '
                    'selected board encoding did not produce a legal plane.'
                )
            feed_extra[self.legal_input] = np.asarray(
                extra['legal'], dtype=self.input_dtypes[self.legal_input]
            )
        elif extra and set(extra) - {'legal'}:
            raise ValueError(f'Unsupported encoded inputs: {sorted(set(extra) - {"legal"})}')

        batch_size = tokens.shape[0]
        if self.fixed_batch_size is None:
            return self._run(tokens, feed_extra)

        logits = []
        fixed = self.fixed_batch_size
        for start in range(0, batch_size, fixed):
            token_chunk = tokens[start:start + fixed]
            actual_size = len(token_chunk)
            chunk_extra = {
                name: value[start:start + fixed]
                for name, value in feed_extra.items()
            }
            if actual_size < fixed:
                pad_count = fixed - actual_size
                token_chunk = np.concatenate(
                    [token_chunk, np.repeat(token_chunk[-1:], pad_count, axis=0)], axis=0
                )
                chunk_extra = {
                    name: np.concatenate(
                        [value, np.repeat(value[-1:], pad_count, axis=0)], axis=0
                    )
                    for name, value in chunk_extra.items()
                }
            logits.append(self._run(token_chunk, chunk_extra)[:actual_size])
        return np.concatenate(logits, axis=0)

    def _run(self, tokens: np.ndarray, extra: dict[str, np.ndarray]) -> np.ndarray:
        feeds = {self.tokens_input: tokens, **extra}
        output = self.session.run([self.policy_output], feeds)[0]
        if output.ndim != 2 or output.shape[0] != tokens.shape[0]:
            raise ValueError(
                f'Expected policy logits shaped [batch, classes], got {output.shape}'
            )
        return output
