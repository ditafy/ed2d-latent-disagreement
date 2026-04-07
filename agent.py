import os
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MAX_COMPLETION_TOKENS,
    MEMORY_KEEP_RECENT,
    MEMORY_SUMMARIZE_THRESHOLD,
    SUPPORT_MODELS,
)


@dataclass
class AgentResponse:
    text: str
    pooled_vector: torch.Tensor | None = None
    generated_token_count: int | None = None
    sequence_length: int | None = None


class ContextLengthExceeded(Exception):
    """Raised when prompt length exceeds model context window."""


class Agent:
    _BACKEND_CACHE: ClassVar[Dict[Tuple[str, str, str], Tuple[object, object, str]]] = {}

    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.system_prompt = ""
        self.device = self._resolve_device(os.getenv("LOCAL_MODEL_DEVICE", "auto"))
        self.dtype_name = os.getenv("LOCAL_MODEL_DTYPE", "auto")
        self.tokenizer, self.model, self.model_device = self._get_or_create_shared_backend()

    def _validate_model(self):
        """Validate if the model is supported."""
        if self.model_name not in SUPPORT_MODELS and not os.path.exists(self.model_name):
            raise ValueError(f"Model {self.model_name} not in {SUPPORT_MODELS} and path does not exist.")

    def _resolve_dtype(self):
        if self.dtype_name == "float16":
            return torch.float16
        if self.dtype_name == "bfloat16":
            return torch.bfloat16
        if self.dtype_name == "float32":
            return torch.float32
        if self.device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    def _resolve_device(self, device_name: str) -> str:
        if device_name == "cuda":
            return "cuda"
        if device_name == "mps":
            return "mps"
        if device_name == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_or_create_shared_backend(self):
        self._validate_model()
        cache_key = (self.model_name, self.device, self.dtype_name)
        if cache_key in self._BACKEND_CACHE:
            return self._BACKEND_CACHE[cache_key]

        torch_dtype = self._resolve_dtype()
        model_source = self.model_name

        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.to(self.device)
        model.eval()

        backend = (tokenizer, model, self.device)
        self._BACKEND_CACHE[cache_key] = backend
        return backend

    def set_meta_prompt(self, prompt: str):
        self.system_prompt = prompt

    def summarize_memory(self, memory: list) -> str:
        summary_messages = [
            {"role": "system", "content": "Summarize the following debate history into a concise paragraph."},
            *memory,
            {"role": "user", "content": "Please provide the summary."},
        ]
        try:
            return self._generate_from_messages(summary_messages, temperature=0.3, max_new_tokens=256).text
        except Exception as exc:
            print(f"[⚠️ Summarization Failed] {exc}")
            return "[Summary unavailable]"

    def _prepare_memory_context(self, shared_memory: list) -> list:
        """Prepare memory context, summarize when necessary."""
        if len(shared_memory) <= MEMORY_SUMMARIZE_THRESHOLD:
            return shared_memory

        recent = shared_memory[-MEMORY_KEEP_RECENT:]
        summary = self.summarize_memory(shared_memory[:-MEMORY_KEEP_RECENT])
        return [{"role": "system", "content": f"[Debate Summary]: {summary}"}] + recent

    def _limit_tokens(self, max_tokens: int) -> int:
        return max(1, min(max_tokens, MAX_COMPLETION_TOKENS))

    def _calculate_max_tokens(self, rendered_input: str) -> int:
        encoded = self.tokenizer(rendered_input, return_tensors="pt")
        input_len = encoded["input_ids"].shape[1]
        max_context = getattr(self.model.config, "max_position_embeddings", 32768)
        available_tokens = max_context - input_len
        if available_tokens <= 0:
            raise ContextLengthExceeded(
                f"Context tokens {input_len} exceed limit {max_context} for model {self.model_name}"
            )
        return self._limit_tokens(available_tokens)

    def _render_input_text(self, messages: List[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return rendered

        sections = []
        for message in messages:
            role = message["role"].upper()
            sections.append(f"{role}:\n{message['content']}")
        sections.append("ASSISTANT:")
        return "\n\n".join(sections)

    def _generate_text(self, rendered_input: str, temperature: float, max_new_tokens: int) -> Tuple[str, torch.Tensor, int]:
        inputs = self.tokenizer(rendered_input, return_tensors="pt")
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self._limit_tokens(max_new_tokens),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated = self.model.generate(**generation_kwargs)

        generated_only_ids = generated[0, input_length:]
        generated_text = self.tokenizer.decode(generated_only_ids, skip_special_tokens=True).strip()
        new_tokens = generated.shape[1] - input_length
        return generated_text, generated, new_tokens

    def _extract_pooled_vector(self, generated_ids: torch.Tensor, input_length: int) -> torch.Tensor:
        attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        final_hidden = outputs.hidden_states[-1]
        if generated_ids.shape[1] > input_length:
            pooled_source = final_hidden[:, input_length:, :]
        else:
            pooled_source = final_hidden
        return pooled_source.mean(dim=1)

    def _generate_from_messages(
        self,
        messages: List[dict],
        temperature: float,
        max_new_tokens: int | None = None,
        return_mode: str = "text",
    ) -> AgentResponse:
        time.sleep(self.sleep_time)
        rendered_input = self._render_input_text(messages)
        effective_max_tokens = max_new_tokens or self._calculate_max_tokens(rendered_input)
        generated_text, generated_ids, generated_token_count = self._generate_text(
            rendered_input,
            temperature=temperature,
            max_new_tokens=effective_max_tokens,
        )

        pooled_vector = None
        if return_mode == "analysis":
            original_inputs = self.tokenizer(rendered_input, return_tensors="pt")
            input_length = original_inputs["input_ids"].shape[1]
            pooled_vector = self._extract_pooled_vector(generated_ids, input_length)

        return AgentResponse(
            text=generated_text,
            pooled_vector=pooled_vector,
            generated_token_count=generated_token_count,
            sequence_length=generated_ids.shape[1],
        )

    def ask(
        self,
        shared_memory: list,
        prompt: str,
        temperature: float = None,
        return_mode: str = "text",
    ):
        memory_ctx = self._prepare_memory_context(shared_memory)
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages.extend(memory_ctx)
        messages.append({"role": "user", "content": f"{self.name}: {prompt}"})

        effective_temperature = temperature if temperature is not None else self.temperature
        response = self._generate_from_messages(
            messages,
            temperature=effective_temperature,
            return_mode=return_mode,
        )
        return response.text if return_mode == "text" else response


def build_agent(cfg, model_name: str, T: float, sleep: float):
    agent = Agent(model_name, cfg.name, T, sleep)
    agent.set_meta_prompt(cfg.meta_prompt)
    return agent
