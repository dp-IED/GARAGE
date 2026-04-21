"""
LM Studio HTTP Server Inference Client

This module provides an LMInference class that connects to LM Studio's HTTP server
and provides a compatible interface for MLX-style model usage.
"""

import requests
from typing import List, Dict, Optional, Iterator
import json


class LMInference:
    """
    LM Studio HTTP Server inference client.

    Provides a drop-in replacement for MLX model/tokenizer pairs by connecting
    to LM Studio's HTTP server API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "granite-4.0-h-micro-GGUF",
        timeout: int = 300,
    ):
        """
        Initialize LM Studio inference client.

        Args:
            base_url: Base URL for LM Studio HTTP server (default: http://localhost:1234/v1)
            model_name: Name of the model loaded in LM Studio
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.chat_completions_url = f"{self.base_url}/chat/completions"

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template to messages.

        For LM Studio, we don't need to format the prompt ourselves as the API
        handles chat templates. This method returns a formatted string representation
        for compatibility, but the actual formatting is done by LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt (ignored, kept for compatibility)

        Returns:
            Formatted prompt string (for logging/debugging)
        """
        # Format messages for display/logging purposes
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (None = no limit)
            temperature: Sampling temperature
            stream: Whether to stream the response (not yet implemented)
            **kwargs: Additional parameters (repetition_penalty, etc. - may be ignored)

        Returns:
            Generated text
        """
        # Convert prompt to messages format
        # If prompt contains role markers, parse them; otherwise treat as user message
        messages = self._parse_prompt_to_messages(prompt)

        return self.chat_completions(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (None = no limit)
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        # Add any additional parameters that LM Studio supports
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "repetition_context_size" in kwargs:
            # LM Studio / llama.cpp often use repeat_penalty_last_n for this
            payload["repeat_penalty_last_n"] = kwargs["repetition_context_size"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]

        try:
            response = requests.post(
                self.chat_completions_url,
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            # Get more details about the error
            error_details = ""
            try:
                error_response = e.response.json()
                error_details = f" Response: {error_response}"
            except:
                error_details = f" Response text: {e.response.text[:200]}"
            raise RuntimeError(
                f"Failed to connect to LM Studio at {self.base_url}. "
                f"Make sure LM Studio is running with the HTTP server enabled and the model '{self.model_name}' is loaded. "
                f"Error: {e}{error_details}"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to LM Studio at {self.base_url}. "
                f"Make sure LM Studio is running with the HTTP server enabled. Error: {e}"
            )
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response format from LM Studio: {e}. Response: {response.text}"
            )

    def _parse_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Parse a prompt string into messages format.

        Attempts to detect role markers in the prompt. If no roles are found,
        treats the entire prompt as a user message.

        Args:
            prompt: Input prompt string

        Returns:
            List of message dicts
        """
        # Simple heuristic: if prompt contains role markers, parse them
        lines = prompt.split("\n")
        messages = []
        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for role markers
            if line.startswith("system:"):
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                current_role = "system"
                current_content = [line[7:].strip()]
            elif line.startswith("user:"):
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                current_role = "user"
                current_content = [line[5:].strip()]
            elif line.startswith("assistant:"):
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                current_role = "assistant"
                current_content = [line[9:].strip()]
            else:
                if current_content:
                    current_content.append(line)
                else:
                    # No role detected yet, treat as user message
                    current_role = "user"
                    current_content = [line]

        # Add final message
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": "\n".join(current_content)}
            )

        # If no messages were created, treat entire prompt as user message
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        return messages

    def _handle_stream_response(self, response: requests.Response) -> str:
        """
        Handle streaming response from LM Studio.

        Args:
            response: Streaming response object

        Returns:
            Complete generated text
        """
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                    except json.JSONDecodeError:
                        continue
        return full_text


def load_llm_model(
    model_name: str = "granite-4.0-h-micro-GGUF",
    base_url: str = "http://localhost:1234/v1",
) -> LMInference:
    """
    Load LLM model via LM Studio HTTP server.

    This function provides a drop-in replacement for MLX's load_llm_model.
    Instead of returning (model, tokenizer), it returns a single LMInference
    instance that can be used as both model and tokenizer.

    Args:
        model_name: Name of the model loaded in LM Studio
        base_url: Base URL for LM Studio HTTP server

    Returns:
        LMInference instance (can be used as both model and tokenizer)
    """
    print(f"Connecting to LM Studio at {base_url}")
    print(f"Model: {model_name}")

    inference = LMInference(base_url=base_url, model_name=model_name)

    # Test connection by making a simple request
    try:
        test_response = inference.chat_completions(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.1,
        )
        print("✓ Connected to LM Studio successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify connection: {e}")
        print("  Make sure LM Studio is running with the HTTP server enabled")

    return inference
