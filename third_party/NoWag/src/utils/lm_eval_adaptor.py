# import fnmatch

import torch
import transformers
from lm_eval.models.huggingface import HFLM

# adapted from https://github.com/mit-han-lab/llm-awq/tree/main


class LMEvalAdaptor(HFLM):

    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device="cuda"
        )

        self.model_name = model_name
        if "opt" in self.model_name:
            print("WARNING: Check opt output truncation for vocab size")
            raise NotImplementedError
        
        # Store the model in the HFLM _model attribute
        self._model = model
        self._model.eval()

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size
        self._max_length = max_length

        # Determine the model family
        self.model_type = "auto"
        if "llama" in model_name.lower():
            self.model_type = "llama"
        elif "qwen" in model_name.lower():
            self.model_type = "qwen"

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self._model.config, "n_ctx"):
            return self._model.config.n_ctx
        elif hasattr(self._model.config, "max_position_embeddings"):
            return self._model.config.max_position_embeddings
        elif hasattr(self._model.config, "n_positions"):
            return self._model.config.n_positions
        elif hasattr(self._model.config, "max_sequence_length"):
            return self._model.config.max_sequence_length
        else:
            print(self._model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, skip_special_tokens=True):
        """
        Decode a token sequence into text.
        
        Args:
            tokens: Token sequence to decode.
            skip_special_tokens: Whether to skip special tokens.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if self.model_type == "qwen":
                # Handle Qwen model outputs
                outputs = self._model(inps, output_hidden_states=False)
                return outputs.logits
            else:
                # Use the default handling for other models, including LLaMA
                if isinstance(
                    self._model,
                    transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
                ):
                    dec_inps = torch.cat(
                        [
                            torch.tensor(
                                self._model.generation_config.decoder_start_token_id,
                            )
                            .tile(len(inps), 1)
                            .to(inps),
                            inps,
                        ],
                        dim=1,
                    )
                    kwargs = {
                        "decoder_input_ids": dec_inps,
                    }
                else:
                    kwargs = {}
                out = self._model(inps, **kwargs)[0]
                return out
            # if "opt" in self.model_name:  # there are a few extra tokens in opt, which we should omit
            #     return out[:, :, :50257]
            # else:
            #     return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id=None, **kwargs):
        if eos_token_id is None:
            eos_token_id = self.eot_token_id
        # Remove unsupported arguments
        kwargs.pop('do_sample', None)
        kwargs.pop('stop', None)  # Remove the stop argument
        kwargs.pop('temperature', None)
        return self._model.generate(
            context, 
            max_length=max_length, 
            eos_token_id=eos_token_id, 
            do_sample=False, 
            **kwargs
        )
