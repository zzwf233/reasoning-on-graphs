import os
import glob
import shutil

from transformers import pipeline, AutoTokenizer, AutoConfig
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
        parser.add_argument('--use_fast_tokenizer', action='store_true', help='prefer fast tokenizer for inference')


    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs)
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        local_model = isinstance(self.args.model_path, str) and os.path.isdir(self.args.model_path)
        if local_model:
            canonical_spm = os.path.join(self.args.model_path, "tokenizer.model")
            if not os.path.exists(canonical_spm):
                candidates = sorted(glob.glob(os.path.join(self.args.model_path, "*tokenizer*.model")))
                if candidates:
                    shutil.copyfile(candidates[0], canonical_spm)
        prefer_fast_tokenizer = bool(getattr(self.args, "use_fast_tokenizer", False))
        tokenizer_kwargs = {"use_fast": prefer_fast_tokenizer}
        if local_model:
            tokenizer_kwargs["local_files_only"] = True
        else:
            tokenizer_kwargs["use_auth_token"] = True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, **tokenizer_kwargs)
        except Exception as e:
            recoverable_error = (
                "NoneType" in str(e) or "piece id is out of range" in str(e)
            )
            if local_model and recoverable_error:
                tokenizer_kwargs["use_fast"] = True
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, **tokenizer_kwargs)
            else:
                raise
        if local_model and not tokenizer_kwargs.get("use_fast", False):
            config = AutoConfig.from_pretrained(
                self.args.model_path, local_files_only=True
            )
            missing_tokens = max(0, config.vocab_size - len(self.tokenizer))
            if missing_tokens > 0:
                extra_tokens = [
                    f"<extra_local_token_{i}>" for i in range(missing_tokens)
                ]
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": extra_tokens}
                )
        if local_model:
            model_kwargs.update({"local_files_only": True})
        else:
            model_kwargs.update({'use_auth_token': True})
        self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        try:
            outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
        except IndexError as e:
            if "piece id is out of range" not in str(e):
                raise
            local_model = isinstance(self.args.model_path, str) and os.path.isdir(self.args.model_path)
            if not local_model:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_path, use_fast=True, local_files_only=True
            )
            self.generator = pipeline(
                "text-generation",
                model=self.args.model_path,
                tokenizer=self.tokenizer,
                device_map="auto",
                model_kwargs={"local_files_only": True},
                torch_dtype=self.DTYPE.get(self.args.dtype, None),
            )
            outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
        return outputs[0]['generated_text'] # type: ignore
