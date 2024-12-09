'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    generation_config.py
'''
import random

from typing import Union, List


from transformers import GenerationConfig

as_rand_max_int = 0x7FFFFFFF  # use int32 max value to generate rand number


class GenerationConfigAdapter:
    @staticmethod
    def from_hf(hf_gen_config: GenerationConfig):

        ## hf define : https://huggingface.co/docs/transformers/main_classes/text_generation
        as_gen_cfg_dict = {
            "top_k": hf_gen_config.top_k,
            "top_p": hf_gen_config.top_p,
            "do_sample": hf_gen_config.do_sample,
            "early_stopping": hf_gen_config.early_stopping,
            "repetition_penalty": hf_gen_config.repetition_penalty,
            "presence_penalty": 0.0, # hf don't have this define.
            "length_penalty": hf_gen_config.length_penalty,
            "temperature": hf_gen_config.temperature,
            "min_length": hf_gen_config.min_length,
            "no_repeat_ngram_size": hf_gen_config.no_repeat_ngram_size,
            "eos_token_id": hf_gen_config.eos_token_id,
            "seed": random.randint(1, as_rand_max_int)
        }
        return as_gen_cfg_dict


class ASGenerationConfigBuilder:
    """
    AllSpark Generation configuration Builder class for generating text with various control parameters.
    The Generation config can create by the build() member function

    Attributes(key name):
        do_sample (bool): Flag to enable/disable sampling in generation. Currently, sampling must be enabled.
        num_beams (int): Number of beams to use in beam search. Note: Beam search is not supported in the current version.
        num_return_sequences (int): Number of sequences to return in beam search. Not functional in the current version.

        early_stopping (bool): If True, generation stops when the EOS token is encountered.
        stop_words_ids (List[List[int]]): List of word IDs that signal the end of generation.
        eos_token_id (int): ID of the EOS (end of sequence) token, to be specified based on your model.

        seed (int): Seed for random number generation to ensure reproducibility.
        bad_words_ids (List[List[int]]): IDs of words to avoid generating, suppressing their occurrence.
        temperature (float): Temperature for sampling, controlling the randomness in generation.
        top_k (int): Top-K sampling parameter, limiting the selection of next tokens.
        top_p (float): Top-P sampling parameter for nucleus sampling.
        repetition_penalty (float): Penalty applied to repeated words.
        length_penalty (float): Penalty based on the length of the generated sequence.
        presence_penalty (float): Penalty for the presence of certain words in the output.
        suppress_repetition_in_generation (bool): If True, uses presence_penalty to suppress word repetition.
        no_repeat_ngram_size (int): Size of n-grams that should not repeat in the generated text.
        logprobs (bool): If True, returns log probabilities of generated tokens. Not supported by some models.
        top_logprobs (int): Specifies number of tokens with log probabilities to return if logprobs is True.

        min_length (int): Minimum length of the generated text. Set to 0 will disable this constraint.
        max_length (int): Maximum total length of generated text, including both prefill and generation parts.

        lora_name (str): Name of the LoRA adaptation, if applicable.
        mm_info (MultiMediaInfo): Multimedia information, specific to certain use cases.
    """
    def __init__(self, hf_gen_config: GenerationConfig = None, seed=random.randint(1, as_rand_max_int), eos_token_id = 0):
        """
        Create a generate config, if hf_gen_config is None, use provided eos token, otherwise use hf config's eos.
        Args:
            hf_gen_config: huggingface generation config.
            seed: default seed.
            eos_token_id: provided eos token, only useful when hf config is not available
        """
        if hf_gen_config == None:

            self.dict_store = {'do_sample': True, "seed": seed, "eos_token_id": eos_token_id}
        else:
            self.dict_store = GenerationConfigAdapter.from_hf(hf_gen_config)

            # update user's seed
            self.dict_store["seed"] = seed

            # handle eos_token_id, use generation config default value, if it's a int, use eos token id, if it's a list,
            # set eos_token_id and stop_words_ids
            self.process_eos_tokens(hf_gen_config.eos_token_id, self.dict_store)

            # currently we don't support beam search, so raise error when user want to enable beam search
            if hf_gen_config.num_return_sequences != 1:
                raise ValueError("Generation Config: hf_gen_config.num_return_sequences != 1 not supported.")
            if not hf_gen_config.do_sample:
                print("Warn: Generation Config: hf_gen_config.do_sample == False, only sampling mode is supported, "
                      "will change to do_sample = True.")
                self.dict_store["do_sample"] = True

            # make sure it's open with do sample and early stopping, some model config is mess up, those config will off
            self.do_sample()
            self.early_stopping()

    @staticmethod
    def process_eos_tokens(hf_gen_config_token_id, out_dict):
        if isinstance(hf_gen_config_token_id, int):
            out_dict["eos_token_id"] = hf_gen_config_token_id
        elif isinstance(hf_gen_config_token_id, list) and all(
                isinstance(item, int) for item in hf_gen_config_token_id):
            out_dict["eos_token_id"] = hf_gen_config_token_id[0]
            stop_words_ids = []
            # stop words id support a multiple id list for each stop case, so it's a 2d array.
            for item in hf_gen_config_token_id:
                stop_words_ids.append([item])

            print(f"stop word ids: {stop_words_ids}")

            out_dict["stop_words_ids"] = stop_words_ids

    def update(self, update_fields):
        """
        Inplace Update generation config member by a key,value dict.

        Args:
            update_fields: key, value dict to the update field.

        """
        self.dict_store.update(update_fields)

    def do_sample(self, flag=True):
        """
        change do sample flag, turn on this flag make engine work in sampling mode rather than beam search.
        Args:
            flag: enable or disable do sampling, [default: True], Currently AllSpark only support this mode is on.

        Returns: self

        """
        self.update({'do_sample': flag})
        return self
    def with_mm_info(self, info : "MultiMediaInfo"):
        """
        Add multiple  media information to this request.
        Args:
            info: the multi-media information of this request
        """
        self.update('mm_info', info)
        return self

    def early_stopping(self, flag=True):
        """
        Change early stopping flag, turn on this generate will stop when eos token is generated.
        Args:
            flag: enable or disable early stop, [default:True]

        Returns: self

        """
        self.update({'early_stopping': flag})
        return self

    def log_probs(self, enable_logprobs, top_logprobs):
        self.update({"logprobs": enable_logprobs, "top_logprobs": top_logprobs})
        return self

    def build(self):
        """
        Return the dict format of generation config.

        Returns: python dict format generation config.

        """
        # we need a shallow copy here so that doing 'pop["vocab"]' won't forever remove vocab from builder
        return self.dict_store.copy()




