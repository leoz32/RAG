from factscore.lm import LM
import logging
import os
import sys
import time

import numpy as np
import openai

class OpenAIModel(LM):

    def __init__(
        self,
        model_name,
        cache_file=None,
        key_path="api.key",
        api_base=None,
        chat_model_name="gpt-3.5-turbo",
        instruct_model_name="gpt-3.5-turbo",
    ):
        self.model_name = model_name
        self.key_path = key_path
        self.api_base = api_base
        self.chat_model_name = chat_model_name
        self.instruct_model_name = instruct_model_name
        self.temp = 0.7
        self.save_interval = 100
        self._legacy_api = not hasattr(openai, "OpenAI")
        self._client = None
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        api_key = api_key.strip()
        if self._legacy_api:
            openai.api_key = api_key
            if self.api_base:
                openai.api_base = self.api_base
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.api_base,
            )
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            message = [{"role": "user", "content": prompt}]
            response = call_ChatGPT(
                message,
                client=self._client,
                model_name=self.chat_model_name,
                temp=self.temp,
                max_len=max_sequence_length,
            )
            output = extract_chat_content(response)
            return output, response
        elif self.model_name == "InstructGPT":
            message = [{"role": "user", "content": prompt}]
            response = call_ChatGPT(
                message,
                client=self._client,
                model_name=self.instruct_model_name,
                temp=self.temp,
                max_len=max_output_length,
            )
            output = extract_chat_content(response)
            return output, response
        else:
            raise NotImplementedError()


def extract_chat_content(response):
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content


def _is_invalid_request_error(error):
    legacy_error = getattr(getattr(openai, "error", None), "InvalidRequestError", None)
    modern_error = getattr(openai, "BadRequestError", None)
    return (
        (legacy_error is not None and isinstance(error, legacy_error))
        or (modern_error is not None and isinstance(error, modern_error))
    )


def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False, client=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            if client is None:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=message,
                    max_tokens=max_len,
                    temperature=temp,
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    max_tokens=max_len,
                    temperature=temp,
                )
            received = True
        except Exception as error:
            # print(message)
            num_rate_errors += 1
            if _is_invalid_request_error(error):
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except Exception as error:
            num_rate_errors += 1
            if _is_invalid_request_error(error):
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
