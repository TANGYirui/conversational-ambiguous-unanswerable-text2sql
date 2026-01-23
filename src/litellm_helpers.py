import os
import time
import random
from loguru import logger

import litellm
from litellm.router import RetryPolicy, AllowedFailsPolicy
from litellm import Router
from litellm.caching import Cache
from typing import List

# Check required AWS environment variables
AWS_BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
AWS_REGION = os.getenv("AWS_BEARER_TOKEN_BEDROCK_REGION")

if not AWS_BEARER_TOKEN:
    error_msg = (
        "ERROR: AWS_BEARER_TOKEN_BEDROCK environment variable is not set!\n"
        "Please ensure .vscode/.env file exists and contains AWS_BEARER_TOKEN_BEDROCK.\n"
        "If running test_per_category.sh, it should automatically load the env file.\n"
        "If running Python directly, export the variables first:\n"
        "  export $(grep -v '^#' .vscode/.env | xargs)"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

if not AWS_REGION:
    error_msg = (
        "ERROR: AWS_BEARER_TOKEN_BEDROCK_REGION environment variable is not set!\n"
        "Please ensure .vscode/.env file contains AWS_BEARER_TOKEN_BEDROCK_REGION."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Set AWS_REGION_NAME for LiteLLM (it expects this variable name)
os.environ["AWS_REGION_NAME"] = AWS_REGION
logger.info(f"✓ AWS Bedrock credentials loaded successfully")
logger.info(f"✓ AWS Region: {AWS_REGION}")
logger.info(f"✓ Bearer Token Random Sample: {random.sample(AWS_BEARER_TOKEN, 3)}... (length: {len(AWS_BEARER_TOKEN)})")

# Configure litellm cache
litellm.cache = Cache(type="disk", disk_cache_dir="../.vscode/.litellm_cache")
# litellm.cache.set_cache(ttl=60*60*24*7)
litellm.enable_cache()
litellm.set_verbose = False
litellm.suppress_debug_info = True

# Configuration
NUM_RETRIES = 15
RETRY_AFTER = 30


DEFAULT_RETRY_POLICY = RetryPolicy(
    ContentPolicyViolationErrorRetries=3,         # run 3 retries for ContentPolicyViolationErrors
    AuthenticationErrorRetries=0,                 # run 0 retries for AuthenticationErrorRetries
    BadRequestErrorRetries=1,
    TimeoutErrorRetries=2,
    RateLimitErrorRetries=30,
)

DEFAULT_ALLOWED_FAILS_POLICY = AllowedFailsPolicy(
    ContentPolicyViolationErrorAllowedFails=100, # Allow 100 ContentPolicyViolationError before cooling down a deployment
    RateLimitErrorAllowedFails=100,               # Allow 100 RateLimitErrors before cooling down a deployment
)


# Model list - Using AWS Bedrock models with bearer token authentication
# Note: Removed "us." prefix for compatibility with litellm 1.42.5
MODEL_LIST = [
    {
        "model_name": "llama3-1-70b",
        "litellm_params": {
            "model": "bedrock/meta.llama3-1-70b-instruct-v1:0"
        }
    },
    {
        "model_name": "llama3-1-8b",
        "litellm_params": {
            "model": "bedrock/meta.llama3-1-8b-instruct-v1:0",
        }
    },
    {
        "model_name": "claude-3-sonnet",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        }
    },
    {
        "model_name": "claude-3-haiku",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
        }
    },
    {
        "model_name": "claude-3-5-sonnet",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        }
    },
    {
        "model_name": "mixtral-8x7b",
        "litellm_params": {
            "model": "bedrock/mistral.mixtral-8x7b-instruct-v0:1"
        }
    },
    {
        "model_name": "mixtral-large-2",
        "litellm_params": {
            "model": "bedrock/mistral.mistral-large-2407-v1:0"
        }
    },
    {
        "model_name": "llama3-1-405b",
        "litellm_params": {
            "model": "bedrock/meta.llama3-1-405b-instruct-v1:0"
        }
    },
]


def get_litellm_router(cache_responses=True):
    """
    Create a LiteLLM router with AWS Bedrock models using bearer token authentication.

    Required environment variables:
    - AWS_BEARER_TOKEN_BEDROCK: Your AWS Bedrock bearer token
    - AWS_BEARER_TOKEN_BEDROCK_REGION: AWS region (e.g., us-west-2)

    Args:
        cache_responses: Whether to cache responses (default: True)

    Returns:
        Router: Configured LiteLLM router instance
    """
    router = Router(
        model_list=MODEL_LIST,
        cache_responses=cache_responses,
        # retry_after=retry_after,  # in minutes
        # num_retries=num_retries,
        # retry_policy=DEFAULT_RETRY_POLICY,
        # allowed_fails_policy=DEFAULT_ALLOWED_FAILS_POLICY,
        # cooldown_time=10,
        # set_verbose=True,
        # debug_level="DEBUG",  # defaults to INFO
    )
    logger.info(f"Initialized LiteLLM router with {len(MODEL_LIST)} models using bearer token authentication")
    return router


# Default router instance
DEFAULT_ROUTER = get_litellm_router()


def router_completion_with_ratelimit_retry(model: str = "claude-3-5-sonnet", messages: List = [], router=DEFAULT_ROUTER, temperature=0.0, top_p=1, max_tokens=2048, num_retries=NUM_RETRIES, retry_after=RETRY_AFTER, system: str = None):
    """
    Call LiteLLM router with automatic rate limit retry handling.

    Available models (from MODEL_LIST above):
    - claude-3-5-sonnet (default, recommended)
    - claude-3-sonnet
    - claude-3-haiku
    - llama3-1-405b, llama3-1-70b, llama3-1-8b
    - mixtral-large-2, mixtral-8x7b
    """
    start_time = time.time()
    ex = None
    if system:
        messages = [{"role": "system", "content": system}] + messages
    for ith_retry in range(num_retries):
        response_obj = None
        try:
            response_obj = router.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response_obj.choices[0].message.content
        except Exception as ex:
            if isinstance(ex, litellm.RateLimitError) or isinstance(ex, IndexError) or isinstance(ex, litellm.ServiceUnavailableError) or isinstance(ex, litellm.APIConnectionError):
                sleep_time = retry_after + ith_retry * (0.01 + random.random()) * retry_after
                time.sleep(sleep_time)
                if (1 + ith_retry) % 6 == 0:
                    logger.info(f"Sleep for {sleep_time} for {1 + ith_retry}th retry. Time lapsed since initial call: {time.time() - start_time}")
                    pass
            else:
                time.sleep(15)
                logger.error(f"Unexpected exception during LLM router completion: {ex}")
                if ith_retry > num_retries / 3:
                    # Break exception chain to avoid pickle issues with litellm exceptions
                    raise Exception(f"Completion exception after retries. Exception message: {type(ex).__name__}: {str(ex)}") from None
    logger.error(f"Unexpected exception during LLM router completion: {ex}. We retried {ith_retry + 1} times before raising this exception.")
    # Break exception chain to avoid pickle issues with litellm exceptions
    raise Exception(f"Unknown completion exception after retries. Exception message: {type(ex).__name__}: {str(ex)}") from None


def convert_claude_msg_list_to_litellm_msg_list(claude_msg_list):
    litellm_msg_list = []
    for msg in claude_msg_list:
        if isinstance(msg['content'], list) and isinstance(msg['content'][0], dict):
            litellm_msg_list.append(
                {"role": msg['role'], "content": msg['content'][0]['text']}
            )
        elif isinstance(msg['content'], str):
            litellm_msg_list.append(
                {"role": msg['role'], "content": msg['content']}
            )
    return litellm_msg_list
