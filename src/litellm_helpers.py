import os
import time
import random
from loguru import logger

import litellm
from litellm.router import RetryPolicy, AllowedFailsPolicy
from litellm import Router
from litellm.caching import Cache
from typing import List

# ==========================================
# 1. Check required Azure environment variables
# ==========================================
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

if not AZURE_API_KEY or not AZURE_API_BASE:
    error_msg = (
        "ERROR: AZURE_API_KEY or AZURE_API_BASE environment variable is not set!\n"
        "Please ensure .vscode/.env file exists and contains Azure credentials.\n"
        "If running test_per_category.sh, it should automatically load the env file.\n"
        "If running Python directly, export the variables first:\n"
        "  export $(grep -v '^#' .vscode/.env | xargs)"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

logger.info(f"✓ Azure OpenAI credentials loaded successfully")
logger.info(f"✓ Azure API Base: {AZURE_API_BASE}")
logger.info(f"✓ Azure API Version: {AZURE_API_VERSION}")

# ==========================================
# 2. Configure litellm cache
# ==========================================
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

# ==========================================
# 3. Model list - Using Azure OpenAI models
# ==========================================
MODEL_LIST = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "azure/gpt-4o-mini",
            "api_key": AZURE_API_KEY,
            "api_base": AZURE_API_BASE,
            "api_version": AZURE_API_VERSION
        }
    },
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": AZURE_API_KEY,
            "api_base": AZURE_API_BASE,
            "api_version": AZURE_API_VERSION
        }
    },
    {
        "model_name": "gpt-35-turbo",
        "litellm_params": {
            "model": "azure/gpt-35-turbo",
            "api_key": AZURE_API_KEY,
            "api_base": AZURE_API_BASE,
            "api_version": AZURE_API_VERSION
        }
    }
]

def get_litellm_router(cache_responses=True):
    """
    Create a LiteLLM router with Azure OpenAI models.
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
    logger.info(f"Initialized LiteLLM router with {len(MODEL_LIST)} models using Azure OpenAI")
    return router

# Default router instance
DEFAULT_ROUTER = get_litellm_router()

# ==========================================
# 4. Completion Wrapper
# ==========================================
def router_completion_with_ratelimit_retry(model: str = "gpt-4o-mini", messages: List = [], router=DEFAULT_ROUTER, temperature=0.0, top_p=1, max_tokens=2048, num_retries=NUM_RETRIES, retry_after=RETRY_AFTER, system: str = None):
    """
    Call LiteLLM router with automatic rate limit retry handling.
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

# ==========================================
# 5. Helpers
# ==========================================
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