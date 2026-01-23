import json
import os
import functools
from loguru import logger


def cache_results(cache_path, ignore_cache=False):
    """
    Decorator that caches the results of a function.
    """
    def decorator(func):
        in_memory_cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if ignore_cache:
                # logger.info(f"Ignoring cache for {func.__name__}({args}, {kwargs})")
                logger.info(f"Ignoring cache for {func.__name__}")
                result = func(*args, **kwargs)
                return result

            cache_key = generate_cache_key(args, kwargs)

            # Check in-memory cache first
            if cache_key in in_memory_cache:
                # logger.info(f"Loading cached result from memory for {func.__name__}({args}, {kwargs})")
                logger.info(f"Loading cached result from memory for {func.__name__}")
                return in_memory_cache[cache_key]            

            cache_file = get_cache_file(cache_path, func.__name__)
            cache = load_cached_objects(cache_file)

            # If not in memory, check disk cache
            cache = load_cached_objects(cache_file)
            for key, val in cache.items():
                in_memory_cache[key] = val

            if cache_key in cache:
                # logger.info(f"Loading cached result from disk for {func.__name__}({args}, {kwargs})")
                logger.info(f"Loading cached result from disk for {func.__name__}")
                result = cache[cache_key]
                in_memory_cache[cache_key] = result  # Add to in-memory cache
                return result

            # If not in cache, run function and cache result
            # logger.info(f"Running function {func.__name__}({args}, {kwargs})")
            # logger.info(f"Running function {func.__name__}")
            result = func(*args, **kwargs)
            cache[cache_key] = result
            save_cached_objects(cache, cache_file)
            return result

        return wrapper
    return decorator


def load_cached_objects(cache_file):
    """
    Load cached objects from a local file.
    """
    try:
        with open(cache_file, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cached_objects(cache, cache_file):
    """
    Save cached objects to a local file.
    """
    with open(cache_file, 'w') as file:
        json.dump(cache, file)


def get_cache_file(cache_path, func_name):
    """
    Return the cache file path based on the provided cache path and function name.
    """
    if os.path.isdir(cache_path):
        cache_file = os.path.join(cache_path, f"{func_name}.json")
    else:
        cache_file = cache_path
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    return cache_file


def generate_cache_key(args, kwargs):
    """
    Generate a cache key from the function arguments and keyword arguments.
    """
    hashtable_types = (str, int, float, tuple, dict, set)  # exclude object as they will have different str repsentation on different runs
    hashable_args = tuple(hash_an_object(arg) for arg in args if isinstance(arg, hashtable_types))
    hashable_kwargs = tuple(sorted(list(
        (key, hash_an_object(value)) for key, value in kwargs.items() if is_hashable(key) and isinstance(value, hashtable_types)
    )))
    return repr((hashable_args, hashable_kwargs))


def is_hashable(obj):
    """
    Check if an object is hashable.
    """
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def hash_an_object(obj):
    """
    Convert a value to a hashable representation, including as many items as possible.
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (tuple, list)):
        return hash_list(obj)
    elif isinstance(obj, dict):
        return hashable_dict(obj)
    elif is_hashable(obj):
        return hash(obj)
    else:
        try:
            return repr(obj)
        except:  # noqa
            return ""


def hash_list(object):
    """
    Convert a list to a hashable representation, including as many items as possible.
    """
    return tuple(hash_an_object(item) for item in object)


def hashable_dict(obj):
    """
    Convert a dictionary to a hashable representation, including as many items as possible.
    """
    hashable_items = []
    unhashable_items = []
    for key, value in obj.items():
        hashable_key = hash_an_object(key)
        hashable_value = hash_an_object(value)
        if is_hashable(hashable_key) and is_hashable(hashable_value):
            hashable_items.append((hashable_key, hashable_value))
        elif is_hashable(hashable_key):
            hashable_items.append((hashable_key, value))
            unhashable_items.append((hashable_key, hashable_value))
        elif is_hashable(hashable_value):
            hashable_items.append((key, hashable_value))
            unhashable_items.append((hashable_key, hashable_value))
        else:
            unhashable_items.append((hashable_key, hashable_value))

    if unhashable_items:
        hashable_items.append(tuple(unhashable_items))

    return tuple(sorted(hashable_items))
