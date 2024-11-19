from datetime import datetime
import base64
import yaml
import requests
from dotenv import load_dotenv
import streamlit as st
import os
import aiohttp
import asyncio
import time
load_dotenv()

def load_config(file_path = "config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()

def convert_ns_to_seconds(ns_value):
    return ns_value / 1_000_000_000 

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")