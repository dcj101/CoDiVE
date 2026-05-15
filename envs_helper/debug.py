import os
import traceback

for k, v in sorted(os.environ.items()):
    kl = k.lower()
    if 'proxy' in kl or 'hf_' in kl or 'huggingface' in kl:
        print(f"{k}={v!r}")

print("--- httpx.Client() ---")
import httpx
try:
    c = httpx.Client()
    print("Client OK")
except Exception:
    traceback.print_exc()

print("--- default_client_factory source ---")
from huggingface_hub.utils import _http
import inspect
print(inspect.getsource(_http.default_client_factory))