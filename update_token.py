# update_token.py
from dotenv import load_dotenv, set_key, find_dotenv
import getpass

load_dotenv()
env_path = find_dotenv()
if not env_path:
    raise SystemExit(".env not found")

token = getpass.getpass("Paste IMS bearer token (hidden): ").strip()
if not token:
    print("No token provided; aborting.")
else:
    set_key(env_path, "IMS_TOKEN", token)
    print(" .env updated with new IMS_TOKEN")
