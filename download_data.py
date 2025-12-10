import pandas as pd
import yaml

CONFIG_PATH = "config/settings.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def download_league(url, name):
    df = pd.read_csv(url)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{name}.csv", index=False)

def main():
    cfg = load_config()
    for league, info in cfg["leagues"].items():
        download_league(info["url"], league)

if __name__ == "__main__":
    main()
