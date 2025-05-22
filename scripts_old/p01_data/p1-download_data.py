import os
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import csv
import re
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import urllib3

def download_cnpj_files(destination='download_cnpj'):
    today = datetime.today()
    first_day_of_current_month = today.replace(day=1)
    last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
    year_month = last_day_of_previous_month.strftime("%Y-%m")
    base_url = f"https://arquivos.receitafederal.gov.br/dados/cnpj/dados_abertos_cnpj/{year_month}/"

    print(f"\n🔍 Accessing CNPJ URL: {base_url}")
    os.makedirs(destination, exist_ok=True)

    csv_path = os.path.join(destination, "unique_downloads.csv")
    entries = []

    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error accessing {base_url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.zip')]

    print(f"🔗 Found {len(links)} CNPJ files.")

    for link in links:
        file_url = urljoin(base_url, link)
        filename = os.path.basename(link)
        file_path = os.path.join(destination, filename)
        entries.append((filename, file_path, file_url))

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'file_path', 'file_url'])
        writer.writerows(entries)
    print(f"📁 File list saved to: {csv_path}")

    for filename, file_path, file_url in entries:
        if os.path.exists(file_path):
            print(f"✅ {filename} already exists. Skipping.")
            continue

        print(f"⬇️ Downloading: {filename}")
        try:
            with requests.get(file_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                    total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=filename
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            continue

def download_pgfn_files(destination='download_pgfn'):
    base_url = "https://www.gov.br/pgfn/pt-br/assuntos/divida-ativa-da-uniao/transparencia-fiscal-1/dados-abertos"
    print(f"\n🔍 Accessing PGFN URL: {base_url}")
    os.makedirs(destination, exist_ok=True)

    csv_path = os.path.join(destination, "unique_downloads.csv")
    entries = []

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        response = requests.get(base_url, verify=False)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error accessing {base_url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.find_all('a', href=True)
             if a['href'].lower().endswith((".zip", ".csv", ".xls", ".xlsx", ".json"))]

    print(f"🔗 Found {len(links)} PGFN files.")

    def extract_year_quarter(url):
        path_parts = urlparse(url).path.strip("/").split("/")
        if len(path_parts) < 2:
            return None
        parent_folder = path_parts[-2]
        match = re.search(r"(\d{4})_trimestre_(\d{2})", parent_folder)
        if match:
            year, quarter = match.groups()
            return f"{year}T{int(quarter)}"
        return None

    for link in links:
        file_url = urljoin(base_url, link)
        original_name = os.path.basename(file_url)
        prefix = extract_year_quarter(file_url)
        filename = f"{prefix}_{original_name}" if prefix else original_name
        file_path = os.path.join(destination, filename)
        entries.append((filename, file_path, file_url))

    with open(csv_path, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'file_path', 'file_url'])
        writer.writerows(entries)

    print(f"📁 File list saved to: {csv_path}")
    print(f"📥 Starting download of {len(entries)} PGFN files...")

    for filename, file_path, file_url in entries:
        if os.path.exists(file_path):
            print(f"✅ {filename} already exists. Skipping.")
            continue

        print(f"⬇️ Downloading: {filename}")
        try:
            with requests.get(file_url, stream=True, verify=False, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                    total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=filename
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            continue

if __name__ == "__main__":
    download_cnpj_files(destination="/home/nicolas/Documents/dados/cnpj/raw/")
    download_pgfn_files(destination="/home/nicolas/Documents/dados/pgfn/raw/")
    print("\n✅ All downloads completed.")
