import json
from pathlib import Path
import glob

wd = Path(__file__).parent.resolve()
def load_cookies_from_json(json_path):
    with open(json_path, 'r') as f:
        cookies = json.load(f)
    return cookies

# COOKIES_LIST = []
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "orcid.org.cookies.json")
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "www.researchgate.net.cookies.json")
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "github.com.cookies.json")
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "www.youtube.com.cookies.json")
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "www.ncbi.nlm.nih.gov.cookies.json")
# COOKIES_LIST += load_cookies_from_json(wd / "cookie_json" / "archive.org.cookies.json")

def convert_cookies_to_python():
    all_cookies = []
    # cookie_files = [
    #     "orcid.org.cookies.json",
    #     "www.researchgate.net.cookies.json",
    #     "github.com.cookies.json",
    #     "www.youtube.com.cookies.json",
    #     "www.ncbi.nlm.nih.gov.cookies.json",
    #     "archive.org.cookies.json", 
    #     "nature.com.cookies.json"
    # ]
    json_dir = wd / "cookie_json"
    cookie_files = glob.glob(str(json_dir / "*.json"))
    
    for cookie_file in cookie_files:
        json_path = wd / "cookie_json" / cookie_file
        cookies = load_cookies_from_json(json_path)
        all_cookies.extend(cookies)
    
    # 生成Python格式的cookies文件
    output_path = wd / "cookies_data.py"
    output_str = "COOKIES_LIST = [\n"
    for cookie in all_cookies:
        output_str += f"    {repr(cookie)},\n"
    output_str += "]\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    return output_str

def get_all_cookies():
    all_cookies = []
    json_dir = wd / "cookie_json"
    cookie_files = glob.glob(str(json_dir / "*.json"))
    
    for cookie_file in cookie_files:
        json_path = wd / "cookie_json" / cookie_file
        cookies = load_cookies_from_json(json_path)
        all_cookies.extend(cookies)
    return all_cookies

if __name__ == "__main__":
    print(convert_cookies_to_python())