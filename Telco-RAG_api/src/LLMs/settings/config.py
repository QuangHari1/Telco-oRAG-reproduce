import pathlib
import os
from os.path import abspath, dirname, join
from dynaconf import Dynaconf
from dotenv import load_dotenv

current_dir = dirname(abspath(__file__))
project_dir = pathlib.Path(current_dir).parents[2]
load_dotenv(project_dir / ".env", override=False)
# setting_dir = join(current_dir, "settings")
setting_dir = current_dir

toml_files = list(pathlib.Path(join(setting_dir)).glob('*.toml'))


default_settings_dict = {
"openai_api_key" : os.environ.get("OPENAI_API_KEY", ""),
"any_api_key" : "",
"mistral_api" : "",
"anthropic_api" : "",
"cohere_api" : "",
"google_search_api" : "",
"pplx_api" : "",
"together_api" : "",
"rate_limit" : 9,
"fireworks_api" : ""
}

global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    settings_files=toml_files,
    **default_settings_dict,
)

def get_settings():
    return global_settings
Đang cài dependency vẽ biểu đồ để test report thật trên 200 câu đã match. Việc này chỉ tải package Python, không gọi model/API benchmark.