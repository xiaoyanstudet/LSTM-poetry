import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_mpl_cache = Path.cwd() / "data" / "matplotlib_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import classmodel


TANG_OFFSETS = list(range(0, 58000, 1000))
RAW_FOLDER = urllib.parse.quote("全唐诗")
RAW_URLS = [
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
    "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
    "https://fastly.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
]
PROCESS_VERSION = 2
PUNCTUATION = {"，", "。", "？", "！", "、", "；", "："}

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

_FALLBACK_T2S = str.maketrans(
    {
        "兩": "两",
        "並": "并",
        "亂": "乱",
        "乾": "干",
        "亙": "亘",
        "亞": "亚",
        "佇": "伫",
        "來": "来",
        "侶": "侣",
        "俠": "侠",
        "倉": "仓",
        "個": "个",
        "們": "们",
        "傍": "傍",
        "傑": "杰",
        "傳": "传",
        "傷": "伤",
        "傾": "倾",
        "僧": "僧",
        "儀": "仪",
        "億": "亿",
        "儘": "尽",
        "兒": "儿",
        "內": "内",
        "兩": "两",
        "冊": "册",
        "寫": "写",
        "凈": "净",
        "凍": "冻",
        "劍": "剑",
        "劉": "刘",
        "勁": "劲",
        "勝": "胜",
        "勞": "劳",
        "勢": "势",
        "勸": "劝",
        "區": "区",
        "卻": "却",
        "參": "参",
        "雙": "双",
        "發": "发",
        "變": "变",
        "嘆": "叹",
        "嚴": "严",
        "國": "国",
        "園": "园",
        "圓": "圆",
        "圖": "图",
        "塵": "尘",
        "壘": "垒",
        "壞": "坏",
        "壯": "壮",
        "壽": "寿",
        "夢": "梦",
        "夾": "夹",
        "奪": "夺",
        "奮": "奋",
        "妝": "妆",
        "婦": "妇",
        "嬌": "娇",
        "學": "学",
        "宮": "宫",
        "寢": "寝",
        "實": "实",
        "寧": "宁",
        "寶": "宝",
        "將": "将",
        "對": "对",
        "尋": "寻",
        "導": "导",
        "爾": "尔",
        "塵": "尘",
        "屆": "届",
        "層": "层",
        "屬": "属",
        "岡": "冈",
        "峽": "峡",
        "島": "岛",
        "嶺": "岭",
        "巖": "岩",
        "幣": "币",
        "帶": "带",
        "幫": "帮",
        "幾": "几",
        "庫": "库",
        "廟": "庙",
        "廠": "厂",
        "廣": "广",
        "廬": "庐",
        "廳": "厅",
        "張": "张",
        "彌": "弥",
        "彎": "弯",
        "彙": "汇",
        "彞": "彝",
        "後": "后",
        "徑": "径",
        "從": "从",
        "復": "复",
        "徵": "征",
        "德": "德",
        "憂": "忧",
        "憶": "忆",
        "懷": "怀",
        "懸": "悬",
        "戀": "恋",
        "戰": "战",
        "戲": "戏",
        "戶": "户",
        "拂": "拂",
        "挾": "挟",
        "捲": "卷",
        "掃": "扫",
        "掩": "掩",
        "揚": "扬",
        "換": "换",
        "損": "损",
        "搖": "摇",
        "攝": "摄",
        "擺": "摆",
        "擾": "扰",
        "攜": "携",
        "敗": "败",
        "數": "数",
        "斂": "敛",
        "斷": "断",
        "於": "于",
        "時": "时",
        "晉": "晋",
        "晝": "昼",
        "暉": "晖",
        "暫": "暂",
        "曉": "晓",
        "曠": "旷",
        "會": "会",
        "朧": "胧",
        "東": "东",
        "條": "条",
        "來": "来",
        "楊": "杨",
        "極": "极",
        "樓": "楼",
        "標": "标",
        "樞": "枢",
        "樹": "树",
        "橋": "桥",
        "機": "机",
        "橫": "横",
        "檻": "槛",
        "櫓": "橹",
        "欄": "栏",
        "權": "权",
        "歡": "欢",
        "歲": "岁",
        "歷": "历",
        "歸": "归",
        "殘": "残",
        "殿": "殿",
        "毀": "毁",
        "氣": "气",
        "漢": "汉",
        "湯": "汤",
        "溝": "沟",
        "滄": "沧",
        "滅": "灭",
        "滿": "满",
        "漁": "渔",
        "漢": "汉",
        "漸": "渐",
        "潤": "润",
        "濁": "浊",
        "濃": "浓",
        "濕": "湿",
        "濟": "济",
        "燈": "灯",
        "點": "点",
        "為": "为",
        "爲": "为",
        "烏": "乌",
        "無": "无",
        "煙": "烟",
        "煩": "烦",
        "熒": "荧",
        "燭": "烛",
        "營": "营",
        "爭": "争",
        "愛": "爱",
        "獨": "独",
        "獻": "献",
        "獲": "获",
        "獸": "兽",
        "現": "现",
        "瑤": "瑶",
        "畫": "画",
        "畢": "毕",
        "異": "异",
        "當": "当",
        "疎": "疏",
        "盡": "尽",
        "監": "监",
        "盤": "盘",
        "盧": "卢",
        "眾": "众",
        "睜": "睁",
        "矚": "瞩",
        "礎": "础",
        "禪": "禅",
        "禮": "礼",
        "離": "离",
        "穀": "谷",
        "積": "积",
        "穩": "稳",
        "窮": "穷",
        "竄": "窜",
        "竅": "窍",
        "筆": "笔",
        "築": "筑",
        "簡": "简",
        "簾": "帘",
        "籠": "笼",
        "粧": "妆",
        "糧": "粮",
        "糾": "纠",
        "紀": "纪",
        "約": "约",
        "紅": "红",
        "紗": "纱",
        "納": "纳",
        "紛": "纷",
        "素": "素",
        "絃": "弦",
        "絕": "绝",
        "絲": "丝",
        "綃": "绡",
        "綠": "绿",
        "維": "维",
        "綱": "纲",
        "網": "网",
        "綵": "彩",
        "緋": "绯",
        "緒": "绪",
        "緣": "缘",
        "編": "编",
        "縣": "县",
        "縱": "纵",
        "總": "总",
        "繞": "绕",
        "繡": "绣",
        "繫": "系",
        "續": "续",
        "纖": "纤",
        "缺": "缺",
        "罷": "罢",
        "羅": "罗",
        "羈": "羁",
        "羨": "羡",
        "義": "义",
        "習": "习",
        "翹": "翘",
        "聖": "圣",
        "聞": "闻",
        "聯": "联",
        "聽": "听",
        "職": "职",
        "肅": "肃",
        "脈": "脉",
        "脫": "脱",
        "臨": "临",
        "與": "与",
        "興": "兴",
        "舉": "举",
        "舊": "旧",
        "艙": "舱",
        "艱": "艰",
        "艷": "艳",
        "藝": "艺",
        "蘭": "兰",
        "蘇": "苏",
        "葉": "叶",
        "蕭": "萧",
        "薔": "蔷",
        "薩": "萨",
        "藍": "蓝",
        "處": "处",
        "虛": "虚",
        "號": "号",
        "蟲": "虫",
        "蠟": "蜡",
        "蠻": "蛮",
        "衆": "众",
        "補": "补",
        "裝": "装",
        "裡": "里",
        "裏": "里",
        "製": "制",
        "複": "复",
        "親": "亲",
        "覺": "觉",
        "覽": "览",
        "觀": "观",
        "觴": "觞",
        "觸": "触",
        "計": "计",
        "訓": "训",
        "記": "记",
        "詠": "咏",
        "詩": "诗",
        "詔": "诏",
        "試": "试",
        "話": "话",
        "誇": "夸",
        "誌": "志",
        "語": "语",
        "誤": "误",
        "說": "说",
        "誰": "谁",
        "課": "课",
        "調": "调",
        "諸": "诸",
        "諾": "诺",
        "謀": "谋",
        "謂": "谓",
        "謠": "谣",
        "謝": "谢",
        "謫": "谪",
        "證": "证",
        "識": "识",
        "譜": "谱",
        "讀": "读",
        "變": "变",
        "讚": "赞",
        "豈": "岂",
        "貝": "贝",
        "貞": "贞",
        "負": "负",
        "財": "财",
        "貢": "贡",
        "貧": "贫",
        "貴": "贵",
        "買": "买",
        "費": "费",
        "賀": "贺",
        "賓": "宾",
        "賜": "赐",
        "賞": "赏",
        "賢": "贤",
        "質": "质",
        "賴": "赖",
        "贈": "赠",
        "趙": "赵",
        "跡": "迹",
        "踐": "践",
        "蹤": "踪",
        "躍": "跃",
        "軀": "躯",
        "車": "车",
        "軒": "轩",
        "軟": "软",
        "輕": "轻",
        "輅": "辂",
        "輔": "辅",
        "輩": "辈",
        "輪": "轮",
        "輯": "辑",
        "輸": "输",
        "轉": "转",
        "轍": "辙",
        "轡": "辔",
        "辭": "辞",
        "邊": "边",
        "遙": "遥",
        "遞": "递",
        "遠": "远",
        "適": "适",
        "遲": "迟",
        "遷": "迁",
        "選": "选",
        "遺": "遗",
        "邁": "迈",
        "還": "还",
        "鄉": "乡",
        "鄭": "郑",
        "醫": "医",
        "醜": "丑",
        "釀": "酿",
        "釋": "释",
        "針": "针",
        "鈴": "铃",
        "銀": "银",
        "銜": "衔",
        "銷": "销",
        "鋒": "锋",
        "鋪": "铺",
        "錦": "锦",
        "錄": "录",
        "鍾": "钟",
        "鎖": "锁",
        "鎮": "镇",
        "鏡": "镜",
        "鐵": "铁",
        "鑒": "鉴",
        "長": "长",
        "門": "门",
        "閃": "闪",
        "閉": "闭",
        "開": "开",
        "閑": "闲",
        "間": "间",
        "閣": "阁",
        "閱": "阅",
        "闕": "阙",
        "關": "关",
        "闡": "阐",
        "闢": "辟",
        "陰": "阴",
        "陣": "阵",
        "陳": "陈",
        "陸": "陆",
        "陽": "阳",
        "隄": "堤",
        "險": "险",
        "隱": "隐",
        "雜": "杂",
        "雙": "双",
        "雲": "云",
        "電": "电",
        "霧": "雾",
        "靈": "灵",
        "靜": "静",
        "靦": "腼",
        "韋": "韦",
        "韻": "韵",
        "響": "响",
        "頁": "页",
        "頂": "顶",
        "頃": "顷",
        "項": "项",
        "順": "顺",
        "須": "须",
        "頌": "颂",
        "預": "预",
        "領": "领",
        "頗": "颇",
        "頭": "头",
        "頷": "颔",
        "頻": "频",
        "題": "题",
        "額": "额",
        "顏": "颜",
        "願": "愿",
        "顧": "顾",
        "風": "风",
        "飛": "飞",
        "飄": "飘",
        "餘": "余",
        "館": "馆",
        "饒": "饶",
        "馬": "马",
        "駐": "驻",
        "駕": "驾",
        "騎": "骑",
        "驚": "惊",
        "驛": "驿",
        "驟": "骤",
        "驢": "驴",
        "驥": "骥",
        "髮": "发",
        "鬢": "鬓",
        "鬥": "斗",
        "鬧": "闹",
        "魯": "鲁",
        "鮮": "鲜",
        "鯨": "鲸",
        "魚": "鱼",
        "鳥": "鸟",
        "鳴": "鸣",
        "鳳": "凤",
        "鴉": "鸦",
        "鶯": "莺",
        "鶴": "鹤",
        "鷗": "鸥",
        "鷲": "鹫",
        "鹿": "鹿",
        "麗": "丽",
        "麥": "麦",
        "黃": "黄",
        "齊": "齐",
        "齋": "斋",
        "齒": "齿",
        "龍": "龙",
    }
)


@dataclass
class TrainConfig:
    raw_dir: str = "tangshi"
    data_dir: str = "data"
    poem_type: int = 7
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    embedding_dim: int = 128
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    eval_batches: int = 30
    sample_count: int = 8
    checkpoint_every_batches: int = 200
    log_interval: int = 60
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3
    simplify_text: bool = True
    constrain_format: bool = True
    max_files: int = 0
    download_retries: int = 3
    strict_download: bool = False
    force_download: bool = False
    force_process: bool = False
    resume: bool = True
    device: str = "auto"
    seed: int = 42
    start_words: str = "湖光秋月两相和"
    temperature: float = 0.9
    top_k: int = 8


def choose_device(device_name="auto"):
    if device_name != "auto":
        return torch.device(device_name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _valid_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list)
    except (OSError, json.JSONDecodeError):
        return False


def _download_file(urls, destination, retries=3):
    if isinstance(urls, str):
        urls = [urls]

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    last_error = None

    for url in urls:
        for attempt in range(1, retries + 1):
            if temp_path.exists():
                temp_path.unlink()
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) LSTM-poetry-downloader",
                        "Accept": "application/json,text/plain,*/*",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    downloaded = 0
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = response.read(1024 * 256)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                print(
                                    f"  {destination.name}: {downloaded / 1024 / 1024:.1f}MB "
                                    f"/ {total / 1024 / 1024:.1f}MB ({pct:.0f}%)",
                                    end="\r",
                                )
                temp_path.replace(destination)
                if not _valid_json(destination):
                    destination.unlink(missing_ok=True)
                    raise RuntimeError("downloaded file is not valid JSON")
                print(f"  {destination.name}: done{' ' * 30}")
                return
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if temp_path.exists():
                    temp_path.unlink()
                if attempt < retries:
                    wait_seconds = min(20, 2 ** (attempt - 1)) + random.random()
                    print(
                        f"  {destination.name}: retry {attempt}/{retries} failed "
                        f"({exc}); wait {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                else:
                    print(f"  {destination.name}: mirror failed after {retries} attempts: {url}")

    raise RuntimeError(f"download failed: {destination.name}\nlast error: {last_error}")


def ensure_dataset(raw_dir, max_files=0, force_download=False, download_retries=3, strict_download=False):
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    custom_jsons = sorted(raw_path.glob("*.json"))
    expected_names = {f"poet.tang.{offset}.json" for offset in TANG_OFFSETS}
    has_custom_dataset = custom_jsons and not any(p.name in expected_names for p in custom_jsons)
    if has_custom_dataset and not force_download:
        print(f"Found {len(custom_jsons)} local JSON files in {raw_path}; skip auto-download.")
        return custom_jsons

    offsets = TANG_OFFSETS[:max_files] if max_files and max_files > 0 else TANG_OFFSETS
    print(f"Checking Tang poem JSON dataset in {raw_path} ...")
    failed = []
    for offset in offsets:
        destination = raw_path / f"poet.tang.{offset}.json"
        if destination.exists() and not force_download and _valid_json(destination):
            continue
        print(f"Downloading {destination.name}")
        urls = [template.format(offset=offset) for template in RAW_URLS]
        try:
            _download_file(urls, destination, retries=download_retries)
        except RuntimeError as exc:
            failed.append(destination.name)
            if strict_download:
                raise
            print(f"Warning: skip {destination.name} for now: {exc}")

    jsons = [path for path in sorted(raw_path.glob("*.json")) if _valid_json(path)]
    if not jsons:
        raise FileNotFoundError(f"No JSON files found in {raw_path}")
    if failed:
        print(
            f"Downloaded/available JSON files: {len(jsons)}. "
            f"Missing {len(failed)} files; training will use the available subset."
        )
    return jsons


def to_simplified(text):
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(_FALLBACK_T2S)


def _normalize_text(text, simplify=True):
    text = (
        text.strip()
        .replace(",", "，")
        .replace(".", "。")
        .replace("?", "？")
        .replace("!", "！")
        .replace(" ", "")
    )
    return to_simplified(text) if simplify else text


def _normalize_poem(paragraphs, simplify=True):
    poem = "".join(paragraphs)
    return _normalize_text(poem, simplify=simplify)


def _process_signature(raw_dir, max_files, simplify):
    jsons = sorted(Path(raw_dir).glob("*.json"))
    if max_files and max_files > 0:
        jsons = jsons[:max_files]
    return {
        "version": PROCESS_VERSION,
        "simplify": simplify,
        "simplifier": "opencc" if simplify and _OPENCC is not None else "fallback" if simplify else "none",
        "max_files": max_files,
        "json_count": len(jsons),
        "json_total_bytes": sum(path.stat().st_size for path in jsons),
    }


def _load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _legacy_normalize_poem(paragraphs):
    poem = "".join(paragraphs).strip()
    return (
        poem.replace(",", "，")
        .replace(".", "。")
        .replace("?", "？")
        .replace("!", "！")
        .replace(" ", "")
    )


def _poem_kind(poem):
    if len(poem) == 24 and poem[5] == "，" and poem[11] == "。" and poem[17] == "，" and poem[23] == "。":
        return 5
    if len(poem) == 32 and poem[7] == "，" and poem[15] == "。" and poem[23] == "，" and poem[31] == "。":
        return 7
    return 0


def process_1(path, path2, force=False, max_files=0, simplify=True):
    """Extract five-character and seven-character quatrains from JSON files."""

    output_dir = Path(path2)
    output_dir.mkdir(parents=True, exist_ok=True)
    poem_5_path = output_dir / "poem_5.txt"
    poem_7_path = output_dir / "poem_7.txt"
    meta_path = output_dir / "process_meta.json"
    expected_meta = _process_signature(path, max_files, simplify)

    if not force and poem_5_path.exists() and poem_7_path.exists():
        count_5 = _count_lines(poem_5_path)
        count_7 = _count_lines(poem_7_path)
        cached_meta = _load_json(meta_path, default={})
        if (count_5 > 0 or count_7 > 0) and cached_meta == expected_meta:
            print(f"Using cached processed poems: five={count_5}, seven={count_7}")
            return {"poem_5": count_5, "poem_7": count_7}
        print("Processed poem cache is outdated; rebuilding it.")

    poem_5 = []
    poem_7 = []
    jsons = sorted(Path(path).glob("*.json"))
    if max_files and max_files > 0:
        jsons = jsons[:max_files]
    if not jsons:
        raise FileNotFoundError(f"No JSON files found in {path}")

    print(f"Processing {len(jsons)} JSON files ...")
    for js in jsons:
        with open(js, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skip invalid JSON: {js}")
                continue

        for item in data:
            paragraphs = item.get("paragraphs", [])
            if not paragraphs:
                continue
            poem = _normalize_poem(paragraphs, simplify=simplify)
            kind = _poem_kind(poem)
            tokenized = " ".join(list(poem))
            if kind == 5:
                poem_5.append(tokenized)
            elif kind == 7:
                poem_7.append(tokenized)

    _write_lines(poem_5_path, poem_5)
    _write_lines(poem_7_path, poem_7)
    _save_json(meta_path, expected_meta)
    print(f"Processed poems saved: five={len(poem_5)}, seven={len(poem_7)}")
    return {"poem_5": len(poem_5), "poem_7": len(poem_7)}


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_poems(path):
    poems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            poems.append(line.split() if " " in line else list(line))
    return poems


def build_vocab(poems):
    counts = {}
    for poem in poems:
        for char in poem:
            counts[char] = counts.get(char, 0) + 1
    index_to_word = sorted(counts.keys(), key=lambda ch: (-counts[ch], ch))
    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
    return word_to_index, index_to_word


def encode_poems(poems, word_to_index):
    return [[word_to_index[char] for char in poem] for poem in poems]


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path,
    model,
    optimizer,
    next_epoch,
    global_step,
    config,
    word_to_index,
    index_to_word,
    history,
):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "config": asdict(config),
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "history": history,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _normalize_history(history=None):
    normalized = _empty_history()
    if history:
        normalized.update(history)
    for key in ("train_loss", "val_loss", "perplexity", "format_accuracy", "distinct_1", "distinct_2"):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    normalized.setdefault("best_val_loss", None)
    normalized.setdefault("best_epoch", 0)
    normalized.setdefault("epochs_without_improvement", 0)
    return normalized


def load_checkpoint(path, model, optimizer, device, word_to_index, index_to_word):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("word_to_index") != word_to_index or checkpoint.get("index_to_word") != index_to_word:
        print("Checkpoint vocab differs from current data; start a fresh run.")
        return 0, 0, _empty_history()

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    history = _normalize_history(checkpoint.get("history"))
    next_epoch = int(checkpoint.get("next_epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed from {path}: next_epoch={next_epoch + 1}, global_step={global_step}")
    return next_epoch, global_step, history


def evaluate(model, loader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            flat_labels = labels.reshape(-1)
            loss = model.loss(outputs, flat_labels)
            total_loss += float(loss.detach().cpu()) * flat_labels.numel()
            total_tokens += flat_labels.numel()
    if total_tokens == 0:
        return {"val_loss": 0.0, "perplexity": 0.0}
    val_loss = total_loss / total_tokens
    return {"val_loss": val_loss, "perplexity": math.exp(min(val_loss, 20.0))}


def _sample_next_id(logits, temperature=0.9, top_k=8, banned_ids=None):
    logits = logits.detach().to("cpu").clone()
    if banned_ids:
        logits[list(banned_ids)] = -torch.inf
    if not torch.isfinite(logits).any():
        logits = torch.zeros_like(logits)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
        probs = torch.softmax(values, dim=-1)
        pick = torch.multinomial(probs, 1)
        return int(indices[pick].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _format_marks(poem_type):
    if poem_type == 5:
        return {5: "，", 11: "。", 17: "，", 23: "。"}
    return {7: "，", 15: "。", 23: "，", 31: "。"}


def _append_generated_char(result, char, word_to_index, model, hidden, device):
    result.append(char)
    if char not in word_to_index:
        return hidden, None
    input_id = torch.tensor([[word_to_index[char]]], dtype=torch.long, device=device)
    _, hidden = model(input_id, hidden)
    return hidden, input_id


def generate_poetry(
    model,
    index_to_word,
    word_to_index,
    poem_type=7,
    start_words="",
    device=None,
    temperature=0.9,
    top_k=8,
    simplify_text=True,
    constrain_format=True,
):
    model.eval()
    device = device or next(model.parameters()).device
    target_len = 24 if poem_type == 5 else 32
    marks = _format_marks(poem_type)
    punctuation_ids = {word_to_index[ch] for ch in PUNCTUATION if ch in word_to_index}
    candidate_ids = [i for i, word in enumerate(index_to_word) if word not in PUNCTUATION]
    if not candidate_ids:
        candidate_ids = list(range(len(index_to_word)))

    result = []
    hidden = None
    input_id = None
    start_chars = [
        ch
        for ch in _normalize_text(start_words, simplify=simplify_text)
        if ch not in PUNCTUATION and ch.strip()
    ]

    with torch.no_grad():
        for char in start_chars:
            while constrain_format and len(result) in marks and len(result) < target_len:
                hidden, input_id = _append_generated_char(
                    result, marks[len(result)], word_to_index, model, hidden, device
                )
            if len(result) >= target_len:
                return "".join(result[:target_len])
            if char not in word_to_index:
                continue
            hidden, input_id = _append_generated_char(result, char, word_to_index, model, hidden, device)
            if len(result) >= target_len:
                return "".join(result[:target_len])

        if input_id is None:
            first_id = random.choice(candidate_ids)
            hidden, input_id = _append_generated_char(
                result, index_to_word[first_id], word_to_index, model, hidden, device
            )

        while len(result) < target_len:
            if constrain_format and len(result) in marks:
                hidden, input_id = _append_generated_char(
                    result, marks[len(result)], word_to_index, model, hidden, device
                )
                continue
            output, hidden = model(input_id, hidden)
            banned_ids = punctuation_ids if constrain_format else None
            next_id = _sample_next_id(output[-1], temperature=temperature, top_k=top_k, banned_ids=banned_ids)
            hidden, input_id = _append_generated_char(
                result, index_to_word[next_id], word_to_index, model, hidden, device
            )

    return "".join(result[:target_len])


def format_accuracy(samples, poem_type):
    if not samples:
        return 0.0
    target_len = 24 if poem_type == 5 else 32
    marks = _format_marks(poem_type)
    correct = 0
    for poem in samples:
        if len(poem) != target_len:
            continue
        if all(poem[pos] == mark for pos, mark in marks.items()):
            correct += 1
    return correct / len(samples)


def distinct_n(samples, n):
    total = 0
    uniq = set()
    for sample in samples:
        chars = [ch for ch in sample if ch.strip()]
        if len(chars) < n:
            continue
        for i in range(len(chars) - n + 1):
            gram = tuple(chars[i : i + n])
            uniq.add(gram)
            total += 1
    return len(uniq) / total if total else 0.0


def generation_metrics(model, index_to_word, word_to_index, config, device):
    samples = [
        generate_poetry(
            model,
            index_to_word,
            word_to_index,
            poem_type=config.poem_type,
            start_words=config.start_words if i == 0 else "",
            device=device,
            temperature=config.temperature,
            top_k=config.top_k,
            simplify_text=config.simplify_text,
            constrain_format=config.constrain_format,
        )
        for i in range(config.sample_count)
    ]
    return {
        "samples": samples,
        "format_accuracy": format_accuracy(samples, config.poem_type),
        "distinct_1": distinct_n(samples, 1),
        "distinct_2": distinct_n(samples, 2),
    }


def save_samples(path, epoch, metrics):
    lines = [f"epoch={epoch}", f"format_accuracy={metrics['format_accuracy']:.4f}", f"distinct_1={metrics['distinct_1']:.4f}", f"distinct_2={metrics['distinct_2']:.4f}", ""]
    lines.extend(metrics["samples"])
    _write_lines(path, lines)


def plot_metrics(history, figure_path, poem_type):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Poetry LSTM training metrics ({poem_type}-char quatrain)")

    axes[0, 0].plot(epochs, history.get("train_loss", []), marker="o", label="train loss")
    axes[0, 0].plot(epochs, history.get("val_loss", []), marker="o", label="val loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, history.get("perplexity", []), marker="o", color="#d95f02")
    axes[0, 1].set_title("Validation perplexity")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.25)

    labels = ["format", "distinct-1", "distinct-2"]
    values = [
        history.get("format_accuracy", [0.0])[-1] if history.get("format_accuracy") else 0.0,
        history.get("distinct_1", [0.0])[-1] if history.get("distinct_1") else 0.0,
        history.get("distinct_2", [0.0])[-1] if history.get("distinct_2") else 0.0,
    ]
    axes[1, 0].bar(labels, values, color=["#1b9e77", "#7570b3", "#e7298a"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Generation quality")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    summary = "\n".join(
        [
            f"last train loss: {history.get('train_loss', [0])[-1]:.4f}" if history.get("train_loss") else "last train loss: n/a",
            f"last val loss: {history.get('val_loss', [0])[-1]:.4f}" if history.get("val_loss") else "last val loss: n/a",
            f"last perplexity: {history.get('perplexity', [0])[-1]:.2f}" if history.get("perplexity") else "last perplexity: n/a",
            f"format accuracy: {values[0]:.2%}",
            f"distinct-1: {values[1]:.2%}",
            f"distinct-2: {values[2]:.2%}",
            f"best epoch: {history.get('best_epoch') or 'n/a'}",
        ]
    )
    axes[1, 1].text(0.02, 0.95, summary, va="top", fontsize=12)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _empty_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "format_accuracy": [],
        "distinct_1": [],
        "distinct_2": [],
        "best_val_loss": None,
        "best_epoch": 0,
        "epochs_without_improvement": 0,
    }


def run_training(config):
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    checkpoint_dir = data_dir / "checkpoints"
    figure_dir = data_dir / "figures"
    sample_dir = data_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(
        config.raw_dir,
        max_files=config.max_files,
        force_download=config.force_download,
        download_retries=config.download_retries,
        strict_download=config.strict_download,
    )
    process_1(
        config.raw_dir,
        config.data_dir,
        force=config.force_process,
        max_files=config.max_files,
        simplify=config.simplify_text,
    )

    poem_path = data_dir / f"poem_{config.poem_type}.txt"
    poems = read_poems(poem_path)
    if not poems:
        raise RuntimeError(f"No {config.poem_type}-character poems found in {poem_path}")

    word_to_index, index_to_word = build_vocab(poems)
    encoded = encode_poems(poems, word_to_index)
    dataset = classmodel.MyDataset(encoded)

    val_size = max(1, int(len(dataset) * config.val_ratio)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size
    if val_size:
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    device = choose_device(config.device)
    print(f"Device: {device}")
    print(f"Poems: train={train_size}, val={val_size}, vocab={len(index_to_word)}")

    model = classmodel.PoemLstm(
        vocab_size=len(index_to_word),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    latest_checkpoint = checkpoint_dir / f"poem_{config.poem_type}_latest.pt"
    best_checkpoint = checkpoint_dir / f"poem_{config.poem_type}_best.pt"
    start_epoch = 0
    global_step = 0
    history = _empty_history()
    if config.resume and latest_checkpoint.exists():
        start_epoch, global_step, history = load_checkpoint(
            latest_checkpoint, model, optimizer, device, word_to_index, index_to_word
        )
    history = _normalize_history(history)
    best_val_loss = history.get("best_val_loss")
    epochs_without_improvement = int(history.get("epochs_without_improvement") or 0)

    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            total_loss = 0.0
            total_tokens = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs, _ = model(inputs)
                flat_labels = labels.reshape(-1)
                loss = model.loss(outputs, flat_labels)
                loss.backward()
                optimizer.step()

                global_step += 1
                total_loss += float(loss.detach().cpu()) * flat_labels.numel()
                total_tokens += flat_labels.numel()

                if config.log_interval and batch_idx % config.log_interval == 0:
                    print(
                        f"epoch {epoch + 1}/{config.epochs} "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss={float(loss.detach().cpu()):.4f}"
                    )

                if config.checkpoint_every_batches and global_step % config.checkpoint_every_batches == 0:
                    save_checkpoint(
                        latest_checkpoint,
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        config,
                        word_to_index,
                        index_to_word,
                        history,
                    )

            train_loss = total_loss / max(1, total_tokens)
            val_stats = evaluate(model, val_loader, device, config.eval_batches) if val_loader else {"val_loss": 0.0, "perplexity": 0.0}
            gen_stats = generation_metrics(model, index_to_word, word_to_index, config, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_stats["val_loss"])
            history["perplexity"].append(val_stats["perplexity"])
            history["format_accuracy"].append(gen_stats["format_accuracy"])
            history["distinct_1"].append(gen_stats["distinct_1"])
            history["distinct_2"].append(gen_stats["distinct_2"])

            monitor_loss = val_stats["val_loss"] if val_loader else train_loss
            improved = best_val_loss is None or monitor_loss < best_val_loss - config.early_stopping_min_delta
            if improved:
                best_val_loss = monitor_loss
                epochs_without_improvement = 0
                history["best_val_loss"] = best_val_loss
                history["best_epoch"] = epoch + 1
                history["epochs_without_improvement"] = epochs_without_improvement
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    config,
                    word_to_index,
                    index_to_word,
                    history,
                )
            else:
                epochs_without_improvement += 1
                history["best_val_loss"] = best_val_loss
                history["epochs_without_improvement"] = epochs_without_improvement

            save_samples(sample_dir / f"poem_{config.poem_type}_epoch_{epoch + 1:03d}.txt", epoch + 1, gen_stats)
            save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                word_to_index,
                index_to_word,
                history,
            )
            save_checkpoint(
                checkpoint_dir / f"poem_{config.poem_type}_epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                word_to_index,
                index_to_word,
                history,
            )
            figure_path = plot_metrics(history, figure_dir / f"training_metrics_{config.poem_type}.png", config.poem_type)
            print(
                f"epoch {epoch + 1} done: train_loss={train_loss:.4f}, "
                f"val_loss={val_stats['val_loss']:.4f}, ppl={val_stats['perplexity']:.2f}, "
                f"format={gen_stats['format_accuracy']:.2%}, "
                f"best_epoch={history.get('best_epoch')}, figure={figure_path}"
            )
            print("sample:", gen_stats["samples"][0])

            if (
                config.early_stopping_patience
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                print(
                    f"Early stopping: validation loss did not improve for "
                    f"{epochs_without_improvement} epochs. Best epoch: {history.get('best_epoch')}"
                )
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving latest checkpoint before exit ...")
        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            epoch,
            global_step,
            config,
            word_to_index,
            index_to_word,
            history,
        )
        raise

    print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Metrics figure: {figure_dir / f'training_metrics_{config.poem_type}.png'}")
    return model


def train(path1="tangshi", path2="data", peo=7, batchsize=64):
    config = TrainConfig(raw_dir=path1, data_dir=path2, poem_type=peo, batch_size=batchsize)
    return run_training(config)


def gen_poetry(wordsize, index_2_word, type1, wvec, model):
    raise RuntimeError("gen_poetry now uses generate_poetry(model, index_to_word, word_to_index, ...).")
