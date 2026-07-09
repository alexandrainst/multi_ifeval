#!/usr/bin/env python3
"""Fix and upload a MultiIFEval language subset to HF Hub.

Usage: uv run src/scripts/fix_hf_subset.py <lang_code>

Example: uv run src/scripts/fix_hf_subset.py nso
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import upload_file

# Mapping of translated constraint values back to English (200+ entries)
RELATION_MAPPINGS = {
    # Bulgarian
    "по-малко от": "less than",
    "поне": "at least",
    "минимум": "at least",
    # German
    "weniger als": "less than",
    "mindestens": "at least",
    "genau": "exactly",
    # English variants
    "equal to": "exactly",
    "exactly": "exactly",
    "equal": "exactly",
    # French
    "moins de": "less than",
    "au moins": "at least",
    "exactement": "exactly",
    # Northern Sotho
    "ka tlase ga": "less than",
    "ka fase ga": "less than",
    "bonnyane mantšu a": "at least",
    # Gothic (all variants)
    "𐌼𐌹𐌽𐌽𐌿𐌶𐌰": "less than",
    "𐌼𐌹𐌽𐌽𐌹𐌶𐌰": "less than",
    "𐌼𐌹𐌽𐌽𐌹𐌶𐌰 𐌸𐌰𐌿": "less than",
    "𐌼𐌹𐌽𐌽𐌹𐌶𐍉 𐌸𐌰𐌿": "less than",
    "𐌼𐌹𐌽𐍃 𐌸𐌰𐌿": "less than",
    "𐌼𐌹𐌽": "less than",
    "𐍆𐌰𐌿𐍂": "less than",
    "𐍆𐌰𐌿𐍂 𐌼𐌹𐌽𐌽𐌹𐌶𐌰": "less than",
    "𐍆𐌰𐌿𐍂 𐌰𐍆𐌰𐍂": "less than",
    "𐍆𐌰𐌿𐍂 𐌹𐌽𐌽𐌰": "less than",
    "𐍆𐌰𐌿𐍂 𐌼𐌹𐌽 𐌸𐌰𐌽": "less than",
    "𐍆𐌰𐍅𐌹𐌶𐌰 𐌸𐌰𐌿": "less than",
    "𐍆𐌰𐍅𐌹𐌶𐌹𐍃 𐌸𐌰𐌿": "less than",
    "𐌰𐍄 𐌼𐌹𐌽𐌽𐌹𐍃𐍄": "at least",
    "𐌰𐍄 𐌼𐌹𐌽𐌹𐍃𐍄": "at least",
    "𐌰𐍄 𐌼𐌹𐌽𐌹𐍃𐍄𐌰𐌼𐌼𐌰": "at least",
    "𐌰𐍄 𐌼𐌹𐌽𐌽𐌹𐍃𐍄𐌰𐌼𐌼𐌰": "at least",
    "𐌰𐍄-𐌼𐌹𐌽𐌽𐌹𐍃𐍄": "at least",
    "𐌰𐍄 𐌻𐌴𐌹𐍃𐍄": "at least",
    "𐌹𐌽 𐌼𐌹𐌽𐍃": "at least",
    "𐍃𐌽 𐌰𐌹𐌸𐌸𐌰𐌿 𐌼𐌰𐌹𐍃": "at least",
    "𐌹𐌱𐌽𐌰 𐌰𐌹𐌸𐌸𐌰𐌿 𐌼𐌰𐌹𐌶𐌰": "at least",
    # Gothic edge cases (with soft hyphen, etc.)
    "𐌰𐍄 𐌼𐌹𐌽" + "\xad" + "𐌽𐌹𐍃𐍄": "at least",
    "𐍆𐌰𐌿𐍂𐌰": "less than",
    "𐍆𐌰𐍅𐌹𐌶𐍉𐌽𐍃 𐌸𐌰𐌿": "less than",
    # Maltese
    "inqas minn": "less than",
    "mill-inqas": "at least",
    # Spanish
    "menos de": "less than",
    "al menos": "at least",
    "por lo menos": "at least",
    # Japanese
    "未満": "less than",
    "至少": "at least",
    # Chinese
    "少于": "less than",
    # Min Dong
    "cé-sēu": "less than",
    "ché-sāu": "at least",
    # Korean
    "미만": "less than",
    "적어도": "at least",
    # Arabic/Moroccan Arabic
    "أقل من": "less than",
    "على الأقل": "at least",
    # Moroccan Arabic (Darija)
    "على لأقل": "at least",
    "قل من": "less than",
    # Russian
    "меньше чем": "less than",
    "по крайней мере": "at least",
    "не менее": "at least",
    # Dutch
    "minder dan": "less than",
    "ten minste": "at least",
    "minstens": "at least",
    # Italian
    "meno di": "less than",
    "almeno": "at least",
    # Polish
    "mniej niż": "less than",
    "co najmniej": "at least",
    # Swedish
    "mindre än": "less than",
    "minst": "at least",
    # Danish/Norwegian
    "mindre end": "less than",
    "mindst": "at least",
    "færre enn": "less than",
    # Finnish
    "vähemmän kuin": "less than",
    "vähintään": "at least",
    # Czech
    "méně než": "less than",
    "alespoň": "at least",
    # Greek
    "λιγότερο από": "less than",
    "τουλάχιστον": "at least",
    # Hebrew
    "פחות מ": "less than",
    "פחות מ-": "less than",
    "לפחות": "at least",
    # Turkish
    "daha az": "less than",
    "en az": "at least",
    # Hindi
    "से कम": "less than",
    "कम से कम": "at least",
    # Punjabi
    "ਘੱਟ": "less than",
    "ਘੱਟੋ ਘੱਟ": "at least",
    # Thai
    "น้อยกว่า": "less than",
    "อย่างน้อย": "at least",
    # Vietnamese
    "ít hơn": "less than",
    "ít nhất": "at least",  # Indonesian/Malay
    "kurang dari": "less than",
    "setidaknya": "at least",
    "sekurang-kurangnya": "at least",
    # Banjar
    "paling sadikit": "at least",
    # Minangkabau
    "paliang indak": "less than",
    "paliang saketek": "at least",
    # Romanian
    "mai puțin de": "less than",
    "cel puțin": "at least",
    # Hungarian
    "kevesebb mint": "less than",
    "legalább": "at least",
    # Ukrainian
    "менше ніж": "less than",
    "щонайменше": "at least",
    # Serbian/Croatian/Bosnian
    "manje od": "less than",
    "najmanje": "at least",
    # Latvian/Lithuanian
    "mazāk nekā": "less than",
    "vismaz": "at least",
    "mažiau nei": "less than",
    "bent": "at least",
    # Slovenian/Slovak
    "manj kot": "less than",
    "vsaj": "at least",
    "menej ako": "less than",
    "aspoň": "at least",
    # Estonian
    "vähem kui": "less than",
    "vähemalt": "at least",
    # Malagasy
    "latsaky ny": "less than",
    "farafahakeliny": "at least",
    # Romansh
    "pli pauc che": "less than",
    "almain": "at least",
    # Ido/Interlingua/Interlingue
    "minora kam": "less than",
    "adminime": "at least",
    "minus de": "less than",
    "al minus": "at least",
    # Lingua Franca Nova (fix)
    "a la min": "at least",
    # Rundi/Kinyarwanda/Fon
    "munsi ya": "less than",
    "nibura": "at least",
    # Fon
    "hú mɔ̌": "less than",
    "bɔ̀": "at least",
    # Gaelic
    "nas lugha na": "less than",
    "co-dhiù": "at least",
    # Breton
    "nebeutoc'h eget": "less than",
    "d'an nebeutañ": "at least",
    # Old English
    "læs þonne": "less than",
    "tō lǣste": "at least",
    "hūru": "at least",
    # Lingua Franca Nova
    "min ca": "less than",
    "a min": "at least",
    # Kurdish
    "kêmtir ji": "less than",
    "kêmtirîn": "at least",
    "herî kêm": "at least",
    # Sundanese
    "kurang ti": "less than",
    "sahenteuna": "at least",
    "saeutikna": "at least",
    # Wolof
    "lu néew": "less than",
    "lu mat": "at least",
    # Ladin/Friulian
    "manco de": "less than",
    "almanco": "at least",
    "mancul di": "less than",
    "almancul": "at least",
    # Palatine German
    "wenischer als": "less than",
    "wenischer wie": "less than",
    # Tetum
    "menus husi": "less than",
    "pelumenus": "at least",
    # Belarusian
    "менш за": "less than",
    "не менш за": "at least",
    # Latin
    "minus quam": "less than",
    "ad minus": "at least",
    # Luxembourgish
    "manner wéi": "less than",
    "op d'mannst": "at least",
    # Manx
    "ny sloo na": "less than",
    "er y chooid sloo": "at least",
    # Occitan
    "mens de": "less than",
    "almens": "at least",
    # Somali
    "ka yar": "less than",
    "ugu yaraan": "at least",
    # Tibetan
    "ཉུང་མཐར": "at least",
    "ལས་ཉུང་བ": "less than",
    # Zulu/Sotho/Twi/Shona
    "okungenani": "at least",
    "bonyane": "at least",
    "ka tlase ho": "less than",
    # Shona
    "zvishoma nezvishoma": "at least",
    "shoma": "less than",
    # Twi
    "anyɛ yiye koraa no": "at least",
    "nsen": "less than",
    # Lower/Upper Sorbian
    "mjenjej ako": "less than",
    "nanejmjenjej": "at least",
    "mjenje hač": "less than",
    "znajmjeńša": "at least",
    # Lombard/Ligurian
    "men de": "less than",
    "almen": "at least",
    # Latgalian
    "mozuok kai": "less than",
    "vysmoz": "at least",
    # Pangasinan
    "mas dechut": "less than",
    "anggaman": "at least",
    # Macedonian
    "помалку од": "less than",
    "најмалку": "at least",
    # Madurese/Aymara
    "paling sakonè'": "at least",
    "palèng sakonè'": "at least",
    "juk'akiwa": "at least",
    # Faroese
    "minni enn": "less than",
    "í minsta lagi": "at least",
    # Saterland Frisian
    "minner as": "less than",
    "touminst": "at least",
    # Khmer/Lao
    "យ៉ាងហោចណាស់": "at least",
    "ຢ່າງໜ້ອຍ": "at least",
    # Hawaiian
    "emi ma lalo o": "less than",
    "ma ka liʻiliʻi loa": "at least",
    # Novial
    "minu kam": "less than",
    "adminim": "at least",
    # Sranan
    "moro mendri leki": "less than",
    "pikinmoro": "at least",
    # Albanian
    "më pak se": "less than",
    "të paktën": "at least",
    # Basque
    "gutxienez": "at least",
    "baino gutxiago": "less than",
    # Haitian
    "mwens pase": "less than",
    "omwen": "at least",
    # Pashto
    "لږترلږه": "at least",
    # Sardinian/Sicilian
    "prus pagu de": "less than",
    "menu di": "less than",
    "armenu": "at least",
    "comente a mìnimu": "at least",
    # Sinhala
    "අවම වශයෙන්": "at least",
    # Yiddish
    "ווייניקער ווי": "less than",
    # Armenian/Abkhaz
    "պակաս": "less than",
    "առնվազն": "at least",
    # Abkhaz
    "иацәажәаӡами": "at least",
    # Bengali
    "কম": "less than",
    "কমপক্ষে": "at least",
    # Catalan
    "menys de": "less than",
    "si més no": "at least",
    # Esperanto
    "malpli ol": "less than",
    "almenaŭ": "at least",
    # Swahili
    "chini ya": "less than",
    "angalau": "at least",
    # Persian/Farsi
    "کمتر از": "less than",
    # And many more...
    # Lingala (ln)
    "na sé ya": "at least",
    # Mon (mnw)
    "အောန်အိုတ်": "at least",
    "အောန်နူ": "at least",
    "ဗွဲအောန်အိုတ်": "at least",
    # Ossetian (os)
    "æппынкъаддæр": "at least",
    # Papiamento (pap)
    "por lo ménos": "at least",
    "menos cu": "less than",
    # Picard (pcd)
    "moins que": "less than",
    "moins d'": "less than",
    # Western Punjabi (pnb)
    "توں گھٹ": "less than",
    # Sindhi (sd)
    "گهٽ": "less than",
    # Shan (shn)
    "တီႈဢေႇသုတ်း": "at least",
    # Sakizaya (szy)
    "makaadih tu": "at least",
    # Tigrinya (ti)
    "እንተወሓደ": "at least",
    # Talysh (tly)
    "ləpə-ləp": "at least",
    # Taroko (trv)
    "ini dhuq 31": "at least",
    "hici bi o 30": "at least",
    # Tahitian (ty)
    "iti a'e i te": "less than",
    # Uzbek (uz)
    "kamroq": "less than",
    # Kalmyk (xal)
    "баһар гихд": "at least",
}


def fix_kwargs(kwargs_list: list[dict]) -> tuple[list[dict], bool]:
    """Fix translated constraint parameters.

    Returns:
        Tuple of (fixed kwargs list, whether any fixes were made)
    """
    fixed = False
    result = []
    for kwarg in kwargs_list:
        new_kwarg = kwarg.copy()
        if "relation" in new_kwarg and new_kwarg["relation"]:
            orig = new_kwarg["relation"]
            if orig in RELATION_MAPPINGS:
                new_kwarg["relation"] = RELATION_MAPPINGS[orig]
                fixed = True
        if "capital_relation" in new_kwarg and new_kwarg["capital_relation"]:
            orig = new_kwarg["capital_relation"]
            if orig in RELATION_MAPPINGS:
                new_kwarg["capital_relation"] = RELATION_MAPPINGS[orig]
                fixed = True
        if "let_relation" in new_kwarg and new_kwarg["let_relation"]:
            orig = new_kwarg["let_relation"]
            if orig in RELATION_MAPPINGS:
                new_kwarg["let_relation"] = RELATION_MAPPINGS[orig]
                fixed = True
        result.append(new_kwarg)
    return result, fixed


def main(lang_code: str) -> None:
    """Fix and upload a single language subset."""
    jsonl_path = Path(f"data/ifeval-{lang_code}.jsonl")
    if not jsonl_path.exists():
        print(f"❌ {lang_code}: File not found")
        return

    # Fix kwargs and create Parquet
    data = {"prompt": [], "instruction_id_list": [], "kwargs": [], "key": []}
    fixed_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fixed_kwargs, was_fixed = fix_kwargs(d["kwargs"])
            if was_fixed:
                fixed_count += 1
                d["kwargs"] = fixed_kwargs
            data["prompt"].append(d["prompt"])
            data["instruction_id_list"].append(d["instruction_id_list"])
            data["kwargs"].append(d["kwargs"])
            data["key"].append(d["key"])

    # Create and upload
    df = pd.DataFrame(data, dtype=object)
    parquet_path = Path(f"hf_upload/{lang_code}/test-00000-of-00001.parquet")
    parquet_path.parent.mkdir(exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path)

    # Upload
    repo_path = f"{lang_code}/test-00000-of-00001.parquet"
    upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=repo_path,
        repo_id="danish-foundation-models/multi-ifeval",
        repo_type="dataset",
        commit_message=f"fix({lang_code}): Fix translated constraint parameters",
    )

    print(f"✅ {lang_code}: Fixed {fixed_count}/{len(df)} examples, uploaded to HF Hub")

    # Verify
    print("   Verifying upload...")
    ds = load_dataset(
        "danish-foundation-models/multi-ifeval",
        lang_code,
        split="test",
        download_mode="force_redownload",
    )
    bad = sum(
        1
        for ex in ds
        for kw in ex.get("kwargs", [])
        if kw.get("relation")
        and kw.get("relation").lower() not in ["less than", "at least", "exactly"]
    )
    if bad == 0:
        print("   ✅ Verified: 0 bad examples")
    else:
        print(f"   ❌ FAILED: {bad} bad examples")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run src/scripts/fix_hf_subset.py <lang_code>")
        sys.exit(1)
    main(sys.argv[1])
