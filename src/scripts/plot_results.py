r"""Plot MultiIFEval evaluation results.

This script loads evaluation results from a JSONL file (EuroEval format) and generates
various visualisations for analysing model performance across languages.

Usage:
    uv run src/scripts/plot_results.py \
        --input results.jsonl \
        --output-dir ./figures \
        --plots bar scatter box compare \
        --multiwikiqa multiwikiqa_results.jsonl

Example:
    # Generate all plots from evaluation results
    uv run src/scripts/plot_results.py \\
        --input evaluation_results.jsonl \\
        --output-dir ./figures

    # Generate only bar chart and scatter plot
    uv run src/scripts/plot_results.py \\
        --input results.jsonl \\
        --output-dir ./output \\
        --plots bar scatter

    # Include MultiWikiQA correlation analysis
    uv run src/scripts/plot_results.py \\
        --input ifeval_results.jsonl \\
        --multiwikiqa multiwikiqa_results.jsonl \\
        --output-dir ./figures \\
        --plots scatter
"""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.ticker import MaxNLocator

# Language family mapping for 305 languages covered by MultiIFEval
LANGUAGE_FAMILIES = {
    # Indo-European
    "Indo-European": [
        "ab",
        "af",
        "sq",
        "hy",
        "as",
        "ast",
        "az",
        "eu",
        "be",
        "bn",
        "bs",
        "br",
        "bg",
        "ca",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "eo",
        "et",
        "fo",
        "fi",
        "fr",
        "fy",
        "gl",
        "ka",
        "de",
        "el",
        "gu",
        "ht",
        "ha",
        "he",
        "hi",
        "hu",
        "is",
        "ga",
        "it",
        "ja",
        "kn",
        "ks",
        "kk",
        "km",
        "ku",
        "ky",
        "lo",
        "la",
        "lv",
        "lt",
        "lb",
        "mk",
        "mg",
        "ms",
        "ml",
        "mt",
        "mi",
        "mr",
        "mn",
        "ne",
        "nb",
        "nn",
        "fa",
        "pl",
        "pt",
        "pa",
        "ro",
        "ru",
        "gd",
        "sr",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "es",
        "sw",
        "sv",
        "tg",
        "ta",
        "tt",
        "te",
        "th",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "cy",
        "xh",
        "yi",
        "yo",
        "zu",
    ],
    # Afro-Asiatic
    "Afro-Asiatic": ["ar", "am", "dv", "ff", "ha", "he", "kab", "mg", "mt", "om", "ti"],
    # Niger-Congo
    "Niger-Congo": [
        "ak",
        "bm",
        "ee",
        "ff",
        "ig",
        "ki",
        "lg",
        "ln",
        "luo",
        "ny",
        "rn",
        "rw",
        "sn",
        "st",
        "sw",
        "tn",
        "ts",
        "tw",
        "wo",
        "xh",
        "yo",
        "za",
        "zu",
    ],
    # Austronesian
    "Austronesian": [
        "akb",
        "ban",
        "bew",
        "bug",
        "ceb",
        "ch",
        "fj",
        "glk",
        "haw",
        "hil",
        "ho",
        "id",
        "jv",
        "kg",
        "kj",
        "kmb",
        "lg",
        "ln",
        "mad",
        "mg",
        "mh",
        "mi",
        "min",
        "ms",
        "na",
        "nhn",
        "nso",
        "ny",
        "pag",
        "pau",
        "quz",
        "sm",
        "tet",
        "tl",
        "tn",
        "to",
        "ts",
        "tw",
        "ty",
        "wo",
    ],
    # Sino-Tibetan
    "Sino-Tibetan": [
        "bo",
        "cdt",
        "cnh",
        "ctg",
        "dcc",
        "dhg",
        "dns",
        "gan",
        "hak",
        "hsi",
        "huu",
        "hxh",
        "ii",
        "jar",
        "jje",
        "kck",
        "kgh",
        "kht",
        "khw",
        "khw",
        "khw",
        "khw",
        "khz",
        "kjb",
        "kjp",
        "ksw",
        "lcm",
        "lif",
        "lis",
        "ljp",
        "lki",
        "lkt",
        "lmn",
        "lnz",
        "lqm",
        "lrk",
        "ltc",
        "lth",
        "luc",
        "lza",
        "mdf",
        "mgo",
        "mhm",
        "mhw",
        "mri",
        "mrr",
        "mup",
        "mwi",
        "mym",
        "myu",
        "nan",
        "ncd",
        "ngl",
        "nlc",
        "nso",
        "nvm",
        "nxq",
        "nzi",
        "oki",
        "omi",
        "ort",
        "otk",
        "pcr",
        "phk",
        "pkt",
        "pky",
        "pls",
        "plu",
        "pnb",
        "pny",
        "pom",
        "pon",
        "pse",
        "pso",
        "pst",
        "qtd",
        "rab",
        "rai",
        "rjs",
        "rmg",
        "rng",
        "rmy",
        "rmz",
        "rof",
        "rug",
        "rut",
        "rwm",
        "sag",
        "sam",
        "sck",
        "scy",
        "sdh",
        "seg",
        "sey",
        "sgs",
        "shn",
        "sja",
        "sjd",
        "sje",
        "sjk",
        "sjn",
        "sjt",
        "sju",
        "slr",
        "smn",
        "smx",
        "sna",
        "snd",
        "snk",
        "som",
        "sot",
        "sqh",
        "srd",
        "srn",
        "ssw",
        "ssy",
        "stq",
        "sty",
        "sun",
        "suz",
        "swg",
        "swv",
        "sxn",
        "syl",
        "szl",
        "tab",
        "taj",
        "tam",
        "taq",
        "tbz",
        "tca",
        "teb",
        "tem",
        "tet",
        "tgk",
        "tgl",
        "thk",
        "thv",
        "tie",
        "tig",
        "tir",
        "tkl",
        "tmh",
        "tna",
        "tnc",
        "tne",
        "tnp",
        "tnz",
        "tog",
        "toj",
        "top",
        "toq",
        "tpi",
        "tpl",
        "tpn",
        "tpq",
        "tqn",
        "tqw",
        "trs",
        "trw",
        "tsd",
        "tsg",
        "tsh",
        "tsi",
        "tsj",
        "tsp",
        "tss",
        "tsw",
        "tsy",
        "tte",
        "ttq",
        "ttu",
        "ttj",
        "ttk",
        "ttl",
        "ttm",
        "ttn",
        "tto",
        "ttp",
        "ttq",
        "ttr",
        "tts",
        "ttu",
        "ttw",
        "ttx",
        "tty",
        "tua",
        "tub",
        "tuc",
        "tud",
        "tue",
        "tuf",
        "tug",
        "tuh",
        "tui",
        "tuj",
        "tuk",
        "tul",
        "tum",
        "tuo",
        "tup",
        "tuq",
        "tur",
        "tus",
        "tuu",
        "tuv",
        "tuw",
        "tux",
        "tuy",
        "tuz",
        "tva",
        "tvd",
        "tve",
        "tvi",
        "tvk",
        "tvl",
        "tvm",
        "tvn",
        "tvo",
        "tvs",
        "tvt",
        "tvw",
        "tvx",
        "tvy",
        "twa",
        "twb",
        "twi",
        "twj",
        "twl",
        "twm",
        "twn",
        "two",
        "twp",
        "twq",
        "twr",
        "twt",
        "twu",
        "twv",
        "twx",
        "twy",
        "txa",
        "txb",
        "txc",
        "txe",
        "txg",
        "txh",
        "txi",
        "txj",
        "txm",
        "txn",
        "txo",
        "txq",
        "txr",
        "txs",
        "txt",
        "txu",
        "txx",
        "txy",
        "tye",
        "tyh",
        "tyi",
        "tyj",
        "tyl",
        "tyn",
        "typ",
        "tyq",
        "tyr",
        "tys",
        "tyt",
        "tyu",
        "tyv",
        "tyx",
        "tyy",
        "tyz",
        "tza",
        "tzh",
        "tzj",
        "tzl",
        "tzm",
        "tzn",
        "tzo",
        "tzp",
        "tzs",
        "tzt",
        "tzu",
        "tzx",
        "zh-cn",
        "zh-tw",
    ],
    # Turkic
    "Turkic": [
        "az",
        "ba",
        "chg",
        "chm",
        "cjs",
        "crh",
        "cv",
        "gag",
        "jun",
        "kaa",
        "kas",
        "kdr",
        "kjh",
        "kk",
        "krc",
        "krl",
        "ku",
        "kum",
        "ky",
        "lbe",
        "liv",
        "mnf",
        "ood",
        "os",
        "otk",
        "ru",
        "sah",
        "sel",
        "sjd",
        "sjk",
        "sjn",
        "sjt",
        "sju",
        "sty",
        "tk",
        "tr",
        "tt",
        "tyv",
        "ug",
        "uz",
    ],
    # Uralic
    "Uralic": [
        "et",
        "fi",
        "hu",
        "kv",
        "kca",
        "koi",
        "kpv",
        "krl",
        "lud",
        "mdf",
        "mhr",
        "mr",
        "myv",
        "ngn",
        "olo",
        "sms",
        "udm",
        "vep",
        "vot",
        "vro",
    ],
    # Austronesian
    "Malayo-Polynesian": ["id", "jv", "mg", "mi", "ms", "sm", "tl", "ty"],
    # Creole
    "Creole": ["ht", "pcm", "pap", "pis", "tpi"],
    # Isolates
    "Language Isolate": ["eu", "ka", "ko", "ja", "bur", "ain"],
    # Other
    "Other": ["und"],
}

# Resource level mapping based on linguistic resources availability
RESOURCE_LEVELS = {
    "high-resource": [
        "en",
        "zh-cn",
        "es",
        "de",
        "fr",
        "ja",
        "ko",
        "pt",
        "ru",
        "it",
        "nl",
        "pl",
        "tr",
        "vi",
        "ar",
        "th",
        "id",
        "hi",
        "bn",
        "ta",
        "te",
        "mr",
        "gu",
        "kn",
        "ml",
        "pa",
        "ur",
        "fa",
        "he",
        "el",
        "cs",
        "hu",
        "ro",
        "uk",
        "sv",
        "da",
        "no",
        "nb",
        "fi",
        "sk",
        "bg",
        "hr",
        "sr",
        "lt",
        "lv",
        "et",
        "sl",
        "ca",
        "eu",
        "gl",
    ],
    "medium-resource": [
        "af",
        "sq",
        "hy",
        "az",
        "be",
        "bs",
        "my",
        "km",
        "ka",
        "kk",
        "ky",
        "lo",
        "mk",
        "mn",
        "ne",
        "ps",
        "si",
        "uz",
        "cy",
        "is",
        "ga",
        "mt",
        "gd",
        "br",
        "fy",
        "lb",
        "mi",
        "sm",
        "to",
        "fj",
        "haw",
        "ty",
        "mg",
        "ny",
        "sn",
        "st",
        "tn",
        "ts",
        "ve",
        "xh",
        "zu",
        "yo",
        "ig",
        "ha",
        "sw",
        "am",
        "ti",
        "so",
        "rm",
        "oc",
        "co",
        "sc",
        "wa",
        "li",
        "zea",
        "stq",
        "bar",
        "als",
        "gsw",
        "ksh",
        "pdc",
        "pfl",
        "sli",
        "vmf",
        "wae",
        "cim",
        "mhn",
        "kdr",
        "ydd",
        "yv",
        "lad",
        "grb",
        "grc",
        "got",
        "non",
        "ang",
        "enm",
        "gmh",
        "goh",
        "osx",
        "odt",
        "dum",
        "fro",
        "frm",
        "xno",
        "pro",
        "oca",
        "roa-opt",
        "mwl",
        "ext",
        "ast",
        "an",
        "arg",
        "mdf",
        "myv",
        "kv",
        "koi",
        "udm",
        "chm",
        "mrj",
        "mhr",
        "kpv",
        "krc",
        "kum",
        "kjh",
        "alt",
        "sah",
        "tyv",
        "krl",
        "vep",
        "izh",
        "vot",
        "liv",
        "sms",
        "smn",
        "sjd",
        "sjk",
        "sje",
        "smj",
        "sju",
        "sd",
        "ks",
        "bho",
        "mag",
        "mai",
        "new",
        "pi",
        "sa",
        "pra",
        "apb",
        "ase",
        "bfi",
        "bzs",
        "csc",
        "dse",
        "ecs",
        "fss",
        "gsg",
        "gus",
        "hab",
        "hds",
        "hks",
        "hps",
        "hsh",
        "jbs",
        "jek",
        "jks",
        "jos",
        "jls",
        "lbs",
        "lsg",
        "lsp",
        "lsr",
        "mzc",
        "mzj",
        "nbs",
        "ncs",
        "nsr",
        "nsl",
        "nzs",
        "psd",
        "pso",
        "pvl",
        "rsl",
        "rsm",
        "sfs",
        "sgg",
        "sqk",
        "svk",
        "swl",
        "tcs",
        "tjd",
        "tsm",
        "tss",
        "tzy",
        "ugn",
        "ukl",
        "uks",
        "zib",
        "zsl",
    ],
    "low-resource": [],  # All others are low-resource by default
}


def get_language_family(language_code: str) -> str:
    """Get the language family for a given language code.

    Args:
        language_code:
            ISO 639-1 or ISO 639-3 language code.

    Returns:
        The language family name, or 'Other' if not found.
    """
    for family, codes in LANGUAGE_FAMILIES.items():
        if language_code in codes:
            return family
    return "Other"


def get_resource_level(language_code: str) -> str:
    """Get the resource level for a given language code.

    Args:
        language_code:
            ISO 639-1 or ISO 639-3 language code.

    Returns:
        Resource level: 'high-resource', 'medium-resource', or 'low-resource'.
    """
    if language_code in RESOURCE_LEVELS["high-resource"]:
        return "high-resource"
    if language_code in RESOURCE_LEVELS["medium-resource"]:
        return "medium-resource"
    return "low-resource"


def load_evaluation_results(file_path: Path) -> pd.DataFrame:
    """Load evaluation results from a JSONL file.

    Args:
        file_path:
            Path to the JSONL file containing evaluation results.

    Returns:
        DataFrame with columns: model, language, dataset, accuracy, timestamp.

    Raises:
        ValueError:
            If no valid records found in the file.
    """
    records = []
    with file_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # EuroEval format: dataset contains language info
                dataset_name = record.get("dataset", "")
                model_name = record.get("model", "unknown")

                # Extract language code from dataset name
                # Format: multi-ifeval-{lang} or ifeval-{lang}
                language_code = None
                if "multi-ifeval-" in dataset_name:
                    language_code = dataset_name.split("multi-ifeval-")[-1]
                elif "ifeval-" in dataset_name:
                    language_code = dataset_name.split("ifeval-")[-1]

                if not language_code:
                    logger.warning(
                        f"Could not extract language code from dataset: {dataset_name}"
                    )
                    continue

                # Extract accuracy from results
                results = record.get("results", {})
                total_results = results.get("total", {})
                accuracy = total_results.get("test_accuracy")

                if accuracy is None:
                    # Try alternative metric names
                    accuracy = total_results.get("accuracy")
                    if accuracy is None:
                        logger.warning(
                            f"No accuracy found for {dataset_name}, skipping"
                        )
                        continue

                records.append(
                    {
                        "model": model_name,
                        "language": language_code,
                        "dataset": dataset_name,
                        "accuracy": accuracy,
                        "timestamp": record.get("timestamp", ""),
                        "translation_type": record.get("translation_type", "machine"),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {e}")
                continue

    if not records:
        msg = f"No valid records found in {file_path}"
        raise ValueError(msg)

    return pd.DataFrame(records)


def load_multiwikiqa_results(file_path: Path | None) -> pd.DataFrame | None:
    """Load MultiWikiQA results from a JSONL file.

    Args:
        file_path:
            Path to the JSONL file containing MultiWikiQA results, or None.

    Returns:
        DataFrame with columns: model, language, f1, em, or None if file_path is None.
    """
    if file_path is None:
        return None

    records = []
    with file_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                dataset_name = record.get("dataset", "")
                model_name = record.get("model", "unknown")

                # Extract language code from dataset name
                # Format: multi-wiki-qa-{lang}
                language_code = None
                if "multi-wiki-qa-" in dataset_name:
                    language_code = dataset_name.split("multi-wiki-qa-")[-1]

                if not language_code:
                    continue

                # Extract F1 and EM scores
                results = record.get("results", {})
                total_results = results.get("total", {})
                f1_score = total_results.get("test_f1")
                em_score = total_results.get("test_em")

                if f1_score is None:
                    continue

                records.append(
                    {
                        "model": model_name,
                        "language": language_code,
                        "f1": f1_score,
                        "em": em_score if em_score else np.nan,
                    }
                )
            except json.JSONDecodeError:
                continue

    if not records:
        return None

    return pd.DataFrame(records)


def create_bar_chart(
    df: pd.DataFrame, output_path: Path, top_n: int | None = None
) -> None:
    """Create a bar chart showing model performance across languages.

    Args:
        df:
            DataFrame with evaluation results.
        output_path:
            Path to save the plot.
        top_n:
            Number of top languages to show. If None, shows all.
    """
    logger.info("Creating bar chart of model performance")

    # Group by language and calculate mean accuracy
    lang_perf = (
        df.groupby("language")["accuracy"].agg(["mean", "std", "count"]).reset_index()
    )
    lang_perf.columns = ["language", "mean_accuracy", "std_accuracy", "num_models"]

    # Sort by mean accuracy
    lang_perf = lang_perf.sort_values("mean_accuracy", ascending=False)

    # Limit to top_n if specified
    if top_n:
        lang_perf = lang_perf.head(top_n)

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.bar(
        range(len(lang_perf)),
        lang_perf["mean_accuracy"],
        yerr=lang_perf["std_accuracy"],
        capsize=3,
        color="steelblue",
        alpha=0.8,
    )

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "MultiIFEval Performance Across Languages"
        + (f" (Top {top_n})" if top_n else ""),
        fontsize=14,
        fontweight="bold",
    )

    # Set x-axis labels
    ax.set_xticks(range(len(lang_perf)))
    ax.set_xticklabels(lang_perf["language"], rotation=90, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    logger.info(f"Bar chart saved to {output_path}")


def create_scatter_plot(
    ifeval_df: pd.DataFrame, multiwikiqa_df: pd.DataFrame | None, output_path: Path
) -> None:
    """Create a scatter plot showing correlation between MultiIFEval and MultiWikiQA.

    Args:
        ifeval_df:
            DataFrame with MultiIFEval results.
        multiwikiqa_df:
            DataFrame with MultiWikiQA results, or None.
        output_path:
            Path to save the plot.
    """
    if multiwikiqa_df is None:
        logger.warning("No MultiWikiQA data provided, skipping scatter plot")
        return

    logger.info("Creating scatter plot of MultiIFEval vs MultiWikiQA correlation")

    # Merge the two dataframes on language and model
    merged = pd.merge(
        ifeval_df[["model", "language", "accuracy"]],
        multiwikiqa_df[["model", "language", "f1"]],
        on=["model", "language"],
        how="inner",
    )

    if len(merged) < 2:
        logger.warning("Not enough overlapping data points for correlation plot")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter points
    ax.scatter(
        merged["f1"],
        merged["accuracy"],
        alpha=0.6,
        s=50,
        c="navy",
        edgecolors="white",
        linewidth=0.5,
    )

    # Calculate and plot correlation
    correlation = merged["f1"].corr(merged["accuracy"])
    if not np.isnan(correlation):
        # Fit a regression line
        z = np.polyfit(merged["f1"], merged["accuracy"], 1)
        p = np.poly1d(z)
        ax.plot(
            merged["f1"],
            p(merged["f1"]),
            "r--",
            alpha=0.8,
            label=f"Correlation: r={correlation:.3f}",
        )

    ax.set_xlabel("MultiWikiQA F1 Score", fontsize=12)
    ax.set_ylabel("MultiIFEval Accuracy (%)", fontsize=12)
    ax.set_title("MultiIFEval vs MultiWikiQA Performance Correlation", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    logger.info(f"Scatter plot saved to {output_path}")


def create_box_plot(
    df: pd.DataFrame, output_path: Path, group_by: str = "family"
) -> None:
    """Create a box plot showing performance distribution by group.

    Args:
        df:
            DataFrame with evaluation results.
        output_path:
            Path to save the plot.
        group_by:
            Either 'family' or 'resource' to group by language family or resource level.
    """
    logger.info(f"Creating box plot grouped by {group_by}")

    # Add grouping column
    df = df.copy()
    if group_by == "family":
        df["group"] = df["language"].apply(get_language_family)
        group_label = "Language Family"
    else:  # resource
        df["group"] = df["language"].apply(get_resource_level)
        group_label = "Resource Level"

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for box plot
    groups = df.groupby("group")["accuracy"]
    group_names = sorted(groups.groups.keys())
    group_data = [groups.get_group(name).values for name in group_names]

    # Create box plot
    bp = ax.boxplot(
        group_data,
        labels=group_names,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanline=True,
    )

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(group_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlabel(group_label, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"MultiIFEval Performance Distribution by {group_label}",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Rotate x-axis labels if there are many groups
    if len(group_names) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    logger.info(f"Box plot saved to {output_path}")


def create_comparison_plot(
    df: pd.DataFrame, output_path: Path, language: str = "da"
) -> None:
    r"""Create a comparison plot of machine-translated vs manually-translated results.

    Args:
        df:
            DataFrame with evaluation results including translation_type.
        output_path:
            Path to save the plot.
        language:
            Language code to compare (default: Danish 'da').
    """
    logger.info(f"Creating comparison plot for language {language}")

    # Filter to the specified language
    lang_df = df[df["language"] == language].copy()

    if len(lang_df) == 0:
        logger.warning(f"No data found for language {language}")
        return

    # Check if we have both translation types
    translation_types = lang_df["translation_type"].unique()

    if len(translation_types) < 2:
        logger.warning(
            f"Only one translation type found for {language}: {translation_types}. "
            "Cannot create comparison plot."
        )
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [
        lang_df[lang_df["translation_type"] == tt]["accuracy"].values
        for tt in translation_types
    ]

    bp = ax.boxplot(
        data_to_plot,
        labels=[tt.replace("_", " ").title() for tt in translation_types],
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanline=True,
    )

    # Color the boxes
    colors = plt.cm.Paired(np.linspace(0, 1, len(translation_types)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Machine-Translated vs Manually-Translated Performance ({language.upper()})",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    logger.info(f"Comparison plot saved to {output_path}")


@click.command()
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the evaluation results JSONL file (EuroEval format).",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("./figures"),
    help="Directory to save the generated plots (default: ./figures).",
)
@click.option(
    "--plots",
    "plot_types",
    type=click.STRING,
    default="bar",
    help="Plots to generate: bar, scatter, box, compare (default: bar).",
)
@click.option(
    "--multiwikiqa",
    "multiwikiqa_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to MultiWikiQA results JSONL file for correlation analysis.",
)
@click.option(
    "--top-n",
    "top_n",
    type=click.INT,
    default=50,
    help="Number of top languages to show in bar chart (default: 50).",
)
@click.option(
    "--language",
    "comparison_language",
    type=click.STRING,
    default="da",
    help="Language code for translation comparison plot (default: da).",
)
@click.option(
    "--group-by",
    "group_by",
    type=click.Choice(["family", "resource"]),
    default="family",
    help="Grouping for box plot: language family or resource level (default: family).",
)
@click.option(
    "--model-filter",
    "model_filter",
    type=click.STRING,
    default=None,
    help="Filter results to specific model name (optional).",
)
def main(
    input_file: Path,
    output_dir: Path,
    plot_types: str,
    multiwikiqa_file: Path | None,
    top_n: int,
    comparison_language: str,
    group_by: str,
    model_filter: str | None,
) -> None:
    r"""Plot MultiIFEval evaluation results.

    Generates various visualisations for analysing model performance across languages
    from EuroEval-format evaluation results.

    Raises:
        ValueError:
            If invalid plot types are requested or no results match the filter.
    """
    # Parse plot types
    requested_plots = [p.strip().lower() for p in plot_types.split(",")]
    valid_plots = {"bar", "scatter", "box", "compare"}
    invalid_plots = set(requested_plots) - valid_plots

    if invalid_plots:
        logger.error(f"Invalid plot types: {invalid_plots}")
        raise ValueError(f"Invalid plot types: {invalid_plots}")

    # Load evaluation results
    logger.info(f"Loading evaluation results from {input_file}")
    df = load_evaluation_results(input_file)

    # Apply model filter if specified
    if model_filter:
        df = df[df["model"].str.contains(model_filter, case=False, na=False)]
        if len(df) == 0:
            logger.error(f"No results found for model filter: {model_filter}")
            raise ValueError(f"No results found for model filter: {model_filter}")

    logger.info(f"Loaded {len(df)} evaluation records")

    # Load MultiWikiQA results if needed
    multiwikiqa_df = None
    if "scatter" in requested_plots:
        if multiwikiqa_file:
            multiwikiqa_df = load_multiwikiqa_results(multiwikiqa_file)
            if multiwikiqa_df is not None:
                logger.info(f"Loaded {len(multiwikiqa_df)} MultiWikiQA records")
        else:
            logger.warning(
                "Scatter plot requested but no MultiWikiQA file provided. "
                "Use --multiwikiqa to enable correlation analysis."
            )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate requested plots
    if "bar" in requested_plots:
        create_bar_chart(
            df=df, output_path=output_dir / "performance_bar_chart.png", top_n=top_n
        )

    if "scatter" in requested_plots:
        create_scatter_plot(
            ifeval_df=df,
            multiwikiqa_df=multiwikiqa_df,
            output_path=output_dir / "multiifeval_vs_multiwikiqa.png",
        )

    if "box" in requested_plots:
        create_box_plot(
            df=df,
            output_path=output_dir / f"performance_box_plot_{group_by}.png",
            group_by=group_by,
        )

    if "compare" in requested_plots:
        create_comparison_plot(
            df=df,
            output_path=output_dir
            / f"translation_comparison_{comparison_language}.png",
            language=comparison_language,
        )

    logger.info("Plot generation complete")


if __name__ == "__main__":
    main()
