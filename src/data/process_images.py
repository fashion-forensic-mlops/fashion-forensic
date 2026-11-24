from __future__ import annotations

from collections import Counter
from pathlib import Path

import polars as pl
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "images.yaml"
ARTICLES_PATH = PROJECT_ROOT / "data" / "raw" / "articles.csv"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)["images"]

INPUT_DIR = PROJECT_ROOT / config["input_directory"]
OUTPUT_DIR = PROJECT_ROOT / config["output_directory"]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

IMAGE_WIDTH = config["output_size"]["width"]
IMAGE_HEIGHT = config["output_size"]["height"]
IMAGE_FORMAT = config["format"].upper()
COLOR_MODE = config["color_mode"]

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not ARTICLES_PATH.is_file():
    raise FileNotFoundError(f"Missing article metadata: {ARTICLES_PATH}")

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Missing image directory: {INPUT_DIR}")

if IMAGE_FORMAT != "JPEG":
    raise ValueError(f"Unsupported configured format: {IMAGE_FORMAT}")

if COLOR_MODE != "RGB":
    raise ValueError(f"Unsupported configured color mode: {COLOR_MODE}")

if config["resize_method"] != "contain":
    raise ValueError("This script currently supports resize_method='contain'.")

if not config["preserve_aspect_ratio"] or not config["padding"]:
    raise ValueError("Expected aspect-ratio preservation and padding to be enabled.")

articles = pl.read_csv(
    ARTICLES_PATH,
    columns=["article_id"],
    infer_schema_length=10000,
).with_columns(
    pl.col("article_id").cast(pl.Int64)
)

if articles["article_id"].n_unique() != articles.height:
    raise ValueError("Duplicate article IDs found in articles.csv.")

#Match the original preprocessing notebook: process JPG files
image_files = sorted(
    path
    for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() == ".jpg"
)

if not image_files:
    raise ValueError(f"No JPG images found in {INPUT_DIR}.")

image_index = pl.DataFrame({
    "image_filename": [path.stem for path in image_files],
    "image_path": [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in image_files
    ],
}).with_columns(
    pl.col("image_filename")
    .cast(pl.Int64)
    .alias("article_id")
)

duplicate_images = (
    image_index
    .group_by("article_id")
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_images.height > 0:
    raise ValueError(
        f"Duplicate image IDs found: {duplicate_images.height}"
    )

unmatched_images = image_index.join(
    articles,
    on="article_id",
    how="anti",
)

if unmatched_images.height > 0:
    raise ValueError(
        f"Images without article metadata: {unmatched_images.height}"
    )

article_image_mapping = articles.join(
    image_index.select(["article_id", "image_path"]),
    on="article_id",
    how="left",
)

missing_images = article_image_mapping.filter(
    pl.col("image_path").is_null()
)

article_image_mapping.write_parquet(
    INTERIM_DIR / "article_image_mapping.parquet"
)

missing_images.write_parquet(
    INTERIM_DIR / "articles_without_images.parquet"
)

print("Articles:", articles.height)
print("Original JPG images:", len(image_files))
print("Articles without original images:", missing_images.height)


#Audit original image integrity
raw_corrupted_images = []

for image_path in tqdm(image_files, desc="Auditing raw images"):
    try:
        with Image.open(image_path) as img:
            img.verify()

    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raw_corrupted_images.append({
            "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "error": str(error),
        })

corrupted_image_report = pl.DataFrame(
    raw_corrupted_images,
    schema={
        "image_path": pl.String,
        "error": pl.String,
    },
)

corrupted_image_report.write_parquet(
    PROJECT_ROOT / "data" / "processed" / "corrupted_images.parquet"
)

print("Corrupted raw images:", corrupted_image_report.height)


#Only remove JPG files inside the configured generated-image directory
#Do not delete other files or directories
for old_image in OUTPUT_DIR.glob("*.jpg"):
    old_image.unlink()

def preprocess_image(image_path: Path, output_path: Path) -> None:
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT

    with Image.open(image_path) as img:
        img = img.convert(COLOR_MODE)

        resized = ImageOps.contain(
            img,
            (width, height),
            method=Image.Resampling.LANCZOS,
        )

        padding_color = tuple(config["padding_color"])

        canvas = Image.new(
            COLOR_MODE,
            (width, height),
            padding_color,
        )

        x = (width - resized.width) // 2
        y = (height - resized.height) // 2

        canvas.paste(resized, (x, y))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        canvas.save(
            output_path,
            format=IMAGE_FORMAT,
        )

processing_errors = []

for image_path in tqdm(image_files, desc="Processing images"):
    article_id = image_path.stem

    output_filename = config["filename_format"].format(
        article_id=article_id
    )

    output_path = OUTPUT_DIR / output_filename

    try:
        preprocess_image(image_path, output_path)

    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        processing_errors.append({
            "article_id": article_id,
            "image_path": image_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "error": str(error),
        })

processing_error_report = pl.DataFrame(
    processing_errors,
    schema={
        "article_id": pl.String,
        "image_path": pl.String,
        "error": pl.String,
    },
)

processing_error_report.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "image_processing_errors.parquet"
)

print("Processing errors:", len(processing_errors))

processed_files = sorted(OUTPUT_DIR.glob("*.jpg"))

processed_dimensions = Counter()
processed_modes = Counter()
processed_formats = Counter()
validation_errors = []

for path in tqdm(processed_files, desc="Validating images"):
    try:
        with Image.open(path) as img:
            dimensions = img.size
            mode = img.mode
            image_format = img.format

            processed_dimensions[dimensions] += 1
            processed_modes[mode] += 1
            processed_formats[image_format] += 1

            if dimensions != (IMAGE_WIDTH, IMAGE_HEIGHT):
                validation_errors.append({
                    "image": path.name,
                    "error": f"Incorrect dimensions: {dimensions}",
                })

            if mode != COLOR_MODE:
                validation_errors.append({
                    "image": path.name,
                    "error": f"Incorrect color mode: {mode}",
                })

            if image_format != IMAGE_FORMAT:
                validation_errors.append({
                    "image": path.name,
                    "error": f"Incorrect format: {image_format}",
                })

            img.verify()

    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        validation_errors.append({
            "image": path.name,
            "error": str(error),
        })

validation_error_report = pl.DataFrame(
    validation_errors,
    schema={
        "image": pl.String,
        "error": pl.String,
    },
)

validation_error_report.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "image_validation_errors.parquet"
)

processed_image_index = pl.DataFrame({
    "image_filename": [path.name for path in processed_files],
    "processed_image_path": [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in processed_files
    ],
}).with_columns(
    pl.col("image_filename")
    .str.replace(".jpg", "")
    .cast(pl.Int64)
    .alias("article_id")
)

processed_image_mapping = article_image_mapping.join(
    processed_image_index.select([
        "article_id",
        "processed_image_path",
    ]),
    on="article_id",
    how="left",
)

processed_image_mapping.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "article_processed_image_mapping.parquet"
)

processed_paths = [
    PROJECT_ROOT / path
    for path in (
        processed_image_mapping["processed_image_path"]
        .drop_nulls()
        .to_list()
    )
]

missing_processed_files = [
    path for path in processed_paths
    if not path.is_file()
]

processed_image_count = (
    processed_image_mapping["processed_image_path"]
    .is_not_null()
    .sum()
)

dimension_summary = pl.DataFrame([
    {
        "width": dimensions[0],
        "height": dimensions[1],
        "count": count,
    }
    for dimensions, count in processed_dimensions.items()
], schema={
    "width": pl.Int64,
    "height": pl.Int64,
    "count": pl.Int64,
})

dimension_summary.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "image_dimension_summary.parquet"
)

color_mode_summary = pl.DataFrame([
    {
        "color_mode": mode,
        "count": count,
    }
    for mode, count in processed_modes.items()
], schema={
    "color_mode": pl.String,
    "count": pl.Int64,
})

color_mode_summary.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "image_color_mode_summary.parquet"
)

image_pipeline_summary = pl.DataFrame({
    "metric": [
        "total_articles",
        "articles_with_original_images",
        "articles_without_original_images",
        "processed_images",
        "missing_processed_files",
        "output_width",
        "output_height",
    ],
    "value": [
        processed_image_mapping.height,
        article_image_mapping["image_path"].is_not_null().sum(),
        article_image_mapping["image_path"].is_null().sum(),
        len(processed_files),
        len(missing_processed_files),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    ],
})

image_pipeline_summary.write_parquet(
    PROJECT_ROOT
    / "data" / "processed"
    / "image_pipeline_summary.parquet"
)

print("\n========== IMAGE PIPELINE SUMMARY ==========")
print("Total articles:", processed_image_mapping.height)
print("Original images:", len(image_files))
print("Processed images:", len(processed_files))
print("Articles with processed images:", processed_image_count)
print("Articles without processed images:",
      processed_image_mapping.height - processed_image_count)
print("Missing physical files:", len(missing_processed_files))
print("Processing errors:", len(processing_errors))
print("Validation errors:", len(validation_errors))
print("Dimensions:", processed_dimensions)
print("Color modes:", processed_modes)
print("Formats:", processed_formats)

assert processed_image_mapping.height == articles.height
assert processed_image_mapping["article_id"].n_unique() == articles.height
assert processed_image_count == len(processed_files)
assert len(missing_processed_files) == 0
assert len(processing_errors) == 0
assert len(validation_errors) == 0

assert (
    processed_image_mapping["processed_image_path"]
    .is_null()
    .sum()
    == missing_images.height
)

assert all(
    dimensions == (IMAGE_WIDTH, IMAGE_HEIGHT)
    for dimensions in processed_dimensions
)

assert set(processed_modes.keys()) <= {COLOR_MODE}
assert set(processed_formats.keys()) <= {IMAGE_FORMAT}

print("\nIMAGE PIPELINE: ALL CHECKS PASSED")