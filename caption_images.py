#!/usr/bin/env python3
"""
caption_images.py

Requirements:
  pip install google-genai psycopg[binary] requests python-dotenv pillow

Env:
  GEMINI_API_KEY  -> Google AI Studio API key
  DATABASE_URL    -> e.g. postgresql://user:pass@host:5432/fruision

What it does:
  - Adds ai_caption (TEXT) and ai_captioned_at (TIMESTAMPTZ) if missing.
  - Finds images with no caption, calls Gemini, updates the row.
"""

import os
import argparse
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import requests
import psycopg
from psycopg.rows import dict_row

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


def add_columns(conn, table="image", caption_col="ai_caption", ts_col="ai_captioned_at"):
    """Add caption columns idempotently."""
    with conn.cursor() as cur:
        cur.execute(f'ALTER TABLE public.{table} ADD COLUMN IF NOT EXISTS {caption_col} TEXT;')
        cur.execute(f'ALTER TABLE public.{table} ADD COLUMN IF NOT EXISTS {ts_col} TIMESTAMPTZ;')
    conn.commit()


def list_columns(conn, table):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
        """, (table,))
        return [r["column_name"] for r in cur.fetchall()]


def pick_image_url_col(conn, table, preferred=None):
    cols = list_columns(conn, table)
    if preferred and preferred in cols:
        return preferred
    for cand in ("image_url", "imageurl", "url", "path"):
        if cand in cols:
            return cand
    raise RuntimeError(f"Could not find an image URL column. Available: {cols}")


def _load_local_image(path):
    data = Path(path).expanduser().read_bytes()
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return data, mime


def fetch_image_bytes(path_or_url, timeout=20):
    parsed = urlparse(str(path_or_url))
    scheme = parsed.scheme.lower()

    if scheme in ("http", "https"):
        r = requests.get(path_or_url, timeout=timeout)
        r.raise_for_status()
        mime = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            mime = "image/jpeg"  # fallback if server omits/lying about type
        return r.content, mime

    if scheme in ("", "file"):
        local_path = parsed.path if scheme == "file" else path_or_url
        return _load_local_image(local_path)

    raise ValueError(f"Unsupported scheme for image source: {path_or_url}")


def caption_with_gemini(client, image_source, model, temperature):
    data, mime = fetch_image_bytes(image_source)
    image_part = types.Part.from_bytes(data=data, mime_type=mime)  # bytes input for images
    prompt = (
        "Write a concise, natural English caption (<= 18 words). "
        "No hashtags, no quotes, sentence case."
    )
    resp = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return (resp.text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    ap.add_argument("--table", default="image")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--image-url-col", default=None, help="Override if your column is not image_url")
    ap.add_argument("--caption-col", default="ai_caption")
    ap.add_argument("--ts-col", default="ai_captioned_at")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for deterministic).")
    ap.add_argument("--only-id", type=int, help="Process a single image id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-path", action="append", help="Caption one or more local image paths (repeatable).")
    ap.add_argument("--local-dir", action="append", help="Caption all images within a directory (non-recursive).")
    ap.add_argument("--recursive", action="store_true", help="When combined with --local-dir, recurse into subdirectories.")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Set --api-key or GEMINI_API_KEY")

    # Gemini client (reads GEMINI_API_KEY from env)
    client = genai.Client(api_key=args.api_key)

    local_paths = []
    if args.local_path:
    # If it's a URL, don't convert with Path (it breaks it)
        for p in args.local_path:
            if str(p).startswith("http"):
                local_paths.append(str(p))  # keep URL as string
            else:
                local_paths.append(Path(p).expanduser())
    if args.local_dir:
        for directory in args.local_dir:
            base = Path(directory).expanduser()
            iterator = base.rglob("*") if args.recursive else base.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                    local_paths.append(candidate)

    if local_paths:
        logging.info("Running in local mode against %d file(s).", len(local_paths))
        processed = 0
        for path in local_paths:
            try:
                caption = caption_with_gemini(client, str(path), args.model, args.temperature)
                if not caption:
                    logging.warning("Empty caption for %s", path)
                    continue
                print(f"{path}\t{caption}")
                processed += 1
            except Exception as exc:
                logging.exception("Failed %s: %s", path, exc)
        print(f"Done. Captioned {processed} file(s).")
        return

    if not args.db_url:
        raise SystemExit("Set --db-url or DATABASE_URL (or provide local image inputs).")

    with psycopg.connect(args.db_url) as conn:
        add_columns(conn, args.table, args.caption_col, args.ts_col)
        url_col = pick_image_url_col(conn, args.table, args.image_url_col)

        where = f"({args.caption_col} IS NULL OR {args.caption_col} = '')"
        params = []
        if args.only_id:
            where += f" AND {args.id_col} = %s"
            params.append(args.only_id)

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT {args.id_col} AS id, {url_col} AS url
                FROM public.{args.table}
                WHERE {where}
                ORDER BY {args.id_col} ASC
                LIMIT %s
                """,
                (*params, args.limit),
            )
            rows = cur.fetchall()

        processed = 0
        for row in rows:
            img_id, url = row["id"], row["url"]
            try:
                caption = caption_with_gemini(client, url, args.model, args.temperature)
                if not caption:
                    logging.warning("Empty caption for id=%s", img_id)
                    continue

                if args.dry_run:
                    print(f"[DRY] {img_id}\t{caption}")
                    continue

                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE public.{args.table}
                        SET {args.caption_col} = %s,
                            {args.ts_col} = NOW()
                        WHERE {args.id_col} = %s
                        """,
                        (caption, img_id),
                    )
                conn.commit()
                print(f"{img_id}\t{caption}")
                processed += 1
            except Exception as e:
                logging.exception("Failed id=%s url=%s: %s", img_id, url, e)
                conn.rollback()

        print(f"Done. Updated {processed} rows.")


if __name__ == "__main__":
    main()












