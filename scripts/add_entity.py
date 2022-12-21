#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def read_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


def write_jsonl(fname, data):
    with open(fname, "w") as f:
        for d in data:
            print(json.dumps(d), file=f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("db_file", type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--description")
    p.add_argument("--url")
    p.add_argument("--source")
    p.add_argument("--source_id")
    args = p.parse_args()

    if not args.db_file.exists():
        args.db_file.mkdir(exist_ok=True, parents=True)
        args.db_file.open("w").close()

    data = read_jsonl(args.db_file)

    if not args.description:
        args.description = "geen omschrijving beschikbaar"

    if not args.url:
        args.url = ""

    if not args.source:
        if data:
            args.source = data[0]["source"]
        else:
            args.source = args.db_file.stem

    if not args.source_id:
        args.source_id = args.title.lower()

    data.append(
        {
            "title": args.title,
            "description": args.description,
            "url": args.url,
            "source": args.source,
            "source_id": args.source_id,
        }
    )
    write_jsonl(args.db_file, data)
