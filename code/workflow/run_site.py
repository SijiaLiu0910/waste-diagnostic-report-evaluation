# -*- coding: utf-8 -*-
"""Four-group report generation for one observation block.

A: MLLM perception, contextual alignment, and diagnosis using internal knowledge.
B: MLLM perception, contextual alignment, regulatory retrieval, and diagnosis.
C: YOLO perception, contextual alignment, and diagnosis using internal knowledge.
D: YOLO perception, contextual alignment, regulatory retrieval, and diagnosis.

Groups run independently. Supply the observation block ID and group at the command line; results are saved to the corresponding output directory.
"""
import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESTUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
RET_OPT = os.path.join(RESTUDY, "code", "retrieval")
sys.path.insert(0, RET_OPT)
import common  # noqa: E402
import formal_pipeline as fp  # noqa: E402
import prompts  # noqa: E402

SITE = "S02-A"
GROUPS = None
OUT_ROOT = "workflow_runs"
CHAT_RETRIES = 1
VL_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
IMG_DIR = os.path.join(RESTUDY, "data", "dataset_B", "images")
META_CSV = os.path.join(RESTUDY, "data", "dataset_B", "metadata.csv")
GT_DIR = os.path.join(RESTUDY, "data", "dataset_B")
VENV_PY = os.environ.get("YOLO_PYTHON", sys.executable)
MAX_EDGE = 1280
PRICE_IN, PRICE_OUT = 0.5, 2.0  # CNY per million input/output tokens.
CLASSES = ["overflow", "no_overflow", "waste"]
NO_ANOMALY_BASIS = "No overflowing waste bins or waste accumulation were detected in this run; regulatory retrieval was not performed."
SECTIONS = ["Scene description", "Regulatory basis", "Severity", "Impact assessment", "Potential causes",
            "Long-term effects", "Recommendations", "Priority"]
ABS_CSV = common.CLAUSE_CSV


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _img_to_datauri(path, max_edge=MAX_EDGE, quality=90):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def data_uri(image_id):
    return _img_to_datauri(os.path.join(IMG_DIR, image_id + ".jpg"))


def annotated_uri(image_id, annot_dir):
    return _img_to_datauri(os.path.join(annot_dir, image_id + ".jpg"))


def thumb_data_uri(image_id, maxw=460):
    return _img_to_datauri(os.path.join(IMG_DIR, image_id + ".jpg"),
                           max_edge=maxw, quality=82)


def save_thumb(image_id, out_path, maxw=460, quality=82):
    """Save a JPEG thumbnail for display in the report."""
    im = Image.open(os.path.join(IMG_DIR, image_id + ".jpg")).convert("RGB")
    im.thumbnail((maxw, maxw))
    im.save(out_path, format="JPEG", quality=quality)
    return out_path


def chat(messages, temperature=0.2, max_tokens=2000, retries=CHAT_RETRIES, timeout=180):
    """Call the hosted model and return (text, usage). Duration includes retries and backoff."""
    common.require_api_key()
    payload = {"model": VL_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    body = json.dumps(payload).encode()
    last_err = None
    call_start = time.perf_counter()
    for attempt in range(retries):
        req = urllib.request.Request(
            common.API_BASE + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + os.environ["SF_KEY"],
                     "Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            ok_dt = time.perf_counter() - t0
            u = d.get("usage", {})
            return d["choices"][0]["message"]["content"], {
                "seconds": round(time.perf_counter() - call_start, 3),
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "retries": attempt,
                "successful_seconds": round(ok_dt, 3)}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            last_err = e
            log("    HTTP %d (attempt %d), server response: %s"
                % (e.code, attempt + 1, detail))
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            last_err = e
            log("    Request failed (attempt %d): %s" % (attempt + 1, e))
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("Chat request failed after %d attempts: %s" % (retries, last_err))


def load_meta(site):
    rows = list(csv.DictReader(open(META_CSV, encoding="utf-8-sig")))
    return [r for r in rows if r["image_id"].startswith(site)]


def parse_counts(obj):
    out = {k: 0 for k in CLASSES}
    for o in obj.get("objects", []):
        if o.get("class") in out:
            out[o["class"]] = int(o.get("count", 0))
    return out


class PerceptionError(Exception):
    pass


def validate_perception(text, image_id):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise PerceptionError("No JSON object found")
    obj = json.loads(m.group(0))
    if obj.get("image_id") != image_id + ".jpg":
        raise PerceptionError("Image identifier mismatch: %s" % obj.get("image_id"))
    objs = obj.get("objects")
    if not isinstance(objs, list) or len(objs) != 3:
        raise PerceptionError("Expected exactly three objects")
    seen, out = set(), {}
    for o in objs:
        cls = o.get("class")
        if cls not in CLASSES or cls in seen:
            raise PerceptionError("Invalid or duplicate class label: %s" % cls)
        cnt = o.get("count")
        if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 0:
            raise PerceptionError("Invalid count: %s" % cnt)
        seen.add(cls)
        out[cls] = cnt
    return out


def field_data_text(meta, counts_map, total):
    lines = ["Observation block: %s" % meta[0]["image_id"].rsplit("-", 1)[0],
             "Collection time: %s" % meta[0]["capture_time"],
             "Setting: %s" % meta[0]["setting"],
             "Location: %s" % meta[0]["location"],
             "Weather: %s" % meta[0]["weather"],
             "Per-image information:"]
    for r in meta:
        c = counts_map[r["image_id"]]
        lines.append("  %s; coordinates %s; overflowing waste bins %d, non-overflowing waste bins %d, waste accumulations %d"
                     % (r["image_id"], r["gis_coordinates"], c["overflow"],
                        c["no_overflow"], c["waste"]))
    lines.append("Counts summed across images: overflowing waste bins %d, non-overflowing waste bins %d, waste accumulations %d"
                 % (total["overflow"], total["no_overflow"], total["waste"]))
    return "\n".join(lines)


def align_prompt(meta, counts_map, total, annot_dir=None):
    sent = [r for r in meta if sum(counts_map[r["image_id"]].values()) > 0]
    text = (prompts.ALIGN
            .replace("{image_ref}", "(%d attached images, identified by image ID)" % len(sent))
            .replace("{field_data}", field_data_text(meta, counts_map, total)))
    content = [{"type": "text", "text": text}]
    for r in sent:
        content.append({"type": "text", "text": "Image identifier: %s" % r["image_id"]})
        uri = (annotated_uri(r["image_id"], annot_dir)
               if annot_dir else data_uri(r["image_id"]))
        content.append({"type": "image_url", "image_url": {"url": uri}})
    omitted = [r["image_id"] for r in meta if r not in sent]
    if omitted:
        content.append({"type": "text",
                        "text": "No targets were detected in the unattached images (%s). Do not describe their contents."
                        % ", ".join(omitted)})
    return [{"role": "user", "content": content}]


def parse_align(text):
    """Parse English alignment headings and anomaly-specific retrieval descriptions."""
    clean = text.replace("**", "")
    m = re.search(r"Scene description\s*[:：](.*?)(?:Retrieval descriptions?\s*[:：]|$)", clean, re.S | re.I)
    scene = m.group(1).strip() if m else ""
    retr = {}
    rm = re.search(r"Retrieval descriptions?\s*[:：](.*)$", clean, re.S | re.I)
    if rm:
        for line in rm.group(1).splitlines():
            line = line.strip().lstrip("-• ").strip()
            for key, label in (("overflow", r"overflow(?:ing)?(?: waste bins?)?"),
                               ("waste", r"waste(?: accumulation)?")):
                match = re.match(label + r"\s*[:：]\s*(.+)", line, re.I)
                if match:
                    retr[key] = match.group(1).strip()
    return scene, retr


def _retrieve_retry(q, clauses, doc_emb, retries=3):
    """Run one query with backoff; raise an error when retries are exhausted."""
    last_err = None
    for attempt in range(retries):
        try:
            return fp.retrieve(q, 60, clauses, doc_emb)
        except Exception as e:
            last_err = e
            log("[RAG] Query failed (attempt %d/%d): %s" % (attempt + 1, retries, e))
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("RAG query failed after %d attempts: %s" % (retries, last_err))


def conditional_rag(total, align_retr, location):
    """Retrieve conditionally on perceived counts; return evidence, evidence text, logs, and skip status.

    Return only when all active queries succeed. Exhausted retries raise an error, preventing diagnosis with partial evidence.
    """
    active = [t for t in ("overflow", "waste") if total[t] > 0]
    if not active:
        return [], [], [], True
    clauses = common.load_clauses()
    codes = [c["code"] for c in clauses]
    doc_emb = fp.build_doc_embeddings(clauses)
    evidence, seen, logs = [], set(), []
    class_names = {"overflow": "overflowing waste bin", "waste": "waste accumulation"}
    for typ in active:
        desc = align_retr.get(typ)
        if desc:
            q, src = desc + " Retrieve applicable provisions.", "align"
        else:
            q, src = "%s: %s. Retrieve applicable provisions." % (location, class_names[typ]), "fallback"
        r = _retrieve_retry(q, clauses, doc_emb)
        kept = [codes[i] for i in r["kept"]]
        logs.append({"type": typ, "query": q, "query_source": src, "top5": kept})
        for i in r["kept"]:
            if codes[i] not in seen:
                seen.add(codes[i])
                evidence.append(clauses[i])
    evtext = ["%s, %s (clause ID %s): %s"
              % (c["name"], c["art"], c["code"], c["body"])
              for c in evidence]
    return evidence, evtext, logs, False


def diagnose_prompt(meta, counts_map, total, scene, with_rag, clauses_text):
    fd = field_data_text(meta, counts_map, total)
    if with_rag:
        text = prompts.DIAG_RAG.format(
            field_data=fd, scene=scene, clauses=clauses_text)
    else:
        text = prompts.DIAG_NO_RAG.format(field_data=fd, scene=scene)
    return [{"role": "user", "content": text}]


def parse_report_blocks(text):
    """Read the prompted two-column Markdown table, or labeled prose sections."""
    canonical = {name.lower(): name for name in SECTIONS}
    out = {}
    active = None
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = re.split(r"(?<!\\)\|", line.strip().strip("|"), maxsplit=1)
            if len(cells) == 2:
                label = re.sub(r"^[0-9]+[.)]\s*", "", cells[0].replace("**", "").strip()).strip()
                active = canonical.get(label.lower())
                if active:
                    out[active] = cells[1].strip()
                continue
        if active and line.strip() and not line.strip().startswith("#"):
            out[active] += "<br>" + line.strip()
    if out:
        return out
    pos = []
    for heading in SECTIONS:
        pattern = (r"(?m)^\s*(?:#+\s*)?\*{0,2}(?:\d+[.)]\s*)?" +
                   re.escape(heading) + r"\*{0,2}\s*[:：]?\s*")
        match = re.search(pattern, text, re.I)
        if match:
            pos.append((match.start(), match.end(), heading))
    pos.sort()
    for index, (_, start, heading) in enumerate(pos):
        end = pos[index + 1][0] if index + 1 < len(pos) else len(text)
        out[heading] = text[start:end].strip().replace("\n", "<br>")
    return out


def parse_sections(text):
    found = {}
    for i, name in enumerate(SECTIONS):
        pat = re.compile(r"%d[\.、\s]\s*%s[：:]?\s*(.*?)(?=\d[\.、\s]|$)"
                         % (i + 2, name), re.S)
        m = pat.search(text)
        if m:
            found[name] = m.group(1).strip()
    return found


_CITE = None


def _cite_maps():
    global _CITE
    if _CITE is not None:
        return _CITE
    with open(ABS_CSV, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        prev = rd.line_num
        code_line = {}
        for row in rd:
            code_line[row[3]] = prev + 1
            prev = rd.line_num
    _CITE = code_line
    return _CITE


def photo_cell(ids, photos_dir):
    """Embed base64 image thumbnails in Markdown and save JPEG thumbnail files."""
    os.makedirs(photos_dir, exist_ok=True)
    parts = []
    for i in ids:
        save_thumb(i, os.path.join(photos_dir, i + ".jpg"))
        parts.append("%s<br>![%s](%s)" % (i, i, thumb_data_uri(i)))
    return "<br><br>".join(parts)


def _link_rag(text):
    return re.sub(r"D-\d{2}-C\d{3}", lambda m: "[%s](../../../../../data/knowledge_base/clause_catalog.csv)" % m.group(0), text)


def _bullets_scene(text, site):
    items = [re.sub(r"[-—\s]+$", "", p).strip(" ；;")
             for p in re.split(r"(?=%s-\d{2})" % re.escape(site), text)]
    return "<br>".join("• " + p for p in items if p)


def _bullets_basis(text):
    ext_block = ""
    if "<br>" in text:
        main, rest = text.split("<br>", 1)
        ext_block = "<br><br>" + rest.strip()
    else:
        main = text
    main = re.sub(r"[-—]{2,}\s*$", "", main).strip()
    if re.search(r"(^|\s)-\s+", main):
        items = [p.strip() for p in re.split(r"\s*-\s+", main) if p.strip()]
    else:
        items = [main] if main else []
    return "<br>".join("• " + p for p in items) + ext_block


def render_report(site, meta, total, blocks, ids, with_rag, photos_dir,
                  counts_map=None, basis_override=None):
    first = meta[0]
    lon, lat = first["gis_coordinates"].split(",")
    blocks = dict(blocks)
    if basis_override:
        blocks["Regulatory basis"] = basis_override
    else:
        basis = blocks.get("Regulatory basis", "")
        if with_rag:
            basis = _link_rag(basis)
        blocks["Regulatory basis"] = _bullets_basis(basis)
    blocks["Scene description"] = _bullets_scene(blocks.get("Scene description", ""), site)
    anomaly_photos = 0
    if counts_map:
        anomaly_photos = sum(1 for i in ids
                             if counts_map[i]["overflow"] > 0
                             or counts_map[i]["waste"] > 0)
    fixed = [
        ("Collection time", first["capture_time"]),
        ("Weather", first["weather"]),
        ("Location", "%s (%s)" % (first["location"], first["setting"])),
        ("GIS coordinates", "Longitude %s, latitude %s" % (lon.strip(), lat.strip())),
        ("Target classes and counts", "Counts summed across images: overflowing waste bins %d, non-overflowing waste bins %d, waste accumulations %d"
         % (total["overflow"], total["no_overflow"], total["waste"])),
        ("Images containing anomalies", anomaly_photos),
        ("Site images", photo_cell(ids, photos_dir)),
    ]
    out = ["# %s\n" % site, "| Category | Details |", "| --- | --- |"]
    for k, v in fixed + [(n, blocks.get(n, "")) for n in SECTIONS]:
        if k != "Site images":
            v = str(v).replace("\n", " ").replace("|", "／")
            if k not in ("Scene description", "Regulatory basis"):
                v = re.sub(r"[-—\s]+$", "", v).strip()
        out.append("| %s | %s |" % (k, v))
    return "\n".join(out)


def _empty_usage():
    return {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_est": 0.0}


def _accum(agg, u):
    agg["seconds"] += u["seconds"]
    agg["prompt_tokens"] += u["prompt_tokens"]
    agg["completion_tokens"] += u["completion_tokens"]
    cost = ((u["prompt_tokens"] * PRICE_IN +
             u["completion_tokens"] * PRICE_OUT) / 1e6)
    agg["cost_est"] += cost


def _finalize_usage(agg):
    return {
        "seconds": round(agg["seconds"], 3),
        "total_tokens": agg["prompt_tokens"] + agg["completion_tokens"],
        "cost_est": round(agg["cost_est"], 5),
    }


def run_group(site, letter, backend, with_rag, meta, ids, out_dir):
    raw_dir = os.path.join(out_dir, "raw")
    rep_dir = os.path.join(out_dir, "reports")
    photos_dir = os.path.join(out_dir, "photos")
    annot_dir = (os.path.join(out_dir, "annotated", letter)
                 if backend == "yolo" else None)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    if annot_dir:
        os.makedirs(annot_dir, exist_ok=True)
    stage_rows = []

    log("[%s] %s perception started" % (letter, "MLLM" if backend == "mllm" else "YOLO"))
    perc_usage = _empty_usage()
    counts_map, total = {}, {k: 0 for k in CLASSES}
    if backend == "mllm":
        for n, i in enumerate(ids, 1):
            pf = os.path.join(raw_dir, "%s_percept_%s.json" % (letter, i))
            if os.path.exists(pf):
                log("    Perception %d/%d %s (cached)" % (n, len(ids), i))
                pj = json.load(open(pf, encoding="utf-8"))
                c = pj["counts"]
                if "usage" in pj:
                    _accum(perc_usage, pj["usage"])
            else:
                log("    Perception %d/%d %s (requesting...)" % (n, len(ids), i))
                ptext = (prompts.PERCEPT
                         .replace("{image_ref}", "(attached image)")
                         .replace("{image_id}", i + ".jpg")
                         .replace("{label_rule}", prompts.LABEL_RULE))
                c, last_err = None, None
                for attempt in range(3):
                    text, u = chat([{"role": "user", "content": [
                        {"type": "text", "text": ptext},
                        {"type": "image_url",
                         "image_url": {"url": data_uri(i)}},
                    ]}])
                    _accum(perc_usage, u)
                    try:
                        c = validate_perception(text, i)
                        break
                    except Exception as e:
                        last_err = e
                        log("    Validation failed (attempt %d): %s" % (attempt + 1, e))
                if c is None:
                    raise RuntimeError("Perception failed for %s: %s" % (i, last_err))
                json.dump({"image_id": i, "raw": text, "counts": c,
                           "usage": u},
                          open(pf, "w"), ensure_ascii=False, indent=2)
            counts_map[i] = c
            for k in CLASSES:
                total[k] += c[k]
    else:
        ycmd = [VENV_PY, os.path.join(HERE, "yolo_perceive.py"),
                IMG_DIR, "--annotated-dir", annot_dir] + ids
        y0 = time.perf_counter()
        yout = subprocess.run(ycmd, capture_output=True, text=True, check=True)
        ywall = time.perf_counter() - y0
        ydata = json.loads(yout.stdout)
        json.dump(ydata, open(os.path.join(raw_dir, "%s_yolo.json" % letter),
                              "w"), ensure_ascii=False, indent=2)
        for k, d in enumerate(ydata["per_image"]):
            i = d["image_id"][:-4]
            c = parse_counts(d)
            counts_map[i] = c
            for t in CLASSES:
                total[t] += c[t]
        perc_usage["seconds"] = round(ywall, 3)
    stage_rows.append(("Problem perception", _finalize_usage(perc_usage)))
    log("[%s] Perception complete: overflowing bins %d, non-overflowing bins %d, waste accumulations %d"
        % (letter, total["overflow"], total["no_overflow"], total["waste"]))

    log("[%s] Contextual alignment started" % letter)
    af = os.path.join(raw_dir, "%s_align.json" % letter)
    align_u = _empty_usage()
    if os.path.exists(af):
        aj = json.load(open(af, encoding="utf-8"))
        scene, retr = aj["scene"], aj["retrieval"]
        if "usage" in aj:
            _accum(align_u, aj["usage"])
    else:
        atext, u = chat(
            align_prompt(meta, counts_map, total,
                         annot_dir if backend == "yolo" else None),
            max_tokens=1500)
        _accum(align_u, u)
        scene, retr = parse_align(atext)
        json.dump({"raw": atext, "scene": scene, "retrieval": retr,
                   "usage": u,
                   "sent_images": [r["image_id"] for r in meta
                                   if sum(counts_map[r["image_id"]].values()) > 0]},
                  open(af, "w"), ensure_ascii=False, indent=2)
    stage_rows.append(("Contextual alignment", _finalize_usage(align_u)))

    skipped = False
    if with_rag:
        log("[%s] Conditional regulatory retrieval started" % letter)
        rgf = os.path.join(raw_dir, "%s_rag.json" % letter)
        rag_u = _empty_usage()
        if os.path.exists(rgf):
            rj = json.load(open(rgf, encoding="utf-8"))
            ev, rlogs = rj["evidence"], rj["queries"]
            skipped = rj.get("skipped", False)
            if "usage" in rj:
                _accum(rag_u, rj["usage"])
        else:
            r0 = time.perf_counter()
            ev, evtext, rlogs, skipped = conditional_rag(
                total, retr, meta[0]["location"])
            ru = {"seconds": round(time.perf_counter() - r0, 3),
                  "prompt_tokens": 0, "completion_tokens": 0}
            json.dump({"queries": rlogs, "evidence": ev, "skipped": skipped,
                       "usage": ru},
                      open(rgf, "w"), ensure_ascii=False, indent=2)
            _accum(rag_u, ru)
        ctext = "\n".join(
            ["%s, %s (clause ID %s): %s"
             % (c["name"], c["art"], c["code"], c["body"]) for c in ev])
        stage_rows.append(("RAG", _finalize_usage(rag_u)))
        log("[%s] RAG complete: %s"
            % (letter, "no anomalies; retrieval skipped" if skipped else
               "%d queries, %d unique clauses" % (len(rlogs), len(ev))))
    else:
        ctext = ""

    log("[%s] Diagnostic generation started" % letter)
    df_ = os.path.join(raw_dir, "%s_diagnose.json" % letter)
    if os.path.exists(df_):
        dj = json.load(open(df_, encoding="utf-8"))
        dtext = dj["raw"]
        diag_u = _empty_usage()
        if "usage" in dj:
            _accum(diag_u, dj["usage"])
    else:
        dtext, u = chat(diagnose_prompt(meta, counts_map, total, scene,
                                        with_rag, ctext), max_tokens=2500)
        diag_u = _empty_usage()
        _accum(diag_u, u)
        json.dump({"raw": dtext, "usage": u}, open(df_, "w"),
                  ensure_ascii=False, indent=2)
    stage_rows.append(("Diagnostic generation", _finalize_usage(diag_u)))

    total_u = _empty_usage()
    for _, u in stage_rows:
        total_u["seconds"] += u["seconds"]
        total_u["prompt_tokens"] += u["total_tokens"]
        total_u["cost_est"] += u["cost_est"]
    stage_rows.append(("Total", _finalize_usage(total_u)))

    blocks = parse_report_blocks(dtext) or parse_sections(dtext)
    report = render_report(
        site, meta, total, blocks, ids, with_rag, photos_dir,
        counts_map=counts_map,
        basis_override=NO_ANOMALY_BASIS if with_rag and skipped else None)
    open(os.path.join(rep_dir, "%s_%s.md" % (site, letter)), "w",
         encoding="utf-8").write(report)
    log("[%s] Report written: %s_%s.md" % (letter, site, letter))
    return stage_rows, counts_map


GROUP_LABEL = {
    "A": "A(MLLM)", "B": "B(MLLM+RAG)", "C": "C(YOLO+MLLM)",
    "D": "D(YOLO+MLLM+RAG)"}


def write_efficiency(path, done):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "stage", "seconds", "total_tokens", "cost_est"])
        for letter, rows in done:
            for n, (stage, u) in enumerate(rows):
                w.writerow([GROUP_LABEL[letter] if n == 0 else "", stage,
                            u["seconds"], u["total_tokens"], u["cost_est"]])


LABEL_TO_LETTER = {v: k for k, v in GROUP_LABEL.items()}


def read_efficiency(path):
    """Read existing efficiency records as {group: [(stage, usage), ...]} to preserve other groups when running a subset."""
    if not os.path.exists(path):
        return {}
    out, cur = {}, None
    for row in csv.reader(open(path, encoding="utf-8-sig")):
        if not row or row[0] == "group":
            continue
        if row[0]:
            cur = LABEL_TO_LETTER.get(row[0])
            if cur:
                out[cur] = []
        if cur and len(row) >= 5:
            out[cur].append((row[1], {"seconds": float(row[2]),
                                      "total_tokens": int(row[3]),
                                      "cost_est": float(row[4])}))
    return out


def _norm_label(label):
    return "overflow" if label == "overrflow" else label


def parse_gt(path):
    root = ET.parse(path).getroot()
    out = {}
    for img in root.findall("image"):
        name = img.get("name")[:-4]
        c = {k: 0 for k in CLASSES}
        for box in img.findall("box"):
            c[_norm_label(box.get("label"))] += 1
        out[name] = c
    return out


def write_accuracy(path, site, ids, group_maps):
    gt = parse_gt(os.path.join(GT_DIR, "reference_annotations.xml"))
    rows = []
    for letter, counts_map in group_maps:
        for i in ids:
            g, p = gt[i], counts_map[i]
            rows.append([GROUP_LABEL[letter], i,
                         g["overflow"], g["no_overflow"], g["waste"],
                         p["overflow"], p["no_overflow"], p["waste"]])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "image_id", "true_overflow", "true_no_overflow", "true_waste",
                    "predicted_overflow", "predicted_no_overflow", "predicted_waste"])
        w.writerows(rows)


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python code/workflow/run_site.py S01-A A[,B,C,D]\nB/D require SF_KEY and CLAUSE_CSV (complete 140-row corpus).")
        return
    site = sys.argv[1] if len(sys.argv) > 1 else SITE
    group_arg = sys.argv[2] if len(sys.argv) > 2 else GROUPS
    wanted = None
    if group_arg:
        wanted = {g.strip().upper() for g in str(group_arg).split(",") if g.strip()}
    out_dir = os.path.join(RESTUDY, "results", OUT_ROOT, site)
    os.makedirs(out_dir, exist_ok=True)
    meta = load_meta(site)
    if not meta:
        raise SystemExit("Unknown observation block. Use a value listed in data/dataset_B/metadata.csv.")
    if wanted and not wanted.issubset({"A", "B", "C", "D"}):
        raise SystemExit("Groups must be A, B, C, D or a comma-separated subset.")
    if not wanted or wanted.intersection({"B", "D"}):
        common.load_clauses()
    ids = [r["image_id"] for r in meta]
    full_plan = [("A", "mllm", False), ("B", "mllm", True),
                 ("C", "yolo", False), ("D", "yolo", True)]
    plan = [p for p in full_plan if not wanted or p[0] in wanted]
    log("========== Observation block %s; groups %s; %d images =========="
        % (site, ", ".join(p[0] for p in plan) or "none", len(ids)))
    eff_path = os.path.join(out_dir, "efficiency_cost_details.csv")
    all_done = read_efficiency(eff_path)
    group_maps = []
    for letter, backend, with_rag in plan:
        log("---------- Group %s ----------" % letter)
        stage_rows, counts_map = run_group(site, letter, backend, with_rag,
                                           meta, ids, out_dir)
        all_done[letter] = stage_rows
        group_maps.append((letter, counts_map))
        ordered = [(L, all_done[L]) for L, _, _ in full_plan if L in all_done]
        write_efficiency(eff_path, ordered)
        log("Efficiency records updated: %s" % eff_path)
    if all(L in all_done for L, _, _ in full_plan):
        acc_path = os.path.join(out_dir, "perception_counts.csv")
        full_maps = [(L, _load_counts_for(L, out_dir)) for L, _, _ in full_plan]
        write_accuracy(acc_path, site, ids, full_maps)
        log("Perception count table saved: %s" % acc_path)
    else:
        log("Perception count table will be saved when all four groups are complete; efficiency records updated.")
    log("========== Observation block %s complete ==========" % site)
    log("Output directory: %s" % out_dir)


def _load_counts_for(letter, out_dir):
    """Read cached image counts for a group to assemble the completed four-group count table."""
    raw_dir = os.path.join(out_dir, "raw")
    cmap = {}
    if letter in ("A", "B"):
        for pf in os.listdir(raw_dir):
            m = re.match(r"%s_percept_(.+)\.json" % letter, pf)
            if m:
                d = json.load(open(os.path.join(raw_dir, pf), encoding="utf-8"))
                cmap[m.group(1)] = d["counts"]
    else:
        yf = os.path.join(raw_dir, "%s_yolo.json" % letter)
        for d in json.load(open(yf, encoding="utf-8"))["per_image"]:
            cmap[d["image_id"][:-4]] = parse_counts(d)
    return cmap


if __name__ == "__main__":
    max_resume = int(os.environ.get("AUTO_RESUME", "5"))
    for k in range(max_resume + 1):
        try:
            main()
            break
        except Exception as e:
            print("[Resume] Attempt %d failed: %s" % (k + 1, e), flush=True)
            if k == max_resume:
                raise
            time.sleep(5)
