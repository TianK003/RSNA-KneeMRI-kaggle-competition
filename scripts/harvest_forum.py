"""Harvest the RSNA Knee competition forum via global topic search (ListTopics by forum is 403 for us).

Many search queries, paginated; keep topics whose forum is the competition's; then fetch each topic's post + nested
comments and write artifacts/forum/<id>.md plus artifacts/forum/index.tsv.
"""
import html
import json
import os
import re
import sys
import time

from kaggle import api
from kagglesdk.discussions.types.discussions_api_service import (ApiGetTopicRequest, ApiListCommentsRequest,
                                                                 ApiListTopicsRequest)

OUT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/forum"
FORUM = "RSNA Knee Abnormality Detection"
QUERIES = ["rsna knee", "knee abnormality", "knee mri", "knee", "rsna knee single model", "knee labels", "knee coatnet",
           "knee pseudo label", "knee dinov2", "knee resnet", "knee efficientnet", "knee leaderboard", "knee gold",
           "knee solution", "knee ensemble", "knee teacher", "knee distillation", "knee synovitis", "knee efficiency",
           "knee inference", "knee augmentation", "knee resolution", "knee slices", "knee report", "knee llm",
           "knee cv lb", "knee private", "knee shake", "knee 0.95", "knee 0.94", "knee baseline", "knee qwen",
           "knee vlm", "knee mil", "knee attention", "knee series", "knee dicom", "knee laterality", "knee test",
           "knee teammates", "knee team", "knee noise", "knee soft labels", "knee radiologist", "knee rules",
           "knee vit", "knee optimizer", "knee learning rate", "knee convnext", "knee oai", "knee external data",
           "knee fine-tune", "knee swin"]


def retry(fn, *a):
    for k in range(8):
        try:
            return fn(*a)
        except Exception as e:
            if "429" in str(e) or "503" in str(e) or "500" in str(e):
                time.sleep(min(120, 5 * 2 ** k)); continue
            raise
    return fn(*a)


def strip(s):
    s = re.sub(r"<br\s*/?>", "\n", s or "")
    s = re.sub(r"</p>|</li>", "\n", s)
    s = re.sub(r"<li>", "- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def walk(c, depth, lines):
    lines.append(f"{'  ' * depth}- **{c.author_name}** ({c.post_date.isoformat()[:10] if c.post_date else ''}, "
                 f"{c.votes or 0} votes): " + strip(c.content).replace("\n", "\n" + "  " * (depth + 1)))
    for r in c.replies or []:
        walk(r, depth + 1, lines)


os.makedirs(OUT, exist_ok=True)
api.authenticate()
found = {}
with api.build_kaggle_client() as kc:
    cl = kc.discussions.discussion_api_client
    for q in QUERIES:
        token, pages = "", 0
        while pages < 10:
            req = ApiListTopicsRequest()
            req.search_query = q
            req.page_size = 100
            if token:
                req.page_token = token
            try:
                resp = retry(cl.list_topics, req)
            except Exception as e:
                print("query", q, "failed", type(e).__name__, str(e)[:100], flush=True)
                break
            new = 0
            for t in resp.topics or []:
                if t.forum_name == FORUM and t.id not in found:
                    found[t.id] = t
                    new += 1
            pages += 1
            token = resp.next_page_token
            if not token:
                break
        print(f"query {q!r}: {len(found)} topics so far", flush=True)
    json.dump({str(k): [v.votes or 0, v.comment_count or 0, v.post_date.isoformat()[:10] if v.post_date else "", v.author_name, v.title]
               for k, v in found.items()}, open(os.path.join(OUT, "found.json"), "w", encoding="utf-8"), indent=0)
    print("searched:", len(found), "topics", flush=True)
    rows = []
    for tid, t in sorted(found.items()):
        path = os.path.join(OUT, f"{tid}.md")
        if not os.path.exists(path):
            req = ApiGetTopicRequest()
            req.id = tid
            topic = retry(cl.get_topic, req).topic
            comments, token = [], ""
            while True:
                r = ApiListCommentsRequest()
                r.topic_id = tid
                r.page_size = 100
                if token:
                    r.page_token = token
                resp = retry(cl.list_comments, r)
                comments += resp.comments or []
                token = resp.next_page_token
                if not token:
                    break
            lines = [f"# {topic.title}", "",
                     f"id {tid} · {topic.author_name} · {topic.post_date.isoformat()[:10] if topic.post_date else ''} · "
                     f"{topic.votes or 0} votes · {topic.comment_count or 0} comments · https://www.kaggle.com{topic.url}", "",
                     strip(topic.content), "", "## Comments", ""]
            for c in comments:
                walk(c, 0, lines)
            open(path, "w", encoding="utf-8").write("\n".join(lines))
            time.sleep(1.5)
        rows.append((tid, t.votes or 0, t.comment_count or 0, t.post_date.isoformat()[:10] if t.post_date else "",
                     t.author_name, t.title))
with open(os.path.join(OUT, "index.tsv"), "w", encoding="utf-8") as f:
    f.write("id\tvotes\tcomments\tdate\tauthor\ttitle\n")
    for r in sorted(rows, key=lambda r: -r[1]):
        f.write("\t".join(str(x) for x in r) + "\n")
print("topics", len(rows))
