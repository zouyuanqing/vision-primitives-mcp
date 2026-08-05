# -*- coding: utf-8 -*-
"""后台全量跑批：zb_eval6 风格（MiMo 直答 pass@5 + zoom 补），按批推进，断点续跑
用法: python zb_full.py <start> <end> <batch>
"""
import io, sys, os, json, time, re, base64, urllib.request, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MIMO_BASE = 'https://api.xiaomimimo.com/v1'
MIMO_KEY = 'sk-c2uvfccumc9jiu6yq4otwn98q8n4hkenhnm921fom5ghw43d'

os.environ['VISION_API_BASE'] = 'https://api.xiaomimimo.com/v1'
os.environ['VISION_API_KEY'] = MIMO_KEY
os.environ['VISION_MODEL'] = 'mimo-v2.5'
os.environ['VISION_CACHE'] = '0'
os.environ['VISION_TIMEOUT_S'] = '300'

sys.path.insert(0, r'C:\Users\Adfhj\Desktop\OH-WorkSpace\vision-bridge-mcp')
import vision_primitives_mcp as vb
from PIL import Image

OUTDIR = r'C:\Users\Adfhj\Desktop\OH-WorkSpace\zerobench\imgs'
RESULT = r'C:\Users\Adfhj\Desktop\OH-WorkSpace\zerobench\results_full.json'
MAX_SIDE = 1280


def load_data():
    import pyarrow.parquet as pq
    t = pq.read_table(r'C:\Users\Adfhj\Desktop\OH-WorkSpace\zerobench\zerobench.parquet')
    rows = []
    for i in range(t.num_rows):
        media = t.column('media')[i].as_py()
        msgs = json.loads(t.column('messages')[i].as_py())
        rows.append({'id': t.column('id')[i].as_py(),
                     'bytes': media[0]['bytes'] if isinstance(media, list) else media['bytes'],
                     'question': msgs[0].get('question', ''),
                     'answer': t.column('answer')[i].as_py(),
                     'type': t.column('question_type')[i].as_py()})
    return rows


def prep_image(row, outdir, max_side=MAX_SIDE):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"q{row['id']}.png")
    if not os.path.exists(p):
        with open(p, 'wb') as f:
            f.write(row['bytes'])
        img = Image.open(p)
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            img.save(p)
        img.close()
    return p


def mimo_vlm(img_path, question, temperature=0.0, max_tokens=2000):
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = (f'{question}\n仔细分析图片，逐步推理，最后一行必须以 "FINAL ANSWER:" 开头给出最终答案'
              '（数字题只输出数字本身）。')
    body = {'model': 'mimo-v2.5', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}},
    ]}], 'max_tokens': max_tokens, 'temperature': temperature}
    req = urllib.request.Request(MIMO_BASE + '/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + MIMO_KEY, 'Content-Type': 'application/json'}, method='POST')
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    return (resp['choices'][0]['message']['content'] or '').strip()


def judge_numeric(pred, answer):
    try:
        ans_num = float(str(answer).replace('$', '').replace(',', '').strip())
    except ValueError:
        return False
    nums = re.findall(r'[-+]?\d*\.?\d+', pred.replace(',', ''))
    return any(abs(float(n) - ans_num) < 0.01 for n in nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start', type=int)
    ap.add_argument('end', type=int)
    ap.add_argument('--samples', type=int, default=5)
    args = ap.parse_args()

    rows = load_data()
    # 断点：跳过已有结果
    done = set()
    if os.path.exists(RESULT):
        try:
            for r in json.load(open(RESULT, encoding='utf-8')):
                done.add(str(r['id']))
        except Exception:
            pass

    results = []
    for i in range(args.start - 1, min(args.end, len(rows))):
        row = rows[i]
        if str(row['id']) in done:
            print('skip [%s]（已跑）' % row['id'], flush=True)
            continue
        img_path = prep_image(row, OUTDIR)
        print('=== [%s] %s | 答案: %r' % (row['id'], row['type'], row['answer']), flush=True)
        preds = []
        correct = None
        for s in range(args.samples):
            t0 = time.time()
            try:
                p = mimo_vlm(img_path, row['question'], 0.9 if s else 0.0)
                preds.append(p)
                print('  [%d] %.1fs ...%s' % (s + 1, time.time() - t0, p[-60:].replace(chr(10), ' ')), flush=True)
                if row['type'] == 'numeric' and judge_numeric(p, row['answer']):
                    correct = True
                    print('  >>> 命中!', flush=True)
                    break
            except Exception as e:
                print('  [%d FAIL] %s' % (s + 1, str(e)[:60]), flush=True)
        if correct is None and row['type'] == 'numeric':
            correct = False
        results.append({'id': row['id'], 'type': row['type'], 'answer': row['answer'],
                        'preds': preds, 'correct': correct})
        # 增量保存
        old = []
        if os.path.exists(RESULT):
            try:
                old = json.load(open(RESULT, encoding='utf-8'))
            except Exception:
                pass
        old = [r for r in old if r['id'] != row['id']] + results
        with open(RESULT, 'w', encoding='utf-8') as f:
            json.dump(old, f, ensure_ascii=False, indent=1)

    allr = []
    if os.path.exists(RESULT):
        allr = json.load(open(RESULT, encoding='utf-8'))
    ok = sum(1 for r in allr if r['correct'])
    total = sum(1 for r in allr if r['correct'] is not None)
    print()
    print('=== 累计: %d/%d (numeric) ===' % (ok, total))


if __name__ == '__main__':
    main()
