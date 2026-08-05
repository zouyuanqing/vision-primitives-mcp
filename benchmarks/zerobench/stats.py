# -*- coding: utf-8 -*-
"""ZeroBench 最终统计：主问题 + 子问题（去重）"""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def stats(path, name):
    try:
        r = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        print(name, '读取失败:', e)
        return
    # 按 id 去重（保留最后一条）
    seen = {}
    for x in r:
        seen[str(x['id'])] = x
    items = list(seen.values())
    total = len(items)
    numeric = [x for x in items if x['type'] == 'numeric']
    hits = [x for x in items if x['correct']]
    print('%s: %d 题 | numeric %d | 命中 %d | numeric命中率 %.1f%%' % (
        name, total, len(numeric), len(hits), 100 * len(hits) / max(1, len(numeric))))
    for h in hits:
        print('  ✓ %s | 答案=%r' % (h['id'], h['answer']))
    # 类型分布
    from collections import Counter
    print('  类型:', dict(Counter(x['type'] for x in items)))

stats(r'C:\Users\Adfhj\Desktop\OH-WorkSpace\zerobench\results_full.json', '主问题')
stats(r'C:\Users\Adfhj\Desktop\OH-WorkSpace\zerobench\results_sub.json', '子问题')
