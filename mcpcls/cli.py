"""命令行入口。

python -m mcpcls.cli train     --encoder tfidf       训练请求侧与工具侧分类头,建工具索引
python -m mcpcls.cli evaluate  --encoder tfidf       跑离线评估,输出 reports/eval_report.md
python -m mcpcls.cli route "给张三发封邮件"            用训练好的模型路由一条请求
python -m mcpcls.cli label --kind request --in x.jsonl --out y.jsonl   LLM 双路打标
python -m mcpcls.cli fetch-tools --config mcp.json --out data/tools_raw.jsonl  拉取真实 MCP 工具列表

接 Jev 兜底(需要 OPENROUTER_API_KEY):evaluate --live-jev、route "..." --jev-fallback
python -m mcpcls.cli compare --dataset gold --clients jev,haiku --need-versions v1,v2  离线对比
python -m mcpcls.cli realdata translate|label|split|candidates --live   真实数据流水线(T-006)
python -m mcpcls.cli realdata label --single --labelers codebuddy      单路打标 -> labeled_single.jsonl
python -m mcpcls.cli evaluate-real                                      用真实数据重训并出对照报告
"""

import argparse
import json
import pathlib
import pickle
import sys

from . import evaluate as ev
from . import realdata as rd
from .jev import CLIENT_NAMES as CLIENTS
from .realdata import AGREEMENT_DEFAULT, AGREEMENT_RULES, Stopped
from .strategy import STRATEGIES, TRIGGERS

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"


def cmd_train(a):
    """训练并保存路由。--catalog real(默认)用真实目录 + 双语索引 + 软过滤关;
    --catalog synth 逐字节沿用旧行为(手写 53 个工具)。

    真实目录这条链全部离线、不联网:工具/银标/中文描述/真实训练请求都已在 data/ 落盘。
    """
    # 检索编码器(T-019):命令行总能给出;直接构造 Namespace 的调用方没有这个字段时按 None
    retrieval = getattr(a, "retrieval_encoder", None)
    try:
        reranker = _reranker_from_args(a)
    except Exception as e:                       # RerankError:打一行原因,不打堆栈
        print(str(e), file=sys.stderr)
        return 2
    if a.catalog == "synth":
        tools = ev.load_jsonl(a.tools)
        reqs = ev.load_jsonl(a.requests)
        router = ev.build_system(a.encoder, tools, reqs, threshold=a.threshold, top_k=a.top_k,
                                 retrieval_encoder=retrieval, reranker=reranker)
    else:
        from . import tools_real as tr

        tools = ev.load_jsonl(a.tools_real)
        labeled = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        synth_tools = ev.load_jsonl(a.tools)
        # 请求头用合成训练集 + 真实训练集(T-011 的选法);工具头仍用合成 tools.jsonl
        reqs = ev.load_jsonl(a.requests) + ev.load_jsonl(a.real_train)
        router = tr.build_real_router(tools, labeled, zh_rows, reqs, synth_tools,
                                      encoder=a.encoder, threshold=a.threshold, top_k=a.top_k,
                                      retrieval_encoder=retrieval, reranker=reranker)
    router.catalog_name = a.catalog      # 记进 router.pkl,route 输出里显示
    MODELS.mkdir(exist_ok=True)
    router.head.save(MODELS / "request_head.pkl")
    router.catalog.head.save(MODELS / "tool_head.pkl")
    router.catalog.export_jsonl(MODELS / "tool_catalog.jsonl")
    with open(MODELS / "router.pkl", "wb") as f:
        pickle.dump(router, f)
    review = sum(r.needs_review for r in router.catalog.records)
    print(f"训练完成(目录 {a.catalog}):{len(reqs)} 条请求,{len(tools)} 个工具;"
          f"{review} 个工具需人工审核。模型在 {MODELS}")


def _variant(a, tools, train, gold, probe, drop_gold_only):
    """按一种词表口径跑一遍请求侧评估:drop_gold_only 决定是否去掉 13 个 gold-only 词。"""
    fn = ev.request_need_features
    if not a.need_rules:
        fn = None
    elif drop_gold_only:
        fn = ev.request_need_features_ablated
    head_opts = {"need_feature_fn": fn, "need_two_stage": a.two_stage}
    router = ev.build_system(a.encoder, tools, train, threshold=a.threshold, top_k=a.top_k, **head_opts)
    return {"router": router, "requests": ev.eval_requests(router, gold),
            "probe": ev.eval_probe(router, probe), "head_opts": head_opts}


def client_check(names, timeout, flag, model=None):
    """接云端的准入检查:没有 key 时返回退出码 2,打一行中文提示,不打堆栈。"""
    from . import jev

    for name in names:
        kw = {"timeout": timeout}
        if model:
            kw["model"] = model
        try:
            jev.make_client(name, **kw)
        except jev.JevUnavailable as e:
            print(f"未接入 {name}:{e}。设置环境变量 OPENROUTER_API_KEY(Jev 也可用 JEV_API_KEY)后重试,或去掉 --{flag}。",
                  file=sys.stderr)
            return 2
    return 0


def _csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def _reranker_from_args(a):
    """按命令行参数建语义重排器(T-025/T-026);--rerank 没传就是 None,行为与改前完全一样。"""
    name = getattr(a, "rerank", None)
    if not name:
        return None
    from .rerank import SemanticReranker

    return SemanticReranker(name, rerank_n=getattr(a, "rerank_n", 20),
                            fusion=getattr(a, "rerank_fusion", "reorder"),
                            alpha=getattr(a, "rerank_alpha", 0.5),
                            recall_m=getattr(a, "rerank_recall_m", 0))


def rerank_flags(sp):
    """train / route 共用的重排开关(T-025)与混合召回开关(T-026)。"""
    sp.add_argument("--rerank", default=None, metavar="模型名",
                    help="对 tfidf 召回的 Top-N 做语义重排,如 BAAI/bge-m3;"
                         "不传 = 纯 tfidf 召回,行为与以前一致")
    sp.add_argument("--rerank-n", type=int, default=20,
                    help="重排池大小:tfidf 先召回多少条再重排(默认 20)")
    sp.add_argument("--rerank-fusion", choices=["reorder", "linear", "rrf"], default="reorder",
                    help="reorder=纯语义重排 / linear=α·tfidf+(1-α)·语义 / rrf=倒数排名融合")
    sp.add_argument("--rerank-alpha", type=float, default=0.5,
                    help="linear 融合的 α(默认 0.5)")
    sp.add_argument("--rerank-recall-m", type=int, default=0, metavar="M",
                    help="混合召回(T-026):把语义 Top-M 并入召回集再融合重排(需同时给 "
                         "--rerank);补进来的工具 tfidf 分记 0。0 = 只重排 tfidf 召回,与 T-025 一致")


def _soft_thresholds(text):
    """逗号分隔的软 need 阈值(T-020):"None"/"null" 转成 None,其余转 float。"""
    return [None if x.lower() in ("none", "null") else float(x) for x in _csv(text)]


def cmd_compare(a):
    from . import jev
    from .compare import compare, render_compare_report

    clients, need_versions = _csv(a.clients), _csv(a.need_versions)
    bad = [c for c in clients if c not in CLIENTS] + [v for v in need_versions
                                                      if v not in jev.NEED_VERSIONS]
    if bad:
        print(f"不认识的参数值:{', '.join(bad)}。客户端可选 {', '.join(CLIENTS)},"
              f"说明版本可选 {', '.join(jev.NEED_VERSIONS)}。", file=sys.stderr)
        return 2
    if a.live:      # 先查 key,没有就别白跑一遍
        code = client_check(clients, a.jev_timeout, "live")
        if code:
            return code

    def make(name):
        # --codebuddy-model 只改 codebuddy 的模型;缓存名由客户端自己带(codebuddy@<模型>)
        kw = {"timeout": a.jev_timeout}
        if name == "codebuddy" and a.codebuddy_model:
            kw["model"] = a.codebuddy_model
        return jev.make_client(name, **kw)

    res = compare(a.dataset, clients, need_versions, live=a.live, threshold=a.threshold,
                  encoder=a.encoder, tools=ev.load_jsonl(a.tools), train=ev.load_jsonl(a.requests),
                  gold=ev.load_jsonl(a.gold), make=make,
                  need_feature_fn=(ev.request_need_features if a.keep_gold_only_words
                                   else ev.request_need_features_ablated))
    report = render_compare_report(res)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"compare_{a.dataset}.md").write_text(report, encoding="utf-8")
    with open(REPORTS / f"compare_{a.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(report)
    return 0


def run_live_jev(a, router, baseline, gold):
    """--live-jev:在金标上再跑两组对照,写 reports/eval_jev.md 与 .json。

    两种模式各用一个 client,否则第二组会把第一组的调用次数和费用一起算进去。
    """
    from . import jev

    # 不传 --jev-model 时交给 JevClient 决定(JEV_MODEL 环境变量或默认模型)
    fb_client = jev.JevClient(model=a.jev_model, timeout=a.jev_timeout)
    full_client = jev.JevClient(model=a.jev_model, timeout=a.jev_timeout)
    model = fb_client.model
    fallback = ev.eval_jev(router, gold, fb_client, mode="fallback")
    full = ev.eval_jev(router, gold, full_client, mode="full")
    report = ev.render_jev_report(baseline, fallback, full, model=model, threshold=a.threshold)
    (REPORTS / "eval_jev.md").write_text(report, encoding="utf-8")
    with open(REPORTS / "eval_jev.json", "w", encoding="utf-8") as f:
        json.dump({"jev_model": model, "timeout": a.jev_timeout, "baseline": baseline,
                   "fallback": fallback, "full": full}, f, ensure_ascii=False, indent=2)
    print(report)
    return 0


def parse_side(spec: str) -> tuple:
    """`目录[=标签]` -> (标签, 目录);没写标签时用目录名当标签。"""
    path, _, label = spec.partition("=")
    return (label or pathlib.Path(path).name or path, path)


def cmd_compare_labels(a):
    """标签口径对照(T-009 / T-010a):写 reports/label_v3_compare.md / .json。

    要比多套口径时用 `--extra 目录1=标签1,目录2=标签2`(逗号分隔),一套一个目录,
    每个目录里放同名的 labeled_agree.jsonl / review_queue.jsonl / real_train.jsonl /
    real_holdout.jsonl。前两套也可以写在 `--old` / `--new` 上(键仍是 old / new)。
    """
    from .label_compare import compare_sides, render

    tools = ev.load_jsonl(a.tools)
    train = ev.load_jsonl(a.requests)
    sides = [parse_side(a.old), parse_side(a.new)] + [parse_side(s) for s in _csv(a.extra)]
    res = compare_sides(sides, tools, train, encoder=a.encoder)
    report = render(res)
    pathlib.Path(a.report_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report_md).write_text(report, encoding="utf-8")
    with open(a.report_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(report)
    return 0


def cmd_evaluate(a):
    if a.live_jev:      # 先查 key,没有就别白跑一遍评估
        code = client_check(("jev",), a.jev_timeout, "live-jev", model=a.jev_model)
        if code:
            return code
    tools = ev.load_jsonl(a.tools)
    train = ev.load_jsonl(a.requests)
    gold = ev.load_jsonl(a.gold)
    adv = ev.load_jsonl(a.adversarial)
    probe = ev.load_jsonl(a.probe)
    if not a.need_rules:  # 关规则时两个口径相同,不需要对照
        variants = None
        main = _variant(a, tools, train, gold, probe, drop_gold_only=False)
    else:
        variants = {ev.VARIANT_KEPT: _variant(a, tools, train, gold, probe, drop_gold_only=False),
                    ev.VARIANT_DROPPED: _variant(a, tools, train, gold, probe, drop_gold_only=True)}
        main = variants[ev.VARIANT_KEPT if a.keep_gold_only_words else ev.VARIANT_DROPPED]
    router = main["router"]
    req, probe_res = main["requests"], main["probe"]
    cv = ev.cross_validate_requests(a.encoder, tools, train, folds=a.folds,
                                    random_state=a.cv_seed, **main["head_opts"])
    tcv = ev.eval_tools_by_server(a.encoder, tools, train)
    adv_res = ev.eval_tools_adversarial(router, adv)
    need_head = {"need_rules": a.need_rules, "two_stage": a.two_stage,
                 "drop_gold_only": not a.keep_gold_only_words}
    leak = ev.keyword_leak_check(train, gold)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eval_report.md").write_text(
        ev.render_report(req, cv, probe_res, tcv, adv_res, a.encoder, need_head, leak, variants),
        encoding="utf-8")
    with open(REPORTS / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"need_head": need_head, "keyword_leak_check": leak, "requests": req,
                   "requests_cv": cv, "requests_probe": probe_res, "tools_by_server": tcv,
                   "adversarial": adv_res,
                   "requests_variants": {k: {"requests": v["requests"], "probe": v["probe"]}
                                         for k, v in (variants or {}).items()}},
                  f, ensure_ascii=False, indent=2)
    print((REPORTS / "eval_report.md").read_text(encoding="utf-8"))
    if a.live_jev:
        return run_live_jev(a, router, req, gold)
    return 0


def cmd_route(a):
    with open(MODELS / "router.pkl", "rb") as f:
        router = pickle.load(f)
    try:
        router.head.check_taxonomy()   # taxonomy 改过之后旧模型不能用,重训:`cli train`
    except ValueError as e:
        print(f"模型与当前 taxonomy 不一致:{e}", file=sys.stderr)
        return 2
    try:
        reranker = _reranker_from_args(a)
    except Exception as e:                       # RerankError:打一行原因,不打堆栈
        print(str(e), file=sys.stderr)
        return 2
    if reranker is not None:                     # --rerank 传了就覆盖训练时存的重排器
        router.reranker = reranker
    if a.llm_fallback:
        from .labeler import LLMLabeler

        router.llm_fallback = LLMLabeler(model=a.model).as_fallback()
    # --jev-fallback 是 --fallback-client jev 的别名
    name = a.fallback_client or ("jev" if a.jev_fallback else None)
    if name:
        flag = "jev-fallback" if a.jev_fallback else "fallback-client"
        code = client_check((name,), a.jev_timeout, flag, model=a.jev_model)
        if code:
            return code
        from . import jev

        kw = {"timeout": a.jev_timeout}
        if a.jev_model:
            kw["model"] = a.jev_model
        router.llm_fallback = jev.jev_fallback(jev.make_client(name, **kw),
                                               need_version=a.need_version)
    router.fallback_strategy = a.fallback_strategy
    router.fallback_trigger = a.fallback_trigger
    res = router.route(a.text).to_dict()
    res["catalog"] = getattr(router, "catalog_name", "")   # 用的是哪个目录(real / synth)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_label(a):
    from .labeler import LLMLabeler

    lab = LLMLabeler(model=a.model, model_b=a.model_b)
    items = ev.load_jsonl(a.inp)
    agreed = queued = 0
    with open(a.out, "w", encoding="utf-8") as out, open(a.out + ".review.jsonl", "w", encoding="utf-8") as rev:
        for it in items:
            x = it["text"] if a.kind == "request" else it
            r = lab.dual_label(x, a.kind)
            if r["agreed"]:
                lab_ = r["label"]
                row = dict(it, domains=lab_["domains"], actions=lab_["actions"])
                if a.kind == "request":
                    row["need"] = lab_.get("need", "single")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                agreed += 1
            else:
                rev.write(json.dumps(dict(it, llm_a=r["a"], llm_b=r["b"]), ensure_ascii=False) + "\n")
                queued += 1
    print(f"一致入库 {agreed} 条,进人工队列 {queued} 条({a.out}.review.jsonl)")


def cmd_fetch(a):
    from .mcp_fetch import fetch_all

    print(f"共拉取 {fetch_all(a.config, a.out)} 个工具 -> {a.out}")


def local_predict(encoder, tools, train):
    """给候选表用的本地预测函数:f(texts) -> [(need, domains, actions)]。"""
    from . import evaluate as ev

    head = ev.train_request_head(ev.fit_encoder(encoder, tools, train), train)

    def predict(texts):
        return [(o.need, o.domains, o.actions) for o in head.predict(list(texts))]

    return predict


def cmd_realdata(a):
    """真实数据流水线的一步:translate / label / split / candidates。"""
    from . import realdata as rd

    cost = rd.CostBook(a.cost_file)

    # --codebuddy-model 只作用于 codebuddy 那一路 / 翻译客户端,并隔离缓存
    def _model_for(name):
        return a.codebuddy_model if name == "codebuddy" else None

    if a.step == "translate":
        pool = rd.load_jsonl(a.pool)
        client = rd.open_client(a.translator, timeout=a.jev_timeout, model=_model_for(a.translator))
        out = rd.translate(pool, client, a.translator, workers=a.workers, budget=a.budget_usd,
                           cost=cost, live=a.live)
        rd.write_jsonl(a.pool_zh, out["rows"])
        s = out["stats"]
        print(f"翻译完成 {len(out['rows'])} 条 -> {a.pool_zh};"
              f"调用 {s.asked} 次,费用 {s.cost:.6f} 美元,累计 {cost.total():.6f} 美元;"
              f"没翻成保留原文 {out['untranslated']} 条")
        if s.line():
            print(s.line())
        return 0

    if a.step == "label":
        pool_zh = rd.load_jsonl(a.pool_zh)
        names = _csv(a.labelers)
        if a.single and len(names) != 1:
            print("--single 只用一路打标:--labelers 里必须只给一个名字。", file=sys.stderr)
            return 2
        labelers = [(n, rd.open_client(n, timeout=a.jev_timeout, model=_model_for(n)))
                    for n in names]
        if a.single:
            out = rd.single_label(pool_zh, labelers[0], need_version=a.need_version,
                                  workers=a.workers, budget=a.budget_usd, cost=cost, live=a.live)
            rd.write_jsonl(a.single_out, out["rows"])
            s = out["stats"]
            print(f"单路打标完成:{out['labeler']} 标上 {len(out['rows'])} 条 -> {a.single_out};"
                  f"没标上 {out['unlabeled']} 条;累计费用 {cost.total():.6f} 美元")
            print(f"  {out['labeler']}:调用 {s.asked} 次,费用 {s.cost:.6f} 美元"
                  + (f";{s.line()}" if s.line() else ""))
            return 0
        out = rd.dual_label(pool_zh, labelers, need_version=a.need_version, workers=a.workers,
                            budget=a.budget_usd, cost=cost, live=a.live, agreement=a.agreement)
        rd.write_jsonl(a.agree, out["agreed"])
        rd.write_jsonl(a.review, out["review"])
        print(f"打标完成({a.agreement} 口径):一致 {len(out['agreed'])} 条 -> {a.agree};"
              f"待人工 {len(out['review'])} 条 -> {a.review};累计费用 {cost.total():.6f} 美元")
        print(f"  第一路({names[0]})缺答案:{out['primary_missing']} 条"
              "(只能靠单路打标补,详见 labeled_single)")
        for name, s in out["stats"]:
            print(f"  {name}:调用 {s.asked} 次,费用 {s.cost:.6f} 美元"
                  + (f";{s.line()}" if s.line() else ""))
        for key, g in (out.get("by_math_like") or {}).items():
            if not g["n"]:
                continue
            rate = "-" if g["agree_rate"] is None else f"{g['agree_rate'] * 100:.1f}%"
            need_dist = "、".join(f"{k} {v}" for k, v in sorted(g["need"].items())) or "空"
            print(f"  math_like={key}:{g['n']} 条,一致 {g['agreed']} 条({rate});"
                  f"need 分布 {need_dist}")
        return 0

    if a.step == "split":
        agreed = rd.load_jsonl(a.agree)
        train, hold = rd.split(agreed)
        rd.write_jsonl(a.real_train, train)
        rd.write_jsonl(a.holdout, hold)
        print(f"切分完成:训练集 {len(train)} 条 -> {a.real_train};"
              f"留出集 {len(hold)} 条 -> {a.holdout}")
        return 0

    review = rd.load_jsonl(a.review)
    hold = rd.load_jsonl(a.holdout)
    tools = ev.load_jsonl(a.tools)
    train = ev.load_jsonl(a.requests)
    rows = rd.candidates(review, hold, local_predict(a.encoder, tools, train), n=a.n_candidates)
    rd.write_gold_candidates(a.candidates, rows)
    (pathlib.Path(a.candidates).parent / "gold_candidates_README.md").write_text(
        rd.render_candidates_readme(), encoding="utf-8")
    print(f"候选表 {len(rows)} 行 -> {a.candidates}(说明见同目录 gold_candidates_README.md)")
    return 0


def cmd_tools_real(a):
    """真实工具目录(T-010b/T-011):label 联网打标,translate 翻译描述,
    evaluate / report-t011 只读已落盘的文件出报告。"""
    from . import tools_real as tr

    if a.step == "translate":
        if a.live:
            code = client_check((a.translator,), a.jev_timeout, "live")
            if code:
                return code
        tools = ev.load_jsonl(a.tools_real)
        client = rd.open_client_or_cache(a.translator, timeout=a.jev_timeout, live=a.live)
        out = rd.translate_tools(tools, client, a.translator, live=a.live, workers=a.workers,
                                 budget=a.budget_usd)
        rd.write_jsonl(a.tools_zh, out["rows"])
        s = out["stats"]
        print(f"翻译完成 {len(out['rows'])} 条 -> {a.tools_zh};"
              f"调用 {s.asked} 次,失败 {s.failed} 次,费用 {s.cost:.6f} 美元;"
              f"没翻成 {out['untranslated']} 条")
        if s.line():
            print(s.line())
        return 0

    if a.step == "label":
        tools = ev.load_jsonl(a.tools_real)
        clients = [(n, rd.open_client(n, timeout=a.jev_timeout)) for n in _csv(a.labelers)]
        out = tr.label_tools(tools, clients, live=a.live, workers=a.workers,
                             budget=a.budget_usd)
        rd.write_jsonl(a.labeled, out["rows"])
        stats = out["stats"]
        print(f"工具打标完成 {len(out['rows'])} 行 -> {a.labeled};"
              f"一致 {sum(1 for r in out['rows'] if r['agreed'])} 条")
        for name, s in stats:
            print(f"  {name}:调用 {s.asked} 次,费用 {s.cost:.6f} 美元"
                  + (f";{s.line()}" if s.line() else ""))
        missing = out["missing"]
        print(f"  缺答案:jev {missing.get('jev', 0)} 条,haiku {missing.get('haiku', 0)} 条;"
              f"本次合计 {out['cost_usd']:.6f} 美元")
        return 0

    if a.step == "report-t011":
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        res = tr.compute_t011(tools, rows, zh_rows, requests, synth_tools, synth_train,
                              real_train, encoder=a.encoder, folds=a.folds, seed=a.cv_seed)
        report = tr.render_t011(res)
        pathlib.Path(a.t011_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t011_report_md).write_text(report, encoding="utf-8")
        with open(a.t011_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t011_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t011_report_md};数据 -> {a.t011_report_json}")
        return 0

    if a.step == "e2e":
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        renc = ev.build_encoder(a.retrieval_encoder) if a.retrieval_encoder else None
        res = tr.e2e_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                          encoder=a.encoder, folds=a.folds, seed=a.cv_seed, top_k=a.top_k,
                          retrieval_encoder=renc)
        report = tr.render_e2e(res)
        pathlib.Path(a.e2e_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.e2e_report_md).write_text(report, encoding="utf-8")
        with open(a.e2e_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.e2e_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.e2e_report_md};数据 -> {a.e2e_report_json}")
        return 0

    if a.step == "dedupe-eval":
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        res = tr.dedupe_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                             encoder=a.encoder, folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t013(res)
        pathlib.Path(a.t013_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t013_report_md).write_text(report, encoding="utf-8")
        with open(a.t013_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t013_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t013_report_md};数据 -> {a.t013_report_json}")
        return 0

    if a.step == "encoder-eval":
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        if a.retrieval_encoder:
            # T-019:分类头固定 --encoder,检索编码器逐个换;索引都用双语
            specs = [(s, tr.t019_label(s)) for s in _csv(a.retrieval_encoder)]
            res = tr.split_encoder_eval(tools, rows, zh_rows, requests, synth_tools,
                                        synth_train, real_train, head_encoder=a.encoder,
                                        retrieval_encoders=specs, folds=a.folds,
                                        seed=a.cv_seed, top_k=a.top_k)
            report = tr.render_t019(res)
            pathlib.Path(a.t019_report_md).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(a.t019_report_md).write_text(report, encoding="utf-8")
            with open(a.t019_report_json, "w", encoding="utf-8") as f:
                json.dump(tr.t019_metrics_json(res), f, ensure_ascii=False, indent=2)
            print(report)
            print(f"报告 -> {a.t019_report_md};数据 -> {a.t019_report_json}")
            return 0
        res = tr.encoder_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                              folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t016(res)
        pathlib.Path(a.t016_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t016_report_md).write_text(report, encoding="utf-8")
        with open(a.t016_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t016_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t016_report_md};数据 -> {a.t016_report_json}")
        return 0

    if a.step == "soft-need-eval":
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        res = tr.soft_need_eval(tools, rows, zh_rows, requests, synth_tools, synth_train,
                                real_train, head_encoder=a.encoder,
                                retrieval_encoder=a.retrieval_encoder or tr.T020_RETRIEVAL_ENCODER,
                                thresholds=_soft_thresholds(a.soft_thresholds),
                                folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t020(res)
        pathlib.Path(a.t020_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t020_report_md).write_text(report, encoding="utf-8")
        with open(a.t020_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t020_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t020_report_md};数据 -> {a.t020_report_json}")
        return 0

    if a.step == "snapshot":
        # T-021:当前默认配置的端到端快照;只读已经落盘的 data/ 文件,不联网、不改默认值。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        res = tr.snapshot_eval(tools, rows, zh_rows, requests, synth_tools, synth_train,
                               encoder=a.encoder, folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_snapshot(res)
        pathlib.Path(a.snapshot_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.snapshot_report_md).write_text(report, encoding="utf-8")
        with open(a.snapshot_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.snapshot_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.snapshot_report_md};数据 -> {a.snapshot_report_json}")
        return 0

    if a.step == "reference-eval":
        # T-022:用 D-005 定稿参考集评估当前默认配置;只读已落盘文件,不联网、不改默认值。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        req_ref = rd.load_request_reference(a.ref_requests, a.pool_zh)
        risk_ref = tr.load_risk_reference(a.ref_risk)
        res = tr.reference_eval(tools, rows, zh_rows, synth_tools, synth_train, real_train,
                                req_ref, risk_ref, encoder=a.encoder, top_k=a.top_k)
        report = tr.render_reference_eval(res)
        pathlib.Path(a.ref_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.ref_report_md).write_text(report, encoding="utf-8")
        with open(a.ref_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.reference_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.ref_report_md};数据 -> {a.ref_report_json}")
        return 0

    if a.step == "report-t023":
        # T-023:补两类风险信号的报告;只读已落盘文件,不联网、不改默认值。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        risk_ref = tr.load_risk_reference(a.ref_risk)
        res = tr.compute_t023(tools, rows, risk_ref)
        report = tr.render_t023(res)
        pathlib.Path(a.t023_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t023_report_md).write_text(report, encoding="utf-8")
        with open(a.t023_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t023_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t023_report_md};数据 -> {a.t023_report_json}")
        return 0

    if a.step == "t024-report":
        # T-024:need 漏判诊断报告;只读已落盘文件,不联网、不改默认值。
        # 候选 C 用语义编码器(本地缓存),离线;没装 sentence-transformers 时只在导入时报错。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        ref = rd.load_request_reference(a.ref_requests, a.pool_zh)
        res = tr.t024_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                           ref, folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t024(res)
        pathlib.Path(a.t024_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t024_report_md).write_text(report, encoding="utf-8")
        with open(a.t024_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t024_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t024_report_md};数据 -> {a.t024_report_json}")
        return 0

    if a.step == "t025-report":
        # T-025:检索两段式(tfidf 召回 + 语义重排)报告;只读已落盘文件,不联网。
        # 语义模型只用本地 HF 缓存;没装 sentence-transformers / 模型不在缓存时报错退出。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        res = tr.t025_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                           folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t025(res)
        pathlib.Path(a.t025_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t025_report_md).write_text(report, encoding="utf-8")
        with open(a.t025_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t025_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t025_report_md};数据 -> {a.t025_report_json}")
        return 0

    if a.step == "t026-report":
        # T-026:混合召回(tfidf Top-50 ∪ 语义 Top-M)调查与报告;只读已落盘文件,不联网。
        # 语义模型只用本地 HF 缓存;对照行读 T-025 的 json,不重跑。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        res = tr.t026_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                           folds=a.folds, seed=a.cv_seed, top_k=a.top_k, t025_json=a.t025_report_json)
        report = tr.render_t026(res)
        pathlib.Path(a.t026_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t026_report_md).write_text(report, encoding="utf-8")
        with open(a.t026_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t026_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t026_report_md};数据 -> {a.t026_report_json}")
        return 0

    if a.step == "t027-report":
        # T-027:能力域预测的分类头 + 检索结果投票;只读已落盘文件,不联网、不加载语义模型。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        req_ref = rd.load_request_reference(a.ref_requests, a.pool_zh)
        res = tr.t027_eval(tools, rows, zh_rows, requests, synth_tools, synth_train, real_train,
                           req_ref=req_ref, folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t027(res)
        pathlib.Path(a.t027_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t027_report_md).write_text(report, encoding="utf-8")
        with open(a.t027_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t027_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t027_report_md};数据 -> {a.t027_report_json}")
        return 0

    if a.step == "t028-report":
        # T-028:把 T-027 的候选搬到请求级银标(real_train/real_holdout)上重评;离线,只读文件。
        tools = ev.load_jsonl(a.tools_real)
        rows = ev.load_jsonl(a.labeled)
        zh_rows = ev.load_jsonl(a.tools_zh)
        synth_tools = ev.load_jsonl(a.tools)
        synth_train = ev.load_jsonl(a.requests)
        real_train = ev.load_jsonl(a.real_train)
        real_holdout = ev.load_jsonl(a.real_holdout)
        req_ref = rd.load_request_reference(a.ref_requests, a.pool_zh)
        res = tr.t028_eval(tools, rows, zh_rows, synth_tools, synth_train, real_train,
                           real_holdout, req_ref, folds=a.folds, seed=a.cv_seed, top_k=a.top_k)
        report = tr.render_t028(res)
        pathlib.Path(a.t028_report_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.t028_report_md).write_text(report, encoding="utf-8")
        with open(a.t028_report_json, "w", encoding="utf-8") as f:
            json.dump(tr.t028_metrics_json(res), f, ensure_ascii=False, indent=2)
        print(report)
        print(f"报告 -> {a.t028_report_md};数据 -> {a.t028_report_json}")
        return 0

    rows = ev.load_jsonl(a.labeled)
    tools = ev.load_jsonl(a.tools_real)
    requests = tr.load_eval_requests(a.req_map, a.pool_zh, a.agree)
    synth_tools = ev.load_jsonl(a.tools)
    synth_train = ev.load_jsonl(a.requests)
    real_train = ev.load_jsonl(a.real_train)
    zh_by_id = None
    if a.bilingual:
        zh_rows = ev.load_jsonl(a.tools_zh)
        zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    res = tr.evaluate_all(rows, tools, requests, synth_tools, synth_train, real_train,
                          encoder=a.encoder, folds=a.folds, seed=a.cv_seed,
                          bilingual=a.bilingual, zh_by_id=zh_by_id,
                          grid_search=a.grid_search, tool_head_mode=a.tool_head)
    risk = res["risk"]
    picked = tr.audit_rows(rows, risk["risk_by_id"])
    n_audit = tr.write_audit_csv(a.audit_csv, picked, {t["tool_id"]: t for t in tools},
                                 risk["risk_by_id"], risk["gold_risk"])
    report = tr.render_report(res["stats"], risk, res, requests, tools)
    pathlib.Path(a.report_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report_md).write_text(report, encoding="utf-8")
    print(f"报告 -> {a.report_md};抽检表 {n_audit} 行 -> {a.audit_csv}")
    print(report)
    return 0


def cmd_cn_pilot(a):
    """T-017 小样本选型:按来源分层抽 pilot 样本,对每个候选国产模型跑 v3 题目。

    主指标 = 与 Jev 的 primary 双路一致率(Jev / Haiku 的答案读现有 v3 缓存,不重新问)。
    不加 --live 时只读候选模型缓存:调用 0 次,输出与上一次逐字节相同。
    """
    from . import tools_real as tr

    pool = ev.load_jsonl(a.pool_zh)
    ids_path = pathlib.Path(a.pilot_ids)
    if ids_path.exists():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        ids = payload["ids"]
        if payload.get("n") != a.n:
            print(f"已有 pilot 样本 {payload.get('n')} 条,与 --n {a.n} 不一致,按已有文件跑。")
    else:
        payload = tr.pilot_ids_payload(pool, n=a.n, seed=a.seed)
        ids = payload["ids"]
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"抽样 {len(ids)} 条 -> {ids_path}(来源 {payload['by_source']})")
    models = _csv(a.models) or list(tr.CN_MODELS)
    res = tr.cn_pilot(pool, ids, live=a.live, workers=a.workers, models=models)
    for r in res["rows"]:
        if not r["is_candidate"]:
            continue
        st = r["stats"]
        print(f"  {r['model']}:调用 {st['asked']} 次,本次失败 {st.get('failed', 0)} 次,"
              f"缺答案 {r['metrics']['missing']} 条"
              + (f";{st['line']}" if st.get("line") else "")
              + (f";提前停下:{st['stopped']}" if st.get("stopped") else ""))
    report = tr.render_cn_pilot(res)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(report)
    print(f"pilot 结果 -> {a.out_json}")
    return 0


def cmd_cn_report(a):
    """把 pilot 结果、compare-labels 报告、两版 e2e 拼成 `reports/eval_t017.md`(全离线)。"""
    from . import tools_real as tr

    def _read_text(path):
        p = pathlib.Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else f"(缺 {path})"

    def _read_json(path):
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    pilot = _read_json(a.pilot_json)
    report = tr.render_t017(pilot, _read_text(a.compare_md),
                            _read_json(a.e2e_base), _read_json(a.e2e_cn))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"报告 -> {a.out}")
    return 0


def cmd_evaluate_real(a):
    """用真实数据重训:A/B/C 三种训练集 × 合成 gold / 银标留出集;有单路数据再加 D。"""
    from . import realdata as rd

    tools = ev.load_jsonl(a.tools)
    synth = ev.load_jsonl(a.requests)
    gold = ev.load_jsonl(a.gold)
    real_train = rd.load_jsonl(a.real_train) if pathlib.Path(a.real_train).exists() else []
    holdout = rd.load_jsonl(a.holdout) if pathlib.Path(a.holdout).exists() else []
    labeled_single = rd.load_jsonl(a.labeled_single) if pathlib.Path(a.labeled_single).exists() \
        else []
    if not real_train and not holdout and not labeled_single:
        print(f"没有真实数据:{a.real_train} 与 {a.holdout} 都不存在。"
              f"先跑 `realdata label` 与 `realdata split`。", file=sys.stderr)
        return 2
    res = rd.evaluate_real(tools, synth, gold, real_train, holdout,
                           labeled_single=labeled_single, encoder=a.encoder,
                           folds=a.folds, random_state=a.cv_seed)
    report = rd.render_real_eval(res)
    pathlib.Path(a.report_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report_md).write_text(report, encoding="utf-8")
    with open(a.report_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(report)
    print(f"报告已写入 {a.report_md} 与 {a.report_json}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mcpcls")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--encoder", default="tfidf", help="tfidf | st:<模型名> | jev:<模块:函数>")
        sp.add_argument("--tools", default=str(DATA / "tools.jsonl"))
        sp.add_argument("--requests", default=str(DATA / "requests_train.jsonl"))
        sp.add_argument("--threshold", type=float, default=0.55)
        sp.add_argument("--top-k", type=int, default=5)

    sp = sub.add_parser("train"); common(sp)
    sp.add_argument("--catalog", choices=["real", "synth"], default="real",
                    help="real(默认)=真实目录 + 双语索引 + 软过滤关;synth=手写 53 个工具,逐字节沿用旧行为")
    sp.add_argument("--tools-real", default=str(DATA / "tools_real.jsonl"),
                    help="real 目录的工具文件")
    sp.add_argument("--labeled", default=str(DATA / "tools_real_labeled.jsonl"),
                    help="real 目录的银标(domains/actions)")
    sp.add_argument("--tools-zh", default=str(DATA / "tools_real_zh.jsonl"),
                    help="real 目录的中文描述(拼进检索索引)")
    sp.add_argument("--real-train", default=str(DATA / "v3-primary" / "real_train.jsonl"),
                    help="real 目录的请求头训练数据(与 --requests 合成训练集相加)")
    sp.add_argument("--retrieval-encoder", default=None,
                    help="检索索引与查询用这个编码器(如 st:BAAI/bge-m3);"
                         "默认 None=沿用分类头编码器 `--encoder`(T-019)")
    rerank_flags(sp)
    sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("compare-labels")
    sp.add_argument("--old", default=str(DATA), help="第一套口径的目录(默认 data)")
    sp.add_argument("--new", default=str(DATA / "v3"), help="第二套口径的目录(默认 data/v3)")
    sp.add_argument("--extra", default="",
                    help="逗号分隔的更多口径,每项写成 目录=标签(标签省了就用目录名)")
    sp.add_argument("--encoder", default="tfidf")
    sp.add_argument("--tools", default=str(DATA / "tools.jsonl"))
    sp.add_argument("--requests", default=str(DATA / "requests_train.jsonl"),
                    help="合成训练集,C 方案的一半")
    sp.add_argument("--report-md", default=str(REPORTS / "label_v3_compare.md"))
    sp.add_argument("--report-json", default=str(REPORTS / "label_v3_compare.json"))
    sp.set_defaults(fn=cmd_compare_labels)
    sp = sub.add_parser("evaluate"); common(sp)
    sp.add_argument("--gold", default=str(DATA / "requests_gold.jsonl"))
    sp.add_argument("--probe", default=str(DATA / "requests_probe.jsonl"))
    sp.add_argument("--adversarial", default=str(DATA / "tools_adversarial.jsonl"))
    sp.add_argument("--folds", type=int, default=5, help="请求侧交叉验证折数")
    sp.add_argument("--cv-seed", type=int, default=0, help="交叉验证随机种子(固定以复现)")
    sp.add_argument("--need-rules", dest="need_rules", action="store_true", default=True,
                    help="need 头叠加无外部对象规则特征(默认开)")
    sp.add_argument("--no-need-rules", dest="need_rules", action="store_false",
                    help="关掉上一条,用于对照")
    sp.add_argument("--two-stage", dest="two_stage", action="store_true", default=True,
                    help="need 先判要不要工具的两级判定(默认开)")
    sp.add_argument("--no-two-stage", dest="two_stage", action="store_false",
                    help="关掉上一条,用于对照")
    sp.add_argument("--keep-gold-only-words", action="store_true",
                    help="主口径保留 13 个训练集 0 次的词(默认主口径去掉它们,含词结果同表并列)")
    sp.add_argument("--live-jev", action="store_true",
                    help="额外跑两组 Jev 对照(兜底 / 全量),写 reports/eval_jev.md;需要 OPENROUTER_API_KEY")
    sp.add_argument("--jev-model", default=None, help=f"默认 {ev.DEFAULT_JEV_MODEL}")
    sp.add_argument("--jev-timeout", type=float, default=30.0, help="单次调用超时秒数(默认 30)")
    sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("compare"); common(sp)
    sp.add_argument("--gold", default=str(DATA / "requests_gold.jsonl"))
    sp.add_argument("--dataset", choices=["train", "gold"], default="gold",
                    help="gold 只报告;挑策略看 train(本地用 5 折折外预测)")
    sp.add_argument("--clients", default="jev,haiku", help=f"逗号分隔,可选 {CLIENTS}")
    sp.add_argument("--need-versions", default="v1,v2", help="逗号分隔的 need 说明版本")
    sp.add_argument("--live", action="store_true",
                    help="联网取答案并写入 .cache/answers/;不加则只读缓存,缺答案的条目退回本地结果")
    sp.add_argument("--codebuddy-model", default=None,
                    help="codebuddy 客户端用 --model 指定的模型(默认 None = CodeBuddy 默认模型;"
                         "缓存与默认模型分开:codebuddy@<模型>)")
    sp.add_argument("--keep-gold-only-words", action="store_true",
                    help="本地基线的 need 特征保留 13 个 gold-only 词(默认与 evaluate 主口径一致,去掉它们)")
    sp.add_argument("--jev-timeout", type=float, default=30.0, help="单次调用超时秒数(默认 30)")
    sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("route"); sp.add_argument("text")
    sp.add_argument("--llm-fallback", action="store_true")
    sp.add_argument("--jev-fallback", action="store_true",
                    help="低置信度时用 Jev 兜底(需要 key),等价于 --fallback-client jev")
    sp.add_argument("--fallback-client", choices=list(CLIENTS), default=None,
                    help="兜底用哪个云端客户端(需要 key)")
    sp.add_argument("--fallback-strategy", choices=list(STRATEGIES), default="replace",
                    help="local 恒用本地 / replace 整体替换 / need_only 只换 need / both_none 两边都 none 才算 none")
    sp.add_argument("--fallback-trigger", choices=list(TRIGGERS), default="low_conf",
                    help="low_conf 置信度低于阈值才问 / always 每条都问")
    # T-005 在 train 上选定的兜底组合:客户端 jev + need 说明 v2 + replace + low_conf
    sp.add_argument("--need-version", default="v2", help="need 那一题的说明版本(默认 v2,T-005 选定)")
    sp.add_argument("--jev-model", default=None); sp.add_argument("--jev-timeout", type=float, default=30.0)
    sp.add_argument("--model", default="claude-sonnet-5-5")
    rerank_flags(sp)
    sp.set_defaults(fn=cmd_route)
    sp = sub.add_parser("label"); sp.add_argument("--kind", choices=["request", "tool"], default="request")
    sp.add_argument("--in", dest="inp", required=True); sp.add_argument("--out", required=True)
    sp.add_argument("--model", default="claude-sonnet-5-5"); sp.add_argument("--model-b", default=None)
    sp.set_defaults(fn=cmd_label)
    sp = sub.add_parser("fetch-tools"); sp.add_argument("--config", required=True)
    sp.add_argument("--out", default=str(DATA / "tools_raw.jsonl")); sp.set_defaults(fn=cmd_fetch)

    # ---- 真实数据流水线(T-006)----
    sp = sub.add_parser("realdata")
    sp.add_argument("step", choices=["translate", "label", "split", "candidates"])
    common(sp)
    sp.add_argument("--pool", default=str(DATA / "pool.jsonl"))
    sp.add_argument("--pool-zh", default=str(DATA / "pool_zh.jsonl"))
    sp.add_argument("--agree", default=str(DATA / "labeled_agree.jsonl"))
    sp.add_argument("--review", default=str(DATA / "review_queue.jsonl"))
    sp.add_argument("--real-train", default=str(DATA / "real_train.jsonl"))
    sp.add_argument("--holdout", default=str(DATA / "real_holdout.jsonl"))
    sp.add_argument("--candidates", default=str(DATA / "gold_candidates.csv"))
    sp.add_argument("--cost-file", default=str(DATA / "realdata_cost.json"))
    sp.add_argument("--live", action="store_true", help="联网问客户端;不加则只读缓存")
    sp.add_argument("--translator", default="haiku", help="翻译用哪个客户端(走 make_client)")
    sp.add_argument("--labelers", default="jev,haiku", help="双路打标的两路,逗号分隔,第一个是基准路")
    sp.add_argument("--single", action="store_true",
                    help="label 只用 --labelers 里的第一路单路打标,写到 --single-out;"
                         "这批数据不进 labeled_agree / real_train / holdout")
    sp.add_argument("--single-out", default=str(DATA / "labeled_single.jsonl"),
                    help="单路打标结果的落盘位置")
    sp.add_argument("--codebuddy-model", default=None,
                    help="codebuddy 用 --model 指定的模型;缓存名变成 codebuddy@<模型>(文件名里 @ 写成 _at_)")
    sp.add_argument("--need-version", default="v2", help=f"need 说明版本(默认 v2,T-005 选定)")
    sp.add_argument("--agreement", default=AGREEMENT_DEFAULT, choices=list(AGREEMENT_RULES),
                    help="双路一致的判定口径:exact 要求能力域集合完全相同(旧行为);"
                         "primary 只要求第一路概率最高的那个域落在第二路的域里(默认);"
                         "primary_sym 再放宽:两路任一的主域落在对方域里就算一致")
    sp.add_argument("--budget-usd", type=float, default=2.0,
                    help="所有联网步骤共用的累计预算;超了就停,退出码 3")
    sp.add_argument("--workers", type=int, default=4, help="联网并发数")
    sp.add_argument("--n-candidates", type=int, default=200, help="金标候选表行数")
    sp.add_argument("--jev-timeout", type=float, default=30.0)
    sp.set_defaults(fn=cmd_realdata)

    # ---- 真实工具目录(T-010b/T-011)----
    sp = sub.add_parser("tools-real")
    sp.add_argument("step", choices=["translate", "label", "evaluate", "report-t011", "e2e",
                                     "dedupe-eval", "encoder-eval", "soft-need-eval", "snapshot",
                                     "reference-eval", "report-t023", "t024-report",
                                     "t025-report", "t026-report", "t027-report", "t028-report"])
    common(sp)
    sp.add_argument("--tools-real", default=str(DATA / "tools_real.jsonl"))
    sp.add_argument("--tools-zh", default=str(DATA / "tools_real_zh.jsonl"))
    sp.add_argument("--labeled", default=str(DATA / "tools_real_labeled.jsonl"))
    sp.add_argument("--req-map", default=str(DATA / "requests_real_tool_ids.jsonl"))
    sp.add_argument("--pool-zh", default=str(DATA / "pool_zh.jsonl"))
    sp.add_argument("--agree", default=str(DATA / "v3-primary" / "labeled_agree.jsonl"),
                    help="请求侧银标 need(按 need 分组时用)")
    sp.add_argument("--real-train", default=str(DATA / "v3-primary" / "real_train.jsonl"))
    sp.add_argument("--audit-csv", default=str(DATA / "tools_real_risk_audit.csv"))
    sp.add_argument("--report-md", default=str(REPORTS / "eval_t010.md"))
    sp.add_argument("--t011-report-md", default=str(REPORTS / "eval_t011.md"),
                    help="T-011 报告(report-t011)的落盘位置")
    sp.add_argument("--t011-report-json", default=str(REPORTS / "eval_t011.json"),
                    help="T-011 报告数字的机器可读版")
    sp.add_argument("--e2e-report-md", default=str(REPORTS / "eval_e2e.md"),
                    help="T-012 端到端评估(e2e)的落盘位置")
    sp.add_argument("--e2e-report-json", default=str(REPORTS / "eval_e2e.json"),
                    help="T-012 端到端评估数字的机器可读版")
    sp.add_argument("--t013-report-md", default=str(REPORTS / "eval_t013.md"),
                    help="T-013 去重评估(dedupe-eval)的落盘位置")
    sp.add_argument("--t013-report-json", default=str(REPORTS / "eval_t013.json"),
                    help="T-013 去重评估数字的机器可读版")
    sp.add_argument("--t016-report-md", default=str(REPORTS / "eval_t016.md"),
                    help="T-016 编码器评估(encoder-eval)的落盘位置")
    sp.add_argument("--t016-report-json", default=str(REPORTS / "eval_t016.json"),
                    help="T-016 编码器评估数字的机器可读版")
    sp.add_argument("--retrieval-encoder", default="",
                    help="检索索引与查询用哪个编码器(T-019):e2e 传单个 spec;"
                         "encoder-eval 传逗号分隔的多个 spec(给了就走分开配置评估并写 T-019 报告);"
                         "空 = 沿用分类头编码器")
    sp.add_argument("--t019-report-md", default=str(REPORTS / "eval_t019.md"),
                    help="T-019 分开配置评估的落盘位置")
    sp.add_argument("--t019-report-json", default=str(REPORTS / "eval_t019.json"),
                    help="T-019 分开配置评估数字的机器可读版")
    sp.add_argument("--t020-report-md", default=str(REPORTS / "eval_t020.md"),
                    help="T-020 软 need 评估(soft-need-eval)的落盘位置")
    sp.add_argument("--t020-report-json", default=str(REPORTS / "eval_t020.json"),
                    help="T-020 软 need 评估数字的机器可读版")
    sp.add_argument("--soft-thresholds", default="None,0.5,0.6,0.7,0.8,0.9,1.01",
                    help="T-020 软 need 阈值网格,逗号分隔;None 表示改前基线")
    sp.add_argument("--snapshot-report-md", default=str(REPORTS / "eval_current_default.md"),
                    help="T-021 当前默认配置快照(snapshot)的落盘位置")
    sp.add_argument("--snapshot-report-json", default=str(REPORTS / "eval_current_default.json"),
                    help="T-021 当前默认配置快照数字的机器可读版")
    sp.add_argument("--ref-requests", default=str(DATA / "gold_candidates_claude.csv"),
                    help="T-022 请求参考集 CSV(Claude 预标 + gu 抽查定稿,D-005)")
    sp.add_argument("--ref-risk", default=str(DATA / "tools_real_risk_audit_claude.csv"),
                    help="T-022 风险参考集 CSV(60 个工具,真值取 gu_risk 或回退 claude_risk)")
    sp.add_argument("--ref-report-md", default=str(REPORTS / "eval_reference.md"),
                    help="T-022 参考集评估(reference-eval)的落盘位置")
    sp.add_argument("--ref-report-json", default=str(REPORTS / "eval_reference.json"),
                    help="T-022 参考集评估数字的机器可读版")
    sp.add_argument("--t023-report-md", default=str(REPORTS / "eval_t023.md"),
                    help="T-023 风险信号报告(report-t023)的落盘位置")
    sp.add_argument("--t023-report-json", default=str(REPORTS / "eval_t023.json"),
                    help="T-023 风险信号报告数字的机器可读版")
    sp.add_argument("--t024-report-md", default=str(REPORTS / "eval_t024.md"),
                    help="T-024 need 漏判诊断报告(t024-report)的落盘位置")
    sp.add_argument("--t024-report-json", default=str(REPORTS / "eval_t024.json"),
                    help="T-024 need 漏判诊断报告数字的机器可读版")
    sp.add_argument("--t025-report-md", default=str(REPORTS / "eval_t025.md"),
                    help="T-025 检索两段式报告(t025-report)的落盘位置")
    sp.add_argument("--t025-report-json", default=str(REPORTS / "eval_t025.json"),
                    help="T-025 检索两段式报告数字的机器可读版;T-026 的对照行从这里读")
    sp.add_argument("--t026-report-md", default=str(REPORTS / "eval_t026.md"),
                    help="T-026 混合召回报告(t026-report)的落盘位置")
    sp.add_argument("--t026-report-json", default=str(REPORTS / "eval_t026.json"),
                    help="T-026 混合召回报告数字的机器可读版")
    sp.add_argument("--t027-report-md", default=str(REPORTS / "eval_t027.md"),
                    help="T-027 能力域投票报告(t027-report)的落盘位置")
    sp.add_argument("--t027-report-json", default=str(REPORTS / "eval_t027.json"),
                    help="T-027 能力域投票报告数字的机器可读版")
    sp.add_argument("--real-holdout", default=str(DATA / "v3-primary" / "real_holdout.jsonl"),
                    help="T-028 留出集(请求级银标 need≠none 用来只报一次)")
    sp.add_argument("--t028-report-md", default=str(REPORTS / "eval_t028.md"),
                    help="T-028 能力域投票重评报告(t028-report)的落盘位置")
    sp.add_argument("--t028-report-json", default=str(REPORTS / "eval_t028.json"),
                    help="T-028 能力域投票重评报告数字的机器可读版")
    sp.add_argument("--live", action="store_true", help="联网问客户端;不加则只读缓存")
    sp.add_argument("--translator", default="haiku", help="翻译用的客户端(默认 haiku)")
    sp.add_argument("--labelers", default="jev,haiku", help="打标的两路,逗号分隔,第一个是基准路")
    sp.add_argument("--bilingual", action="store_true", help="检索索引拼接中文描述(T-011)")
    sp.add_argument("--grid-search", action="store_true", help="T-011:在开发集上选软过滤参数")
    sp.add_argument("--tool-head", choices=["synth", "real", "both"], default="synth",
                    help="T-011:工具侧分类头训练数据(默认 synth)")
    sp.add_argument("--budget-usd", type=float, default=50.0,
                    help="这次联网步骤的预算上限(只在本次进程内累计,不落盘)")
    sp.add_argument("--workers", type=int, default=4, help="联网并发数")
    sp.add_argument("--folds", type=int, default=5, help="请求头折外预测的折数")
    sp.add_argument("--cv-seed", type=int, default=0, help="折外切分的固定种子")
    sp.add_argument("--jev-timeout", type=float, default=30.0)
    sp.set_defaults(fn=cmd_tools_real)

    sp = sub.add_parser("evaluate-real"); common(sp)
    sp.add_argument("--gold", default=str(DATA / "requests_gold.jsonl"))
    sp.add_argument("--real-train", default=str(DATA / "real_train.jsonl"))
    sp.add_argument("--holdout", default=str(DATA / "real_holdout.jsonl"))
    sp.add_argument("--labeled-single", default=str(DATA / "labeled_single.jsonl"),
                    help="单路打标的数据;存在时加一组 D(合成 + real_train + 单路),只在银标留出集上报")
    sp.add_argument("--folds", type=int, default=5, help="C 方案在 real_train 上的交叉验证折数")
    sp.add_argument("--cv-seed", type=int, default=0)
    sp.add_argument("--report-md", default=str(REPORTS / "eval_t006.md"),
                    help="换口径重跑时改这里,免得覆盖旧口径的报告")
    sp.add_argument("--report-json", default=str(REPORTS / "eval_t006.json"))
    sp.set_defaults(fn=cmd_evaluate_real)

    # ---- 国产第二路打标(T-017)----
    sp = sub.add_parser("cn-pilot")
    sp.add_argument("--pool-zh", default=str(DATA / "pool_zh.jsonl"))
    sp.add_argument("--pilot-ids", default=str(DATA / "pilot_cn_ids.json"),
                    help="pilot 样本 id 列表;不存在时按 --n/--seed 抽样并写到这里")
    sp.add_argument("--out-json", default=str(REPORTS / "eval_t017_pilot.json"))
    sp.add_argument("--models", default="",
                    help="逗号分隔的候选模型;不给则用 tools_real.CN_MODELS 里的 4 个")
    sp.add_argument("--n", type=int, default=210, help="pilot 样本条数")
    sp.add_argument("--seed", type=int, default=0, help="抽样随机种子(固定以复现)")
    sp.add_argument("--workers", type=int, default=8, help="联网并发数")
    sp.add_argument("--live", action="store_true", help="联网问候选模型;不加则只读缓存")
    sp.set_defaults(fn=cmd_cn_pilot)

    sp = sub.add_parser("cn-report")
    sp.add_argument("--pilot-json", default=str(REPORTS / "eval_t017_pilot.json"))
    sp.add_argument("--compare-md", default=str(REPORTS / "eval_t017_compare.md"))
    sp.add_argument("--e2e-base", default=str(REPORTS / "eval_t017_e2e_base.json"))
    sp.add_argument("--e2e-cn", default=str(REPORTS / "eval_t017_e2e_cn.json"))
    sp.add_argument("--out", default=str(REPORTS / "eval_t017.md"))
    sp.set_defaults(fn=cmd_cn_report)

    a = p.parse_args(argv)
    try:
        return a.fn(a) or 0
    except Stopped as e:      # 超预算 / 客户端不可用:打印一行原因,不打堆栈
        print(e.reason, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
