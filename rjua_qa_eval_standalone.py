# -*- coding: utf-8 -*-
"""
RJUA-QA 评测脚本 (自包含独立版)
================================
数据集: 仁济医院泌尿科 x 蚂蚁集团 | arXiv:2312.09785

特性:
  - 关键词匹配 (语义拆分 + 同义词表 + 夹字兜底)
  - LLM Judge (DeepSeek-V3 via SiliconFlow) 做语义兜底
  - 支持断点续跑
  - 可用 .env 文件统一配置

使用方法:
  1. 安装依赖: pip install requests openpyxl pandas python-dotenv
  2. 配置 .env (参见 .env.example)
  3. python rjua_qa_eval_standalone.py
"""

import os, sys, re, json, time, argparse
import hashlib, hmac, secrets
from datetime import datetime
from typing import List, Dict, Optional

# 尝试加载 dotenv (可选)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==================== 配置 ====================
class Config:
    BASE_URL = os.getenv("PLATFORM_URL", "http://192.168.10.107")
    USERNAME = os.getenv("PLATFORM_USERNAME", "")
    PASSWORD = os.getenv("PLATFORM_PASSWORD", "")
    KB_UUID  = os.getenv("KB_UUID", "")

    QA_XLSX   = os.getenv("QA_XLSX", "RJUA-仁济QA-50题评测集.xlsx")
    DOCS_TXT  = os.getenv("DOCS_TXT", "RJUA-仁济源文档-全部50篇.txt")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "rjua_results")

    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
    DELAY_BETWEEN   = int(os.getenv("DELAY_BETWEEN", "1"))
    RETRY_TIMES     = int(os.getenv("RETRY_TIMES", "2"))
    EFFORT          = os.getenv("EFFORT", "high")  # low/medium/high/deep
    PLATFORM_MODEL  = os.getenv("PLATFORM_MODEL", "")  # 空则用平台默认
    PASS_COVERAGE   = float(os.getenv("PASS_COVERAGE", "0.7"))
    JUDGE_ROUNDS    = int(os.getenv("JUDGE_ROUNDS", "3"))  # LLM Judge 多轮取中, 1=单次
    NEW_SESSION_PER_QUESTION = os.getenv("NEW_SESSION", "1") == "1"

    # LLM Judge (可选)
    LLM_KEY   = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
    LLM_URL   = os.getenv("LLM_URL", "https://api.siliconflow.cn/v1/chat/completions")
    USE_PLATFORM_JUDGE = os.getenv("USE_PLATFORM_JUDGE", "0") == "1"  # 1=用平台GPT做Judge


# ==================== KB 客户端 ====================
class KBClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.token = ""
        self.secret = ""
        self.session = None
        import requests
        self.s = requests.Session()

    def _post(self, path: str, body: dict, stream=False, timeout=None):
        body_str = json.dumps(body, ensure_ascii=False)
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.secret:
            h.update(self._sign("POST", path, body_str))
        url = self.cfg.BASE_URL + "/api" + path
        return self.s.post(url, data=body_str.encode("utf-8"), headers=h,
                           stream=stream, timeout=timeout or self.cfg.REQUEST_TIMEOUT)

    def login(self):
        pwd_hash = hashlib.sha256(self.cfg.PASSWORD.encode("utf-8")).hexdigest()
        r = self._post("/knowledge/user/login",
                       {"userName": self.cfg.USERNAME, "userPassword": pwd_hash})
        data = r.json()
        if data.get("returnCode") != 200:
            raise RuntimeError(f"登录失败: {data}")
        d = data["data"]
        self.token = d["access_token"]
        self.secret = d["signing_secret"]
        self.user_uuid = d["uuid"]
        print(f"  [登录成功] user={d.get('username')}")

    def _sign(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time()))
        nonce = secrets.token_hex(16)
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        sign_str = f"{method}\n{path}\n\n{body_hash}\n{ts}\n{nonce}"
        sign = hmac.new(self.secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
        return {"X-Timestamp": ts, "X-Nonce": nonce, "X-Sign": sign}

    def create_session(self) -> str:
        r = self._post("/knowledge/session",
                       {"user_uuid": self.user_uuid, "agent_uuid": self.cfg.KB_UUID})
        data = r.json()
        if "data" not in data:
            return ""
        return data["data"].get("session_id", "")

    def chat(self, sid: str, question: str) -> dict:
        body = {"question": question, "effort": self.cfg.EFFORT}
        if self.cfg.PLATFORM_MODEL:
            body["model"] = self.cfg.PLATFORM_MODEL
        r = self._post(f"/knowledge/session/{sid}/chat", body, stream=True,
                       timeout=self.cfg.REQUEST_TIMEOUT)

        chunks, attachments = [], []
        answer_parts = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            etype = ev.get("type", "")
            estage = ev.get("stage", "")
            edata = ev.get("data", "")

            # 最终答案 (type="content" + stage="answer")
            if etype == "content" and estage == "answer":
                if isinstance(edata, str):
                    answer_parts.append(edata)

            # 检索片段 (type="tool_result" + stage="executor")
            elif etype == "tool_result" and estage == "executor":
                if isinstance(edata, dict):
                    result_text = edata.get("result", "")
                    for m in re.finditer(r"\{\{chunk:([a-f0-9-]+)\}\}", result_text):
                        chunks.append({"chunk_id": m.group(1), "content": result_text})

            # 引用文档
            elif etype == "answer_attachments" and estage == "answer":
                if isinstance(edata, dict) and "files" in edata:
                    for f in edata["files"]:
                        attachments.append({"document_uuid": f.get("document_uuid", ""),
                                           "filename": f.get("filename", "")})

        answer = re.sub(r"\s*\{\{chunk:[a-f0-9-]+\}\}\s*", "", "".join(answer_parts)).strip()
        return {"answer": answer, "chunks": chunks, "attachments": attachments,
                "events_total": len(answer_parts), "raw_error": ""}


# ==================== 评测引擎 ====================
_UNIT_REPL = [
    (r"大于等于", ">="), (r"大于或等于", ">="), (r"不小于", ">="),
    (r"小于等于", "<="), (r"小于或等于", "<="), (r"不超过", "<="), (r"不多于", "<="),
    (r"大于", ">"), (r"小于", "<"),
    (r"μm", "微米"), (r"\bum\b", "微米"),
    (r"m³", "立方米"), (r"m3", "立方米"),
    (r"每立方米", "立方米"), (r"/立方米", "立方米"),
    (r"cells/ml", "细胞每毫升"), (r"cells/m?l", "细胞每毫升"),
    (r"mg", "毫克"), (r"\bkg\b", "千克"), (r"\bml\b", "毫升"),
    (r"\bg/l\b", "克每升"), (r"\bg\b", "克"),
    (r"℃", "摄氏度"), (r"°c", "摄氏度"),
    (r"%rh", "相对湿度"), (r"%RH", "相对湿度"),
    (r"≥", ">="), (r"≤", "<="),
    (r"log10", "log"), (r"\bcm\b", "厘米"),
]

_SYNONYM = {
    "本人": ["该未成年人", "未成年受试者", "受试者本人", "该受试者", "未成年人"],
    "机构": ["该试验", "该临床试验", "试验机构", "研究机构"],
    "监护人": ["法定监护人", "其监护人", "父母或监护人"],
    "知情同意": ["知情同意书", "签署知情同意", "签署同意书", "获取知情同意"],
    "非医学": ["非医学或科学", "非医学或科学的", "非医学和科学"],
    "每年": ["每满一年", "每一年", "每年一次"],
    "每满1年一次": ["每满一年提交一次", "每满1年提交一次"],
    "首次再注册前": ["直至首次再注册", "首次再注册前提交"],
    "之后每5年一次": ["之后每五年提交一次", "之后每5年提交一次"],
    "保存至少5年": ["保存期限为至少5年", "至少保存5年", "保存5年以上"],
    # ===== RJUA 医疗同义词 =====
    "复查": ["随访", "复诊", "再次就诊", "定期复查", "复查随访", "定期随访"],
    "定期复查": ["随访", "复诊", "定期随访", "定期复诊", "定期检查", "规律复查"],
    "监测": ["观察", "检测", "监控", "监视", "动态观察", "关注"],
    "结石大小及肾功能": ["结石大小和肾功能", "结石大小与肾功能", "结石及肾功能"],
    "肾功能": ["肾脏功能", "肾功", "肾的化验指标"],
    "尿常规": ["尿液分析", "尿检", "尿化验", "尿液检查"],
    "血常规": ["血象", "血液分析", "血样检查", "血液检查"],
    "B超": ["超声", "超声波", "B型超声", "彩超", "超声检查", "B超检查"],
    "CT": ["CT扫描", "CT检查", "计算机断层扫描", "腹部CT"],
    "MRI": ["磁共振", "核磁共振", "磁共振成像", "核磁"],
    "PSA": ["前列腺特异抗原", "前列腺特异性抗原", "前列腺癌抗原", "血清PSA"],
    "尿道": ["输尿管", "泌尿道", "泌尿系"],
    "血尿": ["尿中有血", "肉眼血尿", "镜下血尿", "血尿症状"],
    "尿频": ["排尿次数增多", "尿次数增多", "频繁小便"],
    "尿急": ["急着上厕所", "有尿意就憋不住"],
    "排尿困难": ["排尿费力", "排尿不畅", "排尿障碍", "小便困难"],
    "前列腺增生": ["前列腺肥大", "良性前列腺增生", "BPH"],
    "肾积水": ["肾盂积水", "肾脏积水", "积液于肾"],
    "感染": ["炎症", "发炎", "炎性反应", "感染症状"],
    "抗感染": ["抗炎", "消炎", "抗生素治疗", "抗菌治疗"],
    "手术": ["外科手术", "手术治疗", "切除手术", "根治术", "微创手术"],
    "根治术": ["根治性切除", "根治性手术", "全切", "切除手术"],
    "经尿道": ["经尿道", "尿道镜下", "膀胱镜下"],
    "垛间距≥5cm": ["垛间距不小于5厘米", "垛间距不小于5cm"],
    "地面间距≥10cm": ["地面间距不小于10厘米"],
    "墙间距≥30cm": ["墙间距不小于30厘米"],
    "≥5cm": [">=5厘米", "不小于5厘米", "不小于5cm"],
    "≥10cm": [">=10厘米", "不小于10厘米", "不小于10cm"],
    "≥30cm": [">=30厘米", "不小于30厘米", "不小于30cm"],
    "CHO": ["中国仓鼠卵巢细胞", "CHO细胞"],
    "E. coli": ["大肠杆菌", "E.coli"],
    "三步工艺名称": ["蛋白A亲和层析", "离子交换层析", "疏水层析"],
    "海藻糖或蔗糖": ["海藻糖和蔗糖", "海藻糖与蔗糖"],
    "≥4 log10 reduction": [">=4 log", "4 log reduction", "log10 reduction>=4"],
    "取得": ["获得", "获取", "得到"],
    "不得超过": ["不超过"],
    "应在": ["须在", "需要在", "必须在"],
    "须在": ["需要在", "必须在", "应在"],
    "1分钟内报警": ["1分钟内发出报警", "一分钟内发出报警"],
    "独立厂房": ["独立生产区域", "独立的生产厂房", "专用厂房"],
    "本人知情同意": ["未成年人的知情同意", "未成年人知情同意", "其知情同意"],
    "法定监护人同意": ["法定监护人的同意", "法定监护人签署", "监护人签署"],
    "非医学/科学背景": ["非医学或科学的背景", "非医学或科学的", "非医学和科学背景"],
    "独立于机构": ["独立于该机构", "独立于临床试验机构", "独立于该试验"],
    "五个模块全部列出": ["五个模块分别为", "五个模块分别是", "包含五个模块", "五个模块如下"],
    "此后每年": ["每年", "之后每年", "此后每年检测一次", "直至有效期后"],
    "新化合物": ["新的结构明确且具有药理作用的化合物", "新的化合物"],
    "1类-新化合物": ["1类创新药", "境内外均未上市的创新药"],
    "4类-境内已上市": ["境内申请人仿制已在境内上市的原研药品", "境内已上市原研药品"],
    "PRR≥2": ["PRR>=2", "PRR（比例报告比）>=2", "PRR（比例报告比）≥2"],
    "χ²≥4": ["χ²>=4", "χ²检验值>=4", "χ²检验值≥4", "χ2>=4"],
    "病例数≥3": ["病例数>=3", "n>=3"],
    "每满1年一次": ["每满一年提交一次", "每满一年一次", "每年一次"],
    "之后每5年一次": ["之后每五年提交一次", "之后每五年一次"],
    "ALCOA五项": ["Attributable", "Legible", "Contemporaneous", "Original", "Accurate", "ALCOA"],
    "+四项": ["Complete", "Consistent", "Enduring", "Available", "ALCOA+"],
    "共九项全部列出": ["9项", "9个", "九项", "9个要素", "九大要素"],
    "三项调查内容": ["调查内容包括", "调查报告应包括", "调查报告应含"],
    "24小时内": ["24小时以内", "24小时之内", "二十四小时内"],
    "15日内": ["15日内完成", "15天之内", "15天内"],
    "1年内": ["1年之内", "一年内", "一年之内"],
    "限于": ["仅限于", "只限于"],
    "商业化规模": ["商业化生产规模", "商业规模", "商业生产规模"],
    "分开设置": ["应分开", "须分开设置", "需要分开"],
}

def canonicalize(s: str) -> str:
    s = s.lower()
    for pat, rep in _UNIT_REPL:
        s = re.sub(pat, rep, s)
    return s

def normalize(s: str) -> str:
    s = canonicalize(s)
    return re.sub(r"[\s\uff0c\u3002\u3001\uff1b\uff1a\uff1f\uff01,.;:!?()\uff08\uff09\[\]\u3010\u3011\u201c\u201d\u2018\u2019`/\\\\]+", "", s)

def _split_point(s: str) -> List[str]:
    parts = re.split(r'(?<=[\u4e00-\u9fff])(?=[^\u4e00-\u9fff])|(?<=[^\u4e00-\u9fff])(?=[\u4e00-\u9fff])', s)
    return [p.strip() for p in parts if p.strip() and re.search(r'[\u4e00-\u9fff0-9a-zA-Z]', p)]

def _match_fragment(fragment: str, answer: str) -> bool:
    if fragment in answer:
        return True
    for kw, syns in _SYNONYM.items():
        if kw in fragment:
            for s in syns:
                if fragment.replace(kw, s) in answer:
                    return True
    if re.match(r"^[\u4e00-\u9fff]{3,}$", fragment):
        win = 2
        if len(fragment) < win * 2:
            return True
        total, hit = 0, 0
        for i in range(len(fragment) - win + 1):
            total += 1
            if fragment[i:i+win] in answer:
                hit += 1
        if total > 0 and hit / total >= 0.5:
            return True
        long_kws = re.findall(r"[\u4e00-\u9fff]{3,}", fragment)
        if long_kws and any(k in answer for k in long_kws):
            return True
    if re.search(r"[a-zA-Z0-9]", fragment):
        keywords = re.findall(r"[a-zA-Z]{2,}|\d+|[≥≤>≤=!]+", fragment)
        if keywords:
            hit_count = sum(1 for k in keywords if k in answer)
            if hit_count == len(keywords) or hit_count / len(keywords) >= 0.5:
                return True
    return False

def match_point(answer: str, point: str) -> bool:
    a, p = normalize(answer), normalize(point)
    if not p:
        return True
    if p in a:
        return True
    fragments = _split_point(p)
    if len(fragments) >= 2:
        if all(_match_fragment(f, a) for f in fragments):
            return True
    if len(fragments) == 1 and _match_fragment(fragments[0], a):
        return True
    a_lower, p_lower = a.lower(), p.lower()
    for kw, syns in _SYNONYM.items():
        if kw.lower() in p_lower or kw in p:
            for s in syns:
                if s.lower() in a_lower:
                    return True
    kws = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}|\d+(?:[\.\d]*)", point)
    kws = [k for k in kws if len(k) >= 2]
    if len(kws) >= 2:
        hit = sum(1 for k in kws if normalize(k) in a)
        if hit / len(kws) >= 0.7:
            return True
    return False

def extract_numbers(text: str) -> List[str]:
    cn_digit = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}
    for cn, ar in cn_digit.items():
        text = text.replace(cn, ar)
    return re.findall(r"\d+(?:\.\d+)?", text)


# ==================== LLM Judge ====================
_judge_client = None  # 平台 GPT 复用会话

def _llm_judge_api(prompt: str, cfg: Config) -> str:
    import requests
    r = requests.post(cfg.LLM_URL, headers={
        "Authorization": f"Bearer {cfg.LLM_KEY}",
        "Content-Type": "application/json"
    }, json={
        "model": cfg.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20, "temperature": 0.1,
    }, timeout=30)
    return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""

_judge_client = None  # 平台 GPT 评分专用客户端

def _llm_judge_platform(prompt: str, cfg: Config) -> str:
    """用平台 GPT-5.4 做 Judge"""
    global _judge_client
    if _judge_client is None:
        _judge_client = KBClient(cfg)
        _judge_client.login()
        _judge_client._judge_sid = _judge_client.create_session()
    return _judge_client.chat(_judge_client._judge_sid, prompt).get("answer", "")[:50].strip()

def llm_judge(question: str, standard_answer: str, predicted_answer: str, key_points: list = None, cfg: Config = None) -> str:
    if not cfg or not cfg.LLM_KEY:
        return "N/A"
    checklist = "、".join(key_points) if key_points else "无"
    prompt = (
        "你是资深医疗评审专家。请判断'智能体回答'是否在医学上正确且完整地解答了'问题'。\n\n"
        "## 评分标准\n"
        "PASS：智能体回答覆盖了\"必要检查/治疗要点\"的全部或绝大部分，且回答与\"参考答案\"在医学实质一致。\n"
        "PARTIAL：智能体回答方向正确，但遗漏了\"必要检查/治疗要点\"中的1个以上关键项，或表达不够完整。\n"
        "FAIL：智能体回答有医学错误，或几乎完全未涉及必要要点。\n\n"
        "## 重要提示\n"
        "必要检查/治疗要点是关键词级别的清单。智能体回答可能用更正式或更通俗的语言表达相同医学概念。\n"
        "只需判断医学实质是否等价，不要求字面严格匹配。\n\n"
        f"## 题目\n"
        f"问题：{question}\n"
        f"参考答案（医生口吻）：{standard_answer}\n"
        f"必要检查/治疗要点（关键词）：{checklist}\n"
        f"智能体回答：{predicted_answer}\n\n"
        "只输出一个词（PASS/PARTIAL/FAIL）："
    )
    try:
        reply = _llm_judge_platform(prompt, cfg) if cfg.USE_PLATFORM_JUDGE else _llm_judge_api(prompt, cfg)
        for v in ["PASS", "PARTIAL", "FAIL"]:
            if v in reply.strip().upper():
                return v
        return "N/A"
    except Exception:
        return "N/A"


# ==================== 评测核心 ====================
def _word_overlap(text_a: str, text_b: str) -> float:
    """text_a 覆盖了 text_b 的多少关键词 (基准=text_b)"""
    a_words = set(re.findall(r"[\u4e00-\u9fff]{2,}", normalize(text_a)))
    b_words = set(re.findall(r"[\u4e00-\u9fff]{2,}", normalize(text_b)))
    if not b_words:
        return 0.0
    return len(a_words & b_words) / len(b_words)


def _llm_faithfulness(answer: str, source_doc: str, cfg: Config) -> float:
    """RAGAS Faithfulness: 提取断言 → 逐句验证是否被源文档支持"""
    if not cfg.LLM_KEY or not answer.strip() or not source_doc.strip():
        return _word_overlap(answer, source_doc)
    prompt = f"""你是医学评测专家。判断回答中的主张是否被源文档支持。

源文档：
{source_doc[:2000]}

回答：
{answer[:2000]}

第一步：从回答中提取所有独立的医学主张/断言(Claim)，一行一条。
第二步：逐一判断每个主张是否可以由源文档推断出来。
第三步：计算忠实度分数 = 被支持的断言数 / 总断言数。

只需输出最终分数（0-1之间的小数），如：0.85"""
    try:
        reply = _llm_judge_api(prompt, cfg) if not cfg.USE_PLATFORM_JUDGE else _llm_judge_platform(prompt, cfg)
        nums = re.findall(r"\b(?:0\.\d+|1\.0+)\b", reply)
        if nums:
            return min(float(nums[0]), 1.0)
        return _word_overlap(answer, source_doc)
    except:
        return _word_overlap(answer, source_doc)


def _llm_relevancy(question: str, answer: str, cfg: Config) -> float:
    """RAGAS Relevancy: 从回答生成逆向问题 → 与原问题对比"""
    if not cfg.LLM_KEY or not answer.strip():
        return _word_overlap(answer, question)
    prompt = f"""你是评测专家。判断回答与问题的相关程度。

原问题：{question}
回答：{answer[:2000]}

第一步：从回答倒推，生成3个该回答能解答的问题(Reverse Questions)。
第二步：判断每个生成的问题与原问题在话题/意图上是否一致。
第三步：计算相关度分数 = 一致的数量 / 3。

只需输出最终分数（0-1之间的小数），如：0.67"""
    try:
        reply = _llm_judge_api(prompt, cfg) if not cfg.USE_PLATFORM_JUDGE else _llm_judge_platform(prompt, cfg)
        nums = re.findall(r"\b(?:0\.\d+|1\.0+)\b", reply)
        if nums:
            return min(float(nums[0]), 1.0)
        return _word_overlap(answer, question)
    except:
        return _word_overlap(answer, question)


def evaluate_one(qa: dict, answer_text: str, cfg: Config, source_doc: str = "") -> dict:
    std_answer = qa["standard_answer"]
    eval_points = qa["eval_points"]

    # 1. Answer Completeness — 关键词要点覆盖
    matched, missing = [], []
    for pt in eval_points:
        (matched if match_point(answer_text, pt) else missing).append(pt)
    completeness = len(matched) / len(eval_points) if eval_points else 1.0

    # 2. Faithfulness — RAGAS 风格: 提取断言→逐句验证源文档支持
    faithfulness = _llm_faithfulness(answer_text, source_doc, cfg) if source_doc else 0.0

    # 3. Answer Relevancy — RAGAS 风格: 逆向问题生成+对比
    relevancy = _llm_relevancy(qa["question"], answer_text, cfg) if qa["question"] else 1.0

    # 4. Answer Correctness — LLM Judge (多轮取中)
    has_answer = bool(answer_text and answer_text.strip())
    llm_v = "N/A"
    correctness = 0.0
    if cfg.LLM_KEY and has_answer:
        rounds = max(1, cfg.JUDGE_ROUNDS)
        votes = []
        for _ in range(rounds):
            v = llm_judge(qa["question"], std_answer, answer_text, eval_points, cfg)
            if v in ("PASS", "PARTIAL", "FAIL"):
                votes.append(v)
        if votes:
            # 多数票
            from collections import Counter
            cnt = Counter(votes)
            llm_v = cnt.most_common(1)[0][0]
            # 正确度 = 加权平均
            correctness = sum({"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}.get(v, 0) for v in votes) / len(votes)

    # 5. 数值精确性
    std_nums = set(extract_numbers(std_answer))
    pred_nums = set(extract_numbers(answer_text))
    trivial = {"0", "1", "2", "3"}
    std_nums_critical = {n for n in std_nums if n not in trivial}
    num_score = 1.0
    num_missing = []
    if std_nums_critical:
        num_hit = sum(1 for n in std_nums_critical if n in pred_nums)
        num_score = num_hit / len(std_nums_critical)
        num_missing = sorted(std_nums_critical - pred_nums)

    # 6. 综合判定
    has_critical_num_error = bool(num_missing) and num_score < 1.0
    if not has_answer:
        verdict = "FAIL"
        reason = "未获取到答案"
    elif completeness >= cfg.PASS_COVERAGE and not has_critical_num_error:
        verdict = "PASS"
        reason = ""
    elif completeness >= 0.4:
        verdict = "PARTIAL"
        reason = f"覆盖率 {completeness:.0%}"
    else:
        verdict = "FAIL"
        reason = f"覆盖率 {completeness:.0%}"

    # LLM Judge 覆盖最终判定 & 完整度
    if llm_v in ("PASS", "PARTIAL", "FAIL"):
        verdict = llm_v
    # 完整度也复用 LLM 语义判断（不再纯靠关键词）
    if llm_v == "PASS":
        completeness = max(completeness, 1.0)
    elif llm_v == "PARTIAL":
        completeness = max(completeness, 0.5)

    return {
        "q_id": qa["q_id"], "question": qa["question"],
        "category": qa["category"], "difficulty": qa["difficulty"],
        "answer_type": qa["answer_type"],
        "standard_answer": std_answer, "predicted_answer": answer_text,
        "completeness": round(completeness, 3),
        "faithfulness": round(faithfulness, 3),
        "relevancy": round(relevancy, 3),
        "correctness": round(correctness, 3),
        "matched_points": matched, "missing_points": missing,
        "num_score": round(num_score, 3), "num_missing": num_missing,
        "verdict": verdict, "reason": reason,
        "llm_verdict": llm_v,
    }


# ==================== RJUA 数据加载 ====================
def load_rjua_qa(xlsx_path: str) -> List[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["问题列表"]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h).strip() for h in rows[0]]
    _SKIP = ["内容描述准确"]  # 只跳过这一项不可量化的点
    qa_list = []
    for row in rows[1:]:
        if not row[0]:
            continue
        item = dict(zip(headers, [str(v) if v else "" for v in row]))
        advice = item.get("诊疗建议", "")
        raw_points = [p.strip() for p in re.split(r"[,，、]", advice) if p.strip()]
        eval_points = [p for p in raw_points if p not in _SKIP]
        for skipped in set(raw_points) - set(eval_points):
            print(f"  [跳过宽泛] {item['题号']}: {skipped}")
        qa_list.append({
            "q_id": f"Q{str(item['题号']).zfill(3)}",
            "question": item["问题"],
            "standard_answer": item["参考答案"],
            "source_doc": item["数据ID"],
            "source_title": item.get("诊断疾病", ""),
            "category": item.get("诊断疾病", ""),
            "difficulty": "基础",
            "answer_type": "诊疗",
            "eval_points": eval_points,
        })
    return qa_list


# ==================== 主流程 ====================
def main():
    cfg = Config()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not cfg.USERNAME:
        print("错误: 请先配置 .env 文件 (参考 .env.example)")
        print("至少需要: PLATFORM_URL, PLATFORM_USERNAME, PLATFORM_PASSWORD, KB_UUID")
        sys.exit(1)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    resume_path = os.path.join(cfg.OUTPUT_DIR, "results_partial.json")

    # 1. 加载数据
    qa_xlsx = cfg.QA_XLSX if os.path.isabs(cfg.QA_XLSX) else os.path.join(script_dir, cfg.QA_XLSX)
    docs_txt = cfg.DOCS_TXT if os.path.isabs(cfg.DOCS_TXT) else os.path.join(script_dir, cfg.DOCS_TXT)
    print(f"加载 QA: {qa_xlsx}")
    qa_list = load_rjua_qa(qa_xlsx)
    print(f"  共 {len(qa_list)} 题")

    # 加载源文档(用于 Faithfulness 评估)
    docs = {}
    try:
        with open(docs_txt, "r", encoding="utf-8") as f:
            raw = f.read()
        parts = re.split(r"=== 文档 \d+ \| ID: (\d+) ===", raw)
        if len(parts) > 2:
            for i in range(1, len(parts), 2):
                doc_id = parts[i].strip()
                body = parts[i+1].strip()
                body = re.sub(r"^对应疾病:.*\n?---?", "", body).strip()
                docs[doc_id] = {"id": doc_id, "body": body}
        print(f"  共 {len(docs)} 篇源文档")
    except Exception as e:
        print(f"  源文档加载失败: {e}")
    print(f"LLM Judge: {cfg.LLM_MODEL}" + (" (已启用)" if cfg.LLM_KEY else " (未启用)") + "\n")

    # 2. 断点续跑
    results: List[dict] = []
    start_idx = 0
    if os.path.exists(resume_path):
        with open(resume_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"恢复 {start_idx} 题已完成\n")

    # 3. 登录
    print(f"登录: {cfg.BASE_URL}")
    client = KBClient(cfg)
    client.login()
    print("  登录成功\n")

    # 4. 逐题评测
    print(f"{'='*60}\n开始评测 ({cfg.EFFORT})\n{'='*60}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(start_idx, len(qa_list)):
        qa = qa_list[i]
        print(f"\n[{i+1}/{len(qa_list)}] {qa['q_id']} [{qa['category'][:12]}]")
        print(f"  问题: {qa['question'][:60]}...")

        try:
            sid = client.create_session()
        except Exception as e:
            print(f"  会话失败: {e}")
            continue

        r = client.chat(sid, qa["question"])
        if r["raw_error"]:
            print(f"  答案失败: {r['raw_error'][:60]}")
            continue

        # 取对应源文档内容(用于 Faithfulness 计算)
        doc_text = ""
        if qa.get("source_doc"):
            for d in docs.values():
                if d.get("id") == qa["source_doc"]:
                    doc_text = d.get("body", "")
                    break

        ev = evaluate_one(qa, r["answer"], cfg, doc_text)
        results.append(ev)

        tag = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}
        print(f"  {tag[ev['verdict']]} {ev['verdict']} (LLM评估)")
        print(f"  完整={ev['completeness']:.0%} 忠实={ev['faithfulness']:.0%} 相关={ev['relevancy']:.0%} 正确={ev['correctness']:.0%}")
        if ev["missing_points"]:
            print(f"  缺失: {ev['missing_points'][:3]}")

        with open(resume_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        time.sleep(cfg.DELAY_BETWEEN)

    # 5. 统计
    total = len(results)
    if total == 0:
        print("\n警告: 无任何结果, 请检查网络和登录")
        return
    passed = sum(1 for r in results if r.get("verdict") == "PASS")
    partial_cnt = sum(1 for r in results if r.get("verdict") == "PARTIAL")
    failed = sum(1 for r in results if r.get("verdict") == "FAIL")
    avg_completeness = sum(r.get("completeness", 0) for r in results) / total
    avg_faithfulness = sum(r.get("faithfulness", 0) for r in results) / total
    avg_relevancy = sum(r.get("relevancy", 0) for r in results) / total
    avg_correctness = sum(r.get("correctness", 0) for r in results) / total

    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"判定: PASS={passed} ({passed/total*100:.1f}%)  PARTIAL={partial_cnt}  FAIL={failed}")
    print(f"四维均值: 完整={avg_completeness:.0%} 忠实={avg_faithfulness:.0%} 相关={avg_relevancy:.0%} 正确={avg_correctness:.0%}")

    # 6. 保存报告
    report = {
        "meta": {"platform": cfg.BASE_URL, "kb_uuid": cfg.KB_UUID,
                 "timestamp": datetime.now().isoformat(), "dataset": "RJUA-QA (泌尿科)",
                 "total": total, "passed": passed, "partial": partial_cnt, "failed": failed,
                 "effort": cfg.EFFORT, "llm_model": cfg.LLM_MODEL if cfg.LLM_KEY else "N/A"},
        "overall": {"accuracy_pass_only": round(passed/total, 4),
                    "avg_completeness": round(avg_completeness, 4),
                    "avg_faithfulness": round(avg_faithfulness, 4),
                    "avg_relevancy": round(avg_relevancy, 4),
                    "avg_correctness": round(avg_correctness, 4)},
        "details": results,
    }

    json_path = os.path.join(cfg.OUTPUT_DIR, f"rjua_report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Excel 摘要
    try:
        import pandas as pd
        rows = []
        for r in results:
            rows.append({"题号": r["q_id"], "判定": r["verdict"],
                         "覆盖率": f"{r['coverage']*100:.0f}%",
                         "缺失要点": "\n".join(r["missing_points"]),
                         "LLM评分": r.get("llm_verdict", "N/A"),
                         "问题": r["question"], "预测答案": r["predicted_answer"][:200]})
        pd.DataFrame(rows).to_excel(json_path.replace(".json", ".xlsx"), index=False)
    except Exception:
        pass

    print(f"报告: {json_path}")
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
