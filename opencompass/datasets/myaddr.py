import json
import os
import re
import difflib
from datasets import Dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess
from .base import BaseDataset


class AddrDesensitizeDataset(BaseDataset):
    @staticmethod
    def load(path, split='dev'):
        root_path = get_data_path(path)
        data_file = os.path.join(root_path, f'addr_desensitize_{split}.jsonl')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"未找到目标文件：{data_file}")

        dataset_list = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                row = json.loads(line_stripped)
                assert 'id' in row and 'original_text' in row and 'gold_text' in row and 'address_info' in row
                dataset_list.append(row)

        return Dataset.from_list(dataset_list)


class AddrDesensitizeEvaluator(BaseEvaluator):
    # 去除模型“思考/推理”部分，仅保留最终答案
    def _strip_think(self, text: str) -> str:
        s = text or ""
        # 1) 去除 <think>...</think>、<analysis>...</analysis> 等标签内容
        tag_names = [
            'think', 'analysis', 'reasoning', 'scratchpad',
            'reflection', 'chain_of_thought', 'cot'
        ]
        for tag in tag_names:
            s = re.sub(fr'<{tag}[\s>][\s\S]*?</{tag}>', '', s, flags=re.IGNORECASE)

        # 2) 去除 ```think ...```、```analysis ...``` 等代码围栏块
        fence_names = '|'.join([
            'think', 'analysis', 'reasoning', 'reflection', 'scratchpad', 'cot'
        ])
        s = re.sub(fr'```(?:{fence_names})[\s\S]*?```', '', s, flags=re.IGNORECASE)

        # 3) 可选：去除常见前缀段直到空行（保守处理，避免误删正常文本）
        # 例如 “思考：...（若干行）\n\n最终答案...”
        prefix_patterns = [
            r'^\s*(思考|推理|分析|Chain[-\s]?of[-\s]?Thought|Reasoning|Analysis|Thoughts)\s*[:：][\s\S]*?\n\s*\n'
        ]
        for pat in prefix_patterns:
            s = re.sub(pat, '', s, flags=re.IGNORECASE | re.MULTILINE)

        return s.strip()

    # 计算 addr_clean 在 pred_clean 中的最长公共子串比例（相对 addr_clean 长度）
    def _lcs_ratio(self, addr_clean: str, pred_clean: str) -> float:
        if not addr_clean:
            return 0.0
        matcher = difflib.SequenceMatcher(None, addr_clean, pred_clean)
        match = matcher.find_longest_match(0, len(addr_clean), 0, len(pred_clean))
        return (match.size / len(addr_clean)) if len(addr_clean) > 0 else 0.0

    # 归一化常见占位符为统一 token
    def _normalize_placeholders(self, text: str) -> str:
        s = text
        s = re.sub(r'\[[^\]]*?\]', '<mask_token>', s)       # [隐藏地址] 等
        s = re.sub(r'\*{2,}', '<mask_token>', s)            # ******
        s = re.sub(r'\b(address|location|mask|masked)\b', '<mask_token>', s, flags=re.IGNORECASE)
        s = re.sub(r'(隐藏地址|地址已隐藏|地址信息已隐藏|已脱敏)', '<mask_token>', s)
        return s

    def _count_placeholders(self, text: str) -> int:
        s = self._normalize_placeholders(general_postprocess(text))
        return s.count('<mask_token>')

    # 提取非地址内容：移除原始地址和占位符
    def _extract_non_addr_content(self, text: str, address_list):
        s = general_postprocess(text).lower()
        s = self._normalize_placeholders(s)
        for addr in address_list:
            addr_clean = general_postprocess(addr).lower()
            if addr_clean:
                s = s.replace(addr_clean, '')
        s = s.replace('<mask_token>', '')
        return s.strip()

    # references 为 gold_text 时的备用：移除占位符即可
    def _extract_non_addr_from_gold(self, gold_text: str):
        s = general_postprocess(gold_text).lower()
        s = self._normalize_placeholders(s)
        s = s.replace('<mask_token>', '')
        return s.strip()

    # 单个地址块评分（当我们知道原始地址时）
    def _score_one_addr(self, pred_clean: str, addr_clean: str, lcs_threshold: float = 0.4) -> int:
        if addr_clean and addr_clean in pred_clean:
            return 0
        ratio = self._lcs_ratio(addr_clean, pred_clean)
        if ratio >= lcs_threshold and len(addr_clean) >= 3:
            return 5
        return 10

    # references 是 gold_text 时：按占位符数量判断是否全部替换（不因非地址差异降 ADI）
    def _score_by_gold(self, pred: str, gold: str) -> int:
        pred_clean = general_postprocess(pred)
        gold_clean = general_postprocess(gold)

        if pred_clean == gold_clean:
            return 10

        pred_masks = self._count_placeholders(pred_clean)
        gold_masks = self._count_placeholders(gold_clean)

        if gold_masks == 0:
            return 10 if pred_clean == gold_clean else 5

        if pred_masks == gold_masks and pred_masks > 0:
            return 10
        elif pred_masks > 0:
            return 5
        else:
            return 0

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        total_addr_blocks = 0
        total_adi_points = 0
        consistent_sample_num = 0
        total_sample_num = len(predictions)

        for pred, ref in zip(predictions, references):
            # 先剔除“思考/推理”部分，仅保留最终文本
            pred_final = self._strip_think(pred)

            # 兼容两种参考格式：
            # A: (address_info, original_text)
            # B: gold_text
            if isinstance(ref, (list, tuple)) and len(ref) >= 2:
                address_list = ref[0]
                original_text = ref[1]
                pred_clean = general_postprocess(pred_final).lower()

                # ADI：对每个地址块打分
                for addr in address_list:
                    addr_clean = general_postprocess(addr).lower()
                    if not addr_clean:
                        continue
                    block_score = self._score_one_addr(pred_clean, addr_clean, lcs_threshold=0.4)
                    total_adi_points += block_score
                    total_addr_blocks += 1

                # NEC：非地址内容严格一致
                ori_non_addr = self._extract_non_addr_content(original_text, address_list)
                pred_non_addr = self._extract_non_addr_content(pred_final, address_list)
                if ori_non_addr == pred_non_addr:
                    consistent_sample_num += 1

            else:
                # references 是 gold_text（标准答案）
                gold_text = ref
                block_score = self._score_by_gold(pred_final, gold_text)
                total_adi_points += block_score
                total_addr_blocks += 1

                pred_non_addr = self._extract_non_addr_from_gold(pred_final)
                gold_non_addr = self._extract_non_addr_from_gold(gold_text)
                if pred_non_addr == gold_non_addr:
                    consistent_sample_num += 1

        adi_percent = (total_adi_points / (total_addr_blocks * 10)) * 100 if total_addr_blocks > 0 else 0.0
        nec_percent = (consistent_sample_num / total_sample_num) * 100 if total_sample_num > 0 else 0.0

        return {
            "addr_desensitization_integrity_simple": round(adi_percent, 2),
            "non_addr_content_exact_consistency": round(nec_percent, 2)
        }
