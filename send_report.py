import os
import json
import argparse
import requests
from typing import Dict, List


class OpenCompassResultSender:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.headers = {"Content-Type": "application/json"}
        self.default_root = os.path.join("outputs", "default")

    def _get_latest_output_dir(self) -> str:
        """自动查找 outputs/default/ 下日期最新的子目录（命名格式：YYYYMMDD_HHMMSS）"""
        if not os.path.exists(self.default_root):
            raise FileNotFoundError(f"根目录不存在：{self.default_root}")

        subdirs = [
            d for d in os.listdir(self.default_root)
            if os.path.isdir(os.path.join(self.default_root, d))
               and len(d.split("_")) == 2  # 匹配 "日期_时间" 格式
        ]

        if not subdirs:
            raise FileNotFoundError(f"{self.default_root} 下无评估结果子目录")

        latest_dir = sorted(subdirs)[-1]
        latest_path = os.path.abspath(os.path.join(self.default_root, latest_dir))
        print(f"自动识别最新评估目录：{latest_path}")
        return latest_path

    def _read_summary_md(self, output_dir: str) -> str:
        """读取 summary/summary_YYYYMMDD_HHMMSS.md 文件内容（文件名与目录名匹配）"""
        # 提取目录的日期时间（如 "20251024_165813"）
        dir_datetime = os.path.basename(output_dir)
        # 构建 md 文件路径
        md_path = os.path.join(output_dir, "summary", f"summary_{dir_datetime}.md")

        if not os.path.exists(md_path):
            return f"⚠️ 未找到 summary 文件：{md_path}"

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read(2000)  # 限制读取前2000字符，避免消息过长
            # 替换 md 中的标题符号（飞书卡片兼容处理）
            content = content.replace("#", "##").replace("**", "**")
            return f"### 评估详情\n{content}\n\n...（内容已截断，完整文件见输出目录）"
        except Exception as e:
            return f"❌ 读取 summary 文件失败：{str(e)}"

    def _parse_eval_results(self, output_dir: str) -> Dict:
        """解析评估结果，新增 md 文件内容"""
        results = {
            "model": "未知模型",
            "tasks": [],
            "summary": "评估完成，但未找到详细结果",
            "output_dir": output_dir,
            "md_content": ""  # 新增：存储 md 文件内容
        }

        # 读取 summary.md 内容（优先执行）
        results["md_content"] = self._read_summary_md(output_dir)

        # 解析 summary.json（保持原有逻辑）
        summary_path = os.path.join(output_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                results["model"] = next(iter(summary_data.keys())) if summary_data else "未知模型"
                model_metrics = summary_data.get(results["model"], {})
                for task, metrics in model_metrics.items():
                    if "flores_100" in task:
                        bleu_score = metrics.get("bleu", 0)
                        chrf_score = metrics.get("chrf", 0)
                        results["tasks"].append({
                            "task": task,
                            "bleu": f"{bleu_score:.2f}" if bleu_score else "无",
                            "chrf": f"{chrf_score:.2f}" if chrf_score else "无"
                        })
                results["summary"] = f"共完成 {len(results['tasks'])} 个任务评估"
            except Exception as e:
                results["summary"] = f"解析summary.json失败：{str(e)}"
        else:
            dir_name = os.path.basename(output_dir)
            results["summary"] = f"未找到summary.json（可能评估未完成）"
            results["model"] = dir_name.split("_")[0] if "_" in dir_name else "未知模型"

        return results

    def _build_feishu_card(self, results: Dict) -> Dict:
        """构建飞书卡片，整合 md 文件内容"""
        task_elements = []
        if results["tasks"]:
            task_elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "### 任务得分（BLEU / chrF）"}})
            for task in results["tasks"]:
                task_elements.append({
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"- **{task['task']}**: BLEU {task['bleu']} | chrF {task['chrf']}"
                    }
                })
        else:
            task_elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "- 无有效任务得分"}})

        # 新增：添加 md 文件内容到卡片
        md_elements = [
            {"tag": "hr"},  # 分隔线
            {"tag": "div", "text": {"tag": "lark_md", "content": results["md_content"]}}
        ]

        return {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "📊 OpenCompass 评估结果通知"},
                    "template": "blue"
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**模型名称**: {results['model']}"}},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**评估状态**: {results['summary']}"}},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**输出目录**: `{results['output_dir']}`"}},
                    *task_elements,
                    *md_elements,  # 整合 md 内容
                    {"tag": "div", "text": {"tag": "lark_md", "content": "---\n*该通知由自动脚本发送*"}}
                ]
            }
        }

    def send(self, output_dir: str = None) -> bool:
        try:
            if not output_dir:
                output_dir = self._get_latest_output_dir()

            results = self._parse_eval_results(output_dir)
            card = self._build_feishu_card(results)
            response = requests.post(
                url=self.webhook_url,
                headers=self.headers,
                data=json.dumps(card),
                timeout=10
            )
            response.raise_for_status()
            print(f"✅ 飞书消息发送成功（目录：{output_dir}）")
            return True
        except Exception as e:
            print(f"❌ 飞书消息发送失败：{str(e)}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="发送OpenCompass评估结果到飞书（含summary.md内容）")
    parser.add_argument(
        "--webhook", 
        default="https://open.feishu.cn/open-apis/bot/v2/hook/4d1d3786-e01b-4d46-8feb-bfd9a5191525",
        required=False, 
        help="飞书群机器人Webhook地址"
    )
    parser.add_argument(
        "--output_dir", 
        required=False, 
        help="手动指定评估输出目录（默认自动查找最新）"
    )
    args = parser.parse_args()

    sender = OpenCompassResultSender(webhook_url=args.webhook)
    sender.send(output_dir=args.output_dir)
